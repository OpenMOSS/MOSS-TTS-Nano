from __future__ import annotations

import importlib
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM

from moss_tts_nano.defaults import (
    DEFAULT_AUDIO_TOKENIZER_PATH,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT_AUDIO_DIR,
)

MOSS_AUDIO_TOKENIZER_TYPE = "moss-audio-tokenizer-nano"

_DEFAULT_VOICE_FILES: dict[str, tuple[str, str]] = {
    "Junhao": ("zh_1.wav", "Chinese male voice A"),
    "Xiaoyu": ("zh_3.wav", "Chinese female voice A"),
    "Yuewen": ("zh_4.wav", "Chinese female voice B"),
    "Lingyu": ("zh_6.wav", "Chinese female voice C"),
    "Ava": ("en_2.wav", "English female voice A"),
    "Bella": ("en_3.wav", "English female voice B"),
    "Adam": ("en_4.wav", "English male voice A"),
    "Yui": ("jp_2.wav", "Japanese female voice B"),
    "男播音": ("zh_10.wav", "Chinese male broadcaster voice"),
    "杨幂": ("zh_11.wav", "Chinese female voice YangMi style"),
}

DEFAULT_VOICE = "Junhao"
FLASH_ATTENTION_DTYPES = {torch.float16, torch.bfloat16}


@dataclass(frozen=True)
class VoicePreset:
    name: str
    prompt_audio_path: Path
    description: str


def build_default_voice_presets() -> dict[str, VoicePreset]:
    presets: dict[str, VoicePreset] = {}
    for voice_name, (file_name, description) in _DEFAULT_VOICE_FILES.items():
        prompt_path = (DEFAULT_PROMPT_AUDIO_DIR / file_name).resolve()
        presets[voice_name] = VoicePreset(
            name=voice_name,
            prompt_audio_path=prompt_path,
            description=description,
        )
    return presets


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def waveform_to_numpy(waveform: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(waveform):
        array = waveform.detach().cpu().numpy()
    else:
        array = np.asarray(waveform)
    if array.ndim == 1:
        return array.astype(np.float32, copy=False)
    if array.ndim != 2:
        raise ValueError(f"Unsupported waveform shape: {array.shape}")
    if array.shape[0] <= 8 and array.shape[0] < array.shape[1]:
        array = array.T
    return array.astype(np.float32, copy=False)


def _resolve_model_source(value: str | Path) -> str:
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    raw = str(value).strip()
    if not raw:
        raise ValueError("model source must not be empty")
    expanded = Path(raw).expanduser()
    if expanded.exists():
        return str(expanded.resolve())
    return raw


def _existing_local_model_path(value: str) -> Path | None:
    expanded = Path(value).expanduser()
    if expanded.exists():
        return expanded.resolve()
    return None


@lru_cache(maxsize=1)
def _has_flash_attn() -> bool:
    try:
        importlib.import_module("flash_attn")
    except Exception:
        return False
    return True


class NanoTTSService:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        audio_tokenizer_path: str | Path = DEFAULT_AUDIO_TOKENIZER_PATH,
        device: str = "auto",
        dtype: str = "auto",
        attn_implementation: str = "auto",
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        voice_presets: Optional[dict[str, VoicePreset]] = None,
    ) -> None:
        self.checkpoint_path = _resolve_model_source(checkpoint_path)
        self.audio_tokenizer_path = _resolve_model_source(audio_tokenizer_path)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.voice_presets = voice_presets or build_default_voice_presets()
        self.default_voice = DEFAULT_VOICE if DEFAULT_VOICE in self.voice_presets else next(iter(self.voice_presets))

        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        self.attn_implementation = self._resolve_attn_implementation(attn_implementation)

        self._lock = threading.RLock()
        self._model = None
        self._audio_tokenizer = None
        self._checkpoint_global_attn_implementation: str | None = None
        self._checkpoint_local_attn_implementation: str | None = None
        self._configured_global_attn_implementation: str | None = None
        self._configured_local_attn_implementation: str | None = None
        self._configured_audio_tokenizer_attn_implementation: str | None = None
        self._configured_audio_tokenizer_compute_dtype: str | None = None

    def _can_use_flash_attention(self) -> bool:
        return self.device.type == "cuda" and self.dtype in FLASH_ATTENTION_DTYPES and _has_flash_attn()

    def _resolve_runtime_default_attn_implementation(self) -> str:
        return "flash_attention_2" if self._can_use_flash_attention() else "sdpa"

    def _resolve_attn_implementation(self, requested: str | None) -> str | None:
        normalized = str(requested).strip().lower() if requested is not None else "auto"
        if not normalized or normalized in {"auto", "default", "model_default"}:
            return None
        if normalized not in {"sdpa", "flash_attention_2", "eager"}:
            raise ValueError(
                "attn_implementation must be one of: model_default/auto, sdpa, flash_attention_2, eager"
            )
        if normalized == "flash_attention_2":
            if not self._can_use_flash_attention():
                logging.warning(
                    "flash_attention_2 requires CUDA, flash_attn, and fp16/bf16; falling back to sdpa "
                    "(device=%s dtype=%s flash_attn=%s)",
                    self.device,
                    self.dtype,
                    _has_flash_attn(),
                )
                return "sdpa"
        return normalized

    @staticmethod
    def _normalize_loaded_attn_implementation(attn_implementation: object) -> str:
        normalized = str(attn_implementation).strip().lower() if attn_implementation is not None else ""
        if not normalized or normalized == "none":
            return "eager"
        return normalized

    def _resolve_request_attention_implementation(
        self,
        requested: Optional[str],
    ) -> tuple[str, str, str]:
        normalized = str(requested).strip().lower() if requested is not None else ""
        resolved = self._resolve_attn_implementation(normalized)
        if resolved is not None:
            return normalized, resolved, resolved

        if self.attn_implementation is not None:
            return self.attn_implementation, self.attn_implementation, self.attn_implementation

        runtime_default = self._resolve_runtime_default_attn_implementation()
        return "auto", runtime_default, runtime_default

    @staticmethod
    def _resolve_codec_attention_implementation(tts_attn_implementation: str) -> str:
        return "flash_attention_2" if tts_attn_implementation == "flash_attention_2" else "sdpa"

    def _resolve_codec_compute_dtype(self, codec_attn_implementation: str) -> str:
        if codec_attn_implementation == "flash_attention_2":
            return "bf16" if self.dtype == torch.bfloat16 else "fp16"
        return "fp32"

    @staticmethod
    def _apply_model_attention_implementation(model, *, global_attn: str, local_attn: str) -> None:
        if hasattr(model, "_set_attention_implementation"):
            model._set_attention_implementation(global_attn, local_attn_implementation=local_attn)

    def _install_stream_decode_budget_patch(self, model) -> None:
        if self.device.type != "cuda":
            return

        model_cls = model.__class__
        if getattr(model_cls, "_nanotts_stream_decode_budget_patch_installed", False):
            return

        compute_stream_lead = getattr(model_cls, "_compute_stream_lead_seconds", None)
        resolve_stream_budget = getattr(model_cls, "_resolve_stream_decode_frame_budget", None)
        if not callable(compute_stream_lead) or not callable(resolve_stream_budget):
            return

        def _patched_resolve_stream_decode_frame_budget(
            *,
            emitted_samples_total: int,
            sample_rate: int,
            first_audio_emitted_at,
        ) -> int:
            # The upstream streaming policy starts with one decode frame
            # (about 80 ms audio), which makes CUDA realtime decode emit many
            # tiny chunks and underrun browser playback on this checkpoint.
            lead_seconds = compute_stream_lead(
                emitted_samples_total=emitted_samples_total,
                sample_rate=sample_rate,
                first_audio_emitted_at=first_audio_emitted_at,
            )
            if first_audio_emitted_at is None or lead_seconds < 0.20:
                return 4
            if lead_seconds < 0.55:
                return 6
            if lead_seconds < 1.10:
                return 8
            return 12

        model_cls._nanotts_original_resolve_stream_decode_frame_budget = resolve_stream_budget
        model_cls._resolve_stream_decode_frame_budget = staticmethod(_patched_resolve_stream_decode_frame_budget)
        model_cls._nanotts_stream_decode_budget_patch_installed = True
        logging.info("installed Nano-TTS CUDA streaming decode budget patch")

    def _discard_loaded_model_locked(self, reason: str) -> None:
        if self._model is None:
            return
        logging.warning("discarding loaded Nano-TTS model state: %s", reason)
        self._model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _discard_loaded_audio_tokenizer_locked(self, reason: str) -> None:
        if self._audio_tokenizer is None:
            return
        logging.warning("discarding loaded Nano-TTS audio tokenizer state: %s", reason)
        self._audio_tokenizer = None
        self._configured_audio_tokenizer_attn_implementation = None
        self._configured_audio_tokenizer_compute_dtype = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _restore_model_execution_state(self, model):
        current_parameter = next(model.parameters(), None)
        if current_parameter is None or current_parameter.dtype == self.dtype:
            return model
        self._discard_loaded_model_locked(
            f"current_dtype={current_parameter.dtype} expected_dtype={self.dtype}; reloading checkpoint"
        )
        return self._load_model_locked()

    def _read_model_attention_implementation(self, model) -> tuple[str, str]:
        global_attn = self._normalize_loaded_attn_implementation(
            getattr(getattr(model, "transformer", None), "attn_implementation", None)
        )
        local_attn = self._normalize_loaded_attn_implementation(
            getattr(getattr(model, "local_transformer", None), "attn_implementation", None)
        )
        return global_attn, local_attn

    def _ensure_paths(self) -> None:
        checkpoint_path = _existing_local_model_path(self.checkpoint_path)
        if checkpoint_path is not None and not checkpoint_path.exists():
            raise FileNotFoundError(f"Nano-TTS checkpoint not found: {checkpoint_path}")
        audio_tokenizer_path = _existing_local_model_path(self.audio_tokenizer_path)
        if audio_tokenizer_path is not None and not audio_tokenizer_path.exists():
            raise FileNotFoundError(f"Audio tokenizer checkpoint not found: {audio_tokenizer_path}")

    def _load_audio_tokenizer_locked(self, *, tts_attn_implementation: str):
        codec_attn_implementation = self._resolve_codec_attention_implementation(tts_attn_implementation)
        codec_compute_dtype = self._resolve_codec_compute_dtype(codec_attn_implementation)

        if self._audio_tokenizer is None:
            logging.info(
                "loading Nano-TTS audio tokenizer checkpoint=%s device=%s attn=%s compute_dtype=%s",
                self.audio_tokenizer_path,
                self.device,
                codec_attn_implementation,
                codec_compute_dtype,
            )
            audio_tokenizer = AutoModel.from_pretrained(
                self.audio_tokenizer_path,
                trust_remote_code=True,
                local_files_only=_existing_local_model_path(self.audio_tokenizer_path) is not None,
            )
            if hasattr(audio_tokenizer, "eval"):
                audio_tokenizer.eval()
            self._audio_tokenizer = audio_tokenizer

        audio_tokenizer = self._audio_tokenizer
        if hasattr(audio_tokenizer, "to"):
            audio_tokenizer = audio_tokenizer.to(self.device)
        if hasattr(audio_tokenizer, "set_attention_implementation"):
            audio_tokenizer.set_attention_implementation(codec_attn_implementation)
        if hasattr(audio_tokenizer, "set_compute_dtype"):
            audio_tokenizer.set_compute_dtype(codec_compute_dtype)
        if hasattr(audio_tokenizer, "eval"):
            audio_tokenizer.eval()

        self._audio_tokenizer = audio_tokenizer
        self._configured_audio_tokenizer_attn_implementation = codec_attn_implementation
        self._configured_audio_tokenizer_compute_dtype = codec_compute_dtype
        return self._audio_tokenizer

    def _load_model_locked(self):
        if self._model is not None:
            return self._model

        self._ensure_paths()
        logging.info(
            "loading Nano-TTS checkpoint=%s audio_tokenizer=%s device=%s dtype=%s attn=%s",
            self.checkpoint_path,
            self.audio_tokenizer_path,
            self.device,
            self.dtype,
            self.attn_implementation or "model_default",
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            trust_remote_code=True,
            local_files_only=_existing_local_model_path(self.checkpoint_path) is not None,
        )
        model.to(device=self.device, dtype=self.dtype)
        self._checkpoint_global_attn_implementation, self._checkpoint_local_attn_implementation = (
            self._read_model_attention_implementation(model)
        )
        _, default_global_attn, default_local_attn = self._resolve_request_attention_implementation(None)
        self._apply_model_attention_implementation(
            model,
            global_attn=default_global_attn,
            local_attn=default_local_attn,
        )
        self._install_stream_decode_budget_patch(model)
        model.eval()
        self._configured_global_attn_implementation, self._configured_local_attn_implementation = (
            self._read_model_attention_implementation(model)
        )
        self._model = model
        return self._model

    def get_model(self):
        with self._lock:
            return self._load_model_locked()

    def split_voice_clone_text(
        self,
        *,
        text: str,
        voice_clone_max_text_tokens: int = 75,
    ) -> list[str]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return []

        try:
            max_tokens = int(voice_clone_max_text_tokens)
        except Exception:
            max_tokens = 75
        if max_tokens <= 0:
            return [normalized_text]

        with self._lock:
            model = self._load_model_locked()
            if not hasattr(model, "_load_text_tokenizer") or not hasattr(model, "_split_text_into_best_sentences"):
                return [normalized_text]

            text_tokenizer = model._load_text_tokenizer(
                text_tokenizer=None,
                text_tokenizer_path=self.checkpoint_path,
            )
            split_chunks = model._split_text_into_best_sentences(
                text_tokenizer=text_tokenizer,
                text=normalized_text,
                max_tokens=max_tokens,
            )

        effective_chunks = split_chunks if len(split_chunks) > 1 else [normalized_text]
        cleaned_chunks = [str(chunk).strip() for chunk in effective_chunks if str(chunk).strip()]
        return cleaned_chunks or [normalized_text]

    def list_voice_names(self) -> list[str]:
        return list(self.voice_presets.keys())

    def get_voice_preset(self, voice_name: Optional[str]) -> VoicePreset:
        if voice_name and voice_name in self.voice_presets:
            return self.voice_presets[voice_name]
        return self.voice_presets[self.default_voice]

    def resolve_prompt_audio_path(
        self,
        *,
        voice: Optional[str] = None,
        prompt_audio_path: Optional[str | Path] = None,
    ) -> Path:
        if prompt_audio_path:
            resolved = Path(prompt_audio_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Prompt audio not found: {resolved}")
            return resolved

        preset = self.get_voice_preset(voice)
        if not preset.prompt_audio_path.exists():
            raise FileNotFoundError(f"Voice preset prompt audio not found: {preset.prompt_audio_path}")
        return preset.prompt_audio_path

    def preload(self, *, voices: Optional[list[str]] = None, load_model: bool = True) -> dict[str, object]:
        loaded_voices: list[str] = []
        if load_model:
            self.get_model()
        for voice_name in voices or [self.default_voice]:
            preset = self.get_voice_preset(voice_name)
            if preset.prompt_audio_path.exists():
                loaded_voices.append(preset.name)
        return {
            "loaded_voices": loaded_voices,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "attn_implementation": self.attn_implementation or "auto",
            "checkpoint_default_attn_implementation": self._checkpoint_global_attn_implementation or "eager",
            "checkpoint_default_local_attn_implementation": self._checkpoint_local_attn_implementation or "eager",
            "configured_attn_implementation": self._configured_global_attn_implementation or "eager",
            "configured_local_attn_implementation": self._configured_local_attn_implementation or "eager",
            "configured_codec_attn_implementation": self._configured_audio_tokenizer_attn_implementation or "unknown",
            "configured_codec_compute_dtype": self._configured_audio_tokenizer_compute_dtype or "unknown",
        }

    def _build_output_path(self, prefix: str) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        random_suffix = uuid.uuid4().hex[:8]
        return self.output_dir / f"{prefix}_{timestamp}_{random_suffix}.wav"

    def synthesize(
        self,
        *,
        text: str,
        voice: Optional[str] = None,
        mode: str = "voice_clone",
        output_audio_path: Optional[str | Path] = None,
        prompt_audio_path: Optional[str | Path] = None,
        prompt_text: Optional[str] = None,
        max_new_frames: int = 375,
        voice_clone_max_text_tokens: int = 75,
        voice_clone_max_memory_per_sample_gb: float = 1.0,
        tts_max_batch_size: int = 0,
        codec_max_batch_size: int = 0,
        do_sample: bool = True,
        text_temperature: float = 1.0,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 0.8,
        audio_top_p: float = 0.95,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.2,
        nq: Optional[int] = None,
        seed: Optional[int] = None,
        attn_implementation: Optional[str] = None,
    ) -> dict[str, object]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise ValueError("text is required")

        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"continuation", "voice_clone"}:
            raise ValueError("mode must be either 'continuation' or 'voice_clone'")

        effective_prompt_audio_path: Optional[Path] = None
        resolved_voice = self.get_voice_preset(voice).name
        if normalized_mode == "voice_clone":
            effective_prompt_audio_path = self.resolve_prompt_audio_path(
                voice=resolved_voice,
                prompt_audio_path=prompt_audio_path,
            )
        elif prompt_audio_path is not None:
            effective_prompt_audio_path = self.resolve_prompt_audio_path(prompt_audio_path=prompt_audio_path)
            if not prompt_text:
                raise ValueError("continuation mode with prompt_audio_path also requires prompt_text")

        output_path = (
            Path(output_audio_path).expanduser().resolve()
            if output_audio_path is not None
            else self._build_output_path(prefix=f"{resolved_voice}_{normalized_mode}")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        started_at = time.monotonic()
        with self._lock:
            model = self._load_model_locked()
            model = self._restore_model_execution_state(model)
            requested_attn_implementation, effective_global_attn_implementation, effective_local_attn_implementation = (
                self._resolve_request_attention_implementation(attn_implementation)
            )
            audio_tokenizer = self._load_audio_tokenizer_locked(
                tts_attn_implementation=effective_global_attn_implementation
            )
            self._apply_model_attention_implementation(
                model,
                global_attn=effective_global_attn_implementation,
                local_attn=effective_local_attn_implementation,
            )
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            try:
                result = model.inference(
                    text=normalized_text,
                    output_audio_path=str(output_path),
                    mode=normalized_mode,
                    prompt_text=prompt_text,
                    prompt_audio_path=None if effective_prompt_audio_path is None else str(effective_prompt_audio_path),
                    text_tokenizer_path=self.checkpoint_path,
                    audio_tokenizer=audio_tokenizer,
                    device=self.device,
                    nq=nq,
                    max_new_frames=int(max_new_frames),
                    voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
                    voice_clone_max_memory_per_sample_gb=float(voice_clone_max_memory_per_sample_gb),
                    tts_max_batch_size=int(tts_max_batch_size),
                    codec_max_batch_size=int(codec_max_batch_size),
                    do_sample=bool(do_sample),
                    use_kv_cache=True,
                    text_temperature=float(text_temperature),
                    text_top_p=float(text_top_p),
                    text_top_k=int(text_top_k),
                    audio_temperature=float(audio_temperature),
                    audio_top_p=float(audio_top_p),
                    audio_top_k=int(audio_top_k),
                    audio_repetition_penalty=float(audio_repetition_penalty),
                )
            except Exception:
                self._discard_loaded_audio_tokenizer_locked(
                    "inference failed; reloading audio tokenizer on next request"
                )
                self._discard_loaded_model_locked("inference failed; reloading checkpoint on next request")
                raise
            effective_global_attn_implementation, effective_local_attn_implementation = (
                self._read_model_attention_implementation(model)
            )
            current_parameter = next(model.parameters(), None)
            if current_parameter is not None and current_parameter.dtype != self.dtype:
                self._discard_loaded_model_locked(
                    f"inference left model in dtype={current_parameter.dtype}; reloading checkpoint on next request"
                )

        waveform = result["waveform"].detach().cpu()
        waveform_numpy = waveform_to_numpy(waveform)
        return {
            "audio_path": str(output_path),
            "sample_rate": int(result["sample_rate"]),
            "waveform": waveform,
            "waveform_numpy": waveform_numpy,
            "audio_token_ids": result["audio_token_ids"],
            "reference_audio_token_ids": result["reference_audio_token_ids"],
            "elapsed_seconds": time.monotonic() - started_at,
            "voice": resolved_voice,
            "mode": normalized_mode,
            "prompt_audio_path": None if effective_prompt_audio_path is None else str(effective_prompt_audio_path),
            "requested_attn_implementation": requested_attn_implementation,
            "effective_global_attn_implementation": effective_global_attn_implementation,
            "effective_local_attn_implementation": effective_local_attn_implementation,
            "voice_clone_text_chunks": result.get("voice_clone_text_chunks"),
            "voice_clone_chunk_batch_size": result.get("voice_clone_chunk_batch_size"),
            "voice_clone_codec_batch_size": result.get("voice_clone_codec_batch_size"),
        }

    def synthesize_stream(
        self,
        *,
        text: str,
        voice: Optional[str] = None,
        mode: str = "voice_clone",
        output_audio_path: Optional[str | Path] = None,
        prompt_audio_path: Optional[str | Path] = None,
        prompt_text: Optional[str] = None,
        max_new_frames: int = 375,
        voice_clone_max_text_tokens: int = 75,
        voice_clone_max_memory_per_sample_gb: float = 1.0,
        tts_max_batch_size: int = 0,
        codec_max_batch_size: int = 0,
        do_sample: bool = True,
        text_temperature: float = 1.0,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 0.8,
        audio_top_p: float = 0.95,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.2,
        nq: Optional[int] = None,
        seed: Optional[int] = None,
        attn_implementation: Optional[str] = None,
    ) -> Iterator[dict[str, object]]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise ValueError("text is required")

        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"continuation", "voice_clone"}:
            raise ValueError("mode must be either 'continuation' or 'voice_clone'")

        effective_prompt_audio_path: Optional[Path] = None
        resolved_voice = self.get_voice_preset(voice).name
        if normalized_mode == "voice_clone":
            effective_prompt_audio_path = self.resolve_prompt_audio_path(
                voice=resolved_voice,
                prompt_audio_path=prompt_audio_path,
            )
        elif prompt_audio_path is not None:
            effective_prompt_audio_path = self.resolve_prompt_audio_path(prompt_audio_path=prompt_audio_path)
            if not prompt_text:
                raise ValueError("continuation mode with prompt_audio_path also requires prompt_text")

        output_path = (
            Path(output_audio_path).expanduser().resolve()
            if output_audio_path is not None
            else self._build_output_path(prefix=f"{resolved_voice}_{normalized_mode}_stream")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        started_at = time.monotonic()
        final_result: dict[str, object] | None = None
        with self._lock:
            model = self._load_model_locked()
            model = self._restore_model_execution_state(model)
            requested_attn_implementation, effective_global_attn_implementation, effective_local_attn_implementation = (
                self._resolve_request_attention_implementation(attn_implementation)
            )
            audio_tokenizer = self._load_audio_tokenizer_locked(
                tts_attn_implementation=effective_global_attn_implementation
            )
            self._apply_model_attention_implementation(
                model,
                global_attn=effective_global_attn_implementation,
                local_attn=effective_local_attn_implementation,
            )
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            try:
                for event in model.inference_stream(
                    text=normalized_text,
                    output_audio_path=str(output_path),
                    mode=normalized_mode,
                    prompt_text=prompt_text,
                    prompt_audio_path=None if effective_prompt_audio_path is None else str(effective_prompt_audio_path),
                    text_tokenizer_path=self.checkpoint_path,
                    audio_tokenizer=audio_tokenizer,
                    device=self.device,
                    nq=nq,
                    max_new_frames=int(max_new_frames),
                    voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
                    voice_clone_max_memory_per_sample_gb=float(voice_clone_max_memory_per_sample_gb),
                    tts_max_batch_size=int(tts_max_batch_size),
                    codec_max_batch_size=int(codec_max_batch_size),
                    do_sample=bool(do_sample),
                    use_kv_cache=True,
                    text_temperature=float(text_temperature),
                    text_top_p=float(text_top_p),
                    text_top_k=int(text_top_k),
                    audio_temperature=float(audio_temperature),
                    audio_top_p=float(audio_top_p),
                    audio_top_k=int(audio_top_k),
                    audio_repetition_penalty=float(audio_repetition_penalty),
                ):
                    event_type = str(event.get("type", ""))
                    if event_type == "audio":
                        waveform = torch.as_tensor(event["waveform"], dtype=torch.float32).cpu()
                        yield {
                            "type": "audio",
                            "waveform": waveform,
                            "waveform_numpy": waveform_to_numpy(waveform),
                            "sample_rate": int(event["sample_rate"]),
                            "chunk_index": int(event.get("chunk_index", 0)),
                            "is_pause": bool(event.get("is_pause", False)),
                            "emitted_audio_seconds": float(event.get("emitted_audio_seconds", 0.0)),
                            "lead_seconds": float(event.get("lead_seconds", 0.0)),
                        }
                        continue
                    if event_type == "result":
                        final_result = dict(event)
            except Exception:
                self._discard_loaded_audio_tokenizer_locked(
                    "streaming inference failed; reloading audio tokenizer on next request"
                )
                self._discard_loaded_model_locked("streaming inference failed; reloading checkpoint on next request")
                raise

            effective_global_attn_implementation, effective_local_attn_implementation = (
                self._read_model_attention_implementation(model)
            )
            current_parameter = next(model.parameters(), None)
            if current_parameter is not None and current_parameter.dtype != self.dtype:
                self._discard_loaded_model_locked(
                    f"streaming inference left model in dtype={current_parameter.dtype}; reloading checkpoint on next request"
                )

        if final_result is None:
            raise RuntimeError("Streaming synthesis finished without a final result.")

        waveform = torch.as_tensor(final_result["waveform"], dtype=torch.float32).cpu()
        yield {
            "type": "result",
            "audio_path": str(final_result["audio_path"]),
            "sample_rate": int(final_result["sample_rate"]),
            "waveform": waveform,
            "waveform_numpy": waveform_to_numpy(waveform),
            "audio_token_ids": final_result["audio_token_ids"],
            "reference_audio_token_ids": final_result["reference_audio_token_ids"],
            "elapsed_seconds": time.monotonic() - started_at,
            "voice": resolved_voice,
            "mode": normalized_mode,
            "prompt_audio_path": None if effective_prompt_audio_path is None else str(effective_prompt_audio_path),
            "requested_attn_implementation": requested_attn_implementation,
            "effective_global_attn_implementation": effective_global_attn_implementation,
            "effective_local_attn_implementation": effective_local_attn_implementation,
            "voice_clone_text_chunks": final_result.get("voice_clone_text_chunks"),
            "voice_clone_chunk_batch_size": final_result.get("voice_clone_chunk_batch_size"),
            "voice_clone_codec_batch_size": final_result.get("voice_clone_codec_batch_size"),
        }

    def warmup(
        self,
        *,
        text: str = "你好，欢迎使用 Nano-TTS。",
        voice: Optional[str] = None,
    ) -> dict[str, object]:
        return self.synthesize(
            text=text,
            voice=voice or self.default_voice,
            mode="voice_clone",
            output_audio_path=self.output_dir / "_warmup" / "warmup.wav",
            max_new_frames=96,
            voice_clone_max_text_tokens=75,
            do_sample=False,
            text_temperature=1.0,
            text_top_p=1.0,
            text_top_k=50,
            audio_temperature=0.8,
            audio_top_p=0.95,
            audio_top_k=25,
            audio_repetition_penalty=1.0,
        )
