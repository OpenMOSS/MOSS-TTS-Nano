from __future__ import annotations

import logging
import re
import shutil
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sentencepiece as spm

from moss_tts_nano.defaults import DEFAULT_OUTPUT_DIR
from text_normalization_pipeline import WeTextProcessingManager, prepare_tts_request_texts

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR
from ort_cpu_runtime import (
    OrtCpuRuntime,
    _normalize_sample_mode,
    _resolve_stream_decode_frame_budget,
    EXECUTION_PROVIDER_CPU,
    SAMPLE_MODE_FIXED,
    SAMPLE_MODE_FULL,
    SAMPLE_MODE_GREEDY,
)

DEFAULT_BROWSER_ONNX_MODEL_DIR = REPO_ROOT / "models"
DEFAULT_BROWSER_ONNX_TTS_DIR = DEFAULT_BROWSER_ONNX_MODEL_DIR / "MOSS-TTS-Nano-100M-ONNX"
DEFAULT_BROWSER_ONNX_CODEC_DIR = DEFAULT_BROWSER_ONNX_MODEL_DIR / "MOSS-Audio-Tokenizer-Nano-ONNX"
DEFAULT_BROWSER_ONNX_TTS_REPO_ID = "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX"
DEFAULT_BROWSER_ONNX_CODEC_REPO_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
DEFAULT_BROWSER_ONNX_TTS_REPO_URL = f"https://huggingface.co/{DEFAULT_BROWSER_ONNX_TTS_REPO_ID}"
DEFAULT_BROWSER_ONNX_CODEC_REPO_URL = f"https://huggingface.co/{DEFAULT_BROWSER_ONNX_CODEC_REPO_ID}"
DEFAULT_BROWSER_ONNX_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "infer_onnx_output.wav"
DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS = 0.40
DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS = 0.24
SENTENCE_END_PUNCTUATION = set(".!?。！？；;")
CLAUSE_SPLIT_PUNCTUATION = set(",，、；;：:")
CLOSING_PUNCTUATION = set("\"'”’)]}）】》」』")
OPENING_PUNCTUATION = set("\"'“‘([{（【《「『")

_HAN_OR_ALNUM = r"[\u3400-\u9fff\uff10-\uff19\uff21-\uff3a\uff41-\uff5aA-Za-z0-9]"
_ESCAPED_CLOSING = "".join(re.escape(c) for c in CLOSING_PUNCTUATION)
_ESCAPED_OPENING = "".join(re.escape(c) for c in OPENING_PUNCTUATION)
_PARAGRAPH_BREAK = re.compile(r"(?:\r?\n[^\S\r\n]*){2,}|\r[^\S\r\n]*\r")
_SPOKEN_CONTRAST = re.compile(
    rf"(?:^|(?<=[。！？!?；;]))\s*"
    rf"(?P<sentence>"
    rf"(?P<prefix>[{_ESCAPED_OPENING}]*{_HAN_OR_ALNUM}{{0,28}}?(?:(?P<correction>不是|并非|並非)|虽然|雖然){_HAN_OR_ALNUM}{{1,24}})[，,]\s*"
    rf"(?P<focus>(?(correction)(?:而是|是)|(?:但是|但|不过|不過|可是|却|卻)){_HAN_OR_ALNUM}{{2,32}})"
    rf"(?:[。！？!?；;{_ESCAPED_CLOSING}]+|$))\s*"
)


@dataclass(frozen=True)
class ReadingChunk:
    text: str
    pause_ms: int
    focus_fraction: float | None = None


MODEL_MANIFEST_CANDIDATE_RELATIVE_PATHS = (
    "browser_poc_manifest.json",
    "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
    "MOSS-TTS-Nano-ONNX-CPU/browser_poc_manifest.json",
)


def _resolve_model_dir_path(model_dir: str | Path | None) -> Path:
    if model_dir is None:
        return DEFAULT_BROWSER_ONNX_MODEL_DIR.expanduser().resolve()
    return Path(model_dir).expanduser().resolve()


def _default_model_dir_requested(model_dir: str | Path | None) -> bool:
    if model_dir is None:
        return True
    return _resolve_model_dir_path(model_dir) == DEFAULT_BROWSER_ONNX_MODEL_DIR.expanduser().resolve()


def _find_manifest_path(model_dir: Path) -> Path | None:
    for relative_path in MODEL_MANIFEST_CANDIDATE_RELATIVE_PATHS:
        candidate = (model_dir / relative_path).resolve()
        if candidate.is_file():
            return candidate
    return None


def _directory_contains_all(parent: Path, required_names: Sequence[str]) -> bool:
    return all((parent / name).exists() for name in required_names)


def _find_directory_with_required_names(root_dir: Path, required_names: Sequence[str]) -> Path | None:
    if not root_dir.exists():
        return None
    if _directory_contains_all(root_dir, required_names):
        return root_dir
    sentinel_name = str(required_names[0])
    for candidate in root_dir.rglob(sentinel_name):
        parent = candidate.parent
        if _directory_contains_all(parent, required_names):
            return parent
    return None


def _promote_directory_contents(source_dir: Path, target_dir: Path) -> None:
    if source_dir.resolve() == target_dir.resolve():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for child in source_dir.iterdir():
        destination = target_dir / child.name
        if destination.exists():
            continue
        shutil.move(str(child), str(destination))


def _normalize_download_layout(target_dir: Path, required_names: Sequence[str]) -> None:
    candidate_dir = _find_directory_with_required_names(target_dir, required_names)
    if candidate_dir is None:
        return
    _promote_directory_contents(candidate_dir, target_dir)


def _snapshot_download_repo(
    *,
    repo_id: str,
    local_dir: Path,
    allow_patterns: Sequence[str],
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "huggingface_hub is required to auto-download ONNX assets. Install it with `pip install huggingface_hub`."
        ) from exc
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=list(allow_patterns),
    )


def _download_default_browser_onnx_assets(model_dir: Path) -> None:
    logging.info("browser_onnx assets missing under %s; downloading from Hugging Face.", model_dir)
    logging.info("browser_onnx TTS repo: %s", DEFAULT_BROWSER_ONNX_TTS_REPO_URL)
    logging.info("browser_onnx codec repo: %s", DEFAULT_BROWSER_ONNX_CODEC_REPO_URL)
    tts_dir = model_dir / DEFAULT_BROWSER_ONNX_TTS_DIR.name
    codec_dir = model_dir / DEFAULT_BROWSER_ONNX_CODEC_DIR.name
    _snapshot_download_repo(
        repo_id=DEFAULT_BROWSER_ONNX_TTS_REPO_ID,
        local_dir=tts_dir,
        allow_patterns=("*.onnx", "*.data", "*.json", "tokenizer.model"),
    )
    _snapshot_download_repo(
        repo_id=DEFAULT_BROWSER_ONNX_CODEC_REPO_ID,
        local_dir=codec_dir,
        allow_patterns=("*.onnx", "*.data", "*.json"),
    )
    _normalize_download_layout(
        tts_dir,
        required_names=("browser_poc_manifest.json", "tts_browser_onnx_meta.json", "tokenizer.model"),
    )
    _normalize_download_layout(
        codec_dir,
        required_names=("codec_browser_onnx_meta.json",),
    )


def ensure_browser_onnx_model_dir(model_dir: str | Path | None = None) -> Path:
    resolved_model_dir = _resolve_model_dir_path(model_dir)
    manifest_path = _find_manifest_path(resolved_model_dir)
    if manifest_path is not None:
        return resolved_model_dir
    if not _default_model_dir_requested(model_dir):
        tried_paths = [str((resolved_model_dir / item).resolve()) for item in MODEL_MANIFEST_CANDIDATE_RELATIVE_PATHS]
        raise FileNotFoundError(
            "browser_onnx model assets not found under the provided --model-dir. tried: " + ", ".join(tried_paths)
        )
    _download_default_browser_onnx_assets(resolved_model_dir)
    manifest_path = _find_manifest_path(resolved_model_dir)
    if manifest_path is None:
        tried_paths = [str((resolved_model_dir / item).resolve()) for item in MODEL_MANIFEST_CANDIDATE_RELATIVE_PATHS]
        raise FileNotFoundError(
            "browser_onnx assets were downloaded but browser_poc_manifest.json is still missing. "
            + "tried: "
            + ", ".join(tried_paths)
        )
    return resolved_model_dir


def _contains_cjk(text: str) -> bool:
    for character in str(text or ""):
        if (
            "\u4e00" <= character <= "\u9fff"
            or "\u3400" <= character <= "\u4dbf"
            or "\u3040" <= character <= "\u30ff"
            or "\uac00" <= character <= "\ud7af"
        ):
            return True
    return False


def _prepare_text_for_sentence_chunking(text: str) -> str:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        raise ValueError("Text prompt cannot be empty.")
    normalized_text = normalized_text.replace("\r", " ").replace("\n", " ")
    while "  " in normalized_text:
        normalized_text = normalized_text.replace("  ", " ")
    if _contains_cjk(normalized_text):
        if normalized_text[-1] not in SENTENCE_END_PUNCTUATION:
            normalized_text += "。"
        return normalized_text
    if normalized_text[:1].islower():
        normalized_text = normalized_text[:1].upper() + normalized_text[1:]
    if normalized_text[-1].isalnum():
        normalized_text += "."
    if len([item for item in normalized_text.split() if item]) < 5:
        normalized_text = f"        {normalized_text}"
    return normalized_text


def _split_text_by_punctuation(text: str, punctuation: set[str]) -> list[str]:
    sentences: list[str] = []
    current_chars: list[str] = []
    index = 0
    normalized_text = str(text or "")
    while index < len(normalized_text):
        character = normalized_text[index]
        current_chars.append(character)
        if character in punctuation:
            lookahead = index + 1
            while lookahead < len(normalized_text) and normalized_text[lookahead] in CLOSING_PUNCTUATION:
                current_chars.append(normalized_text[lookahead])
                lookahead += 1
            sentence = "".join(current_chars).strip()
            if sentence:
                sentences.append(sentence)
            current_chars.clear()
            while lookahead < len(normalized_text) and normalized_text[lookahead].isspace():
                lookahead += 1
            index = lookahead
            continue
        index += 1
    tail = "".join(current_chars).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _join_sentence_parts(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if _contains_cjk(left) or _contains_cjk(right):
        return left + right
    return f"{left} {right}"


def _merge_audio_channels(channel_arrays: list[np.ndarray]) -> np.ndarray:
    if not channel_arrays:
        return np.zeros((0, 1), dtype=np.float32)
    if len(channel_arrays) == 1:
        return np.asarray(channel_arrays[0], dtype=np.float32).reshape(-1, 1)
    min_length = min(int(channel.shape[0]) for channel in channel_arrays)
    trimmed = [np.asarray(channel[:min_length], dtype=np.float32) for channel in channel_arrays]
    return np.stack(trimmed, axis=1)


def _concat_waveforms(waveforms: list[np.ndarray]) -> np.ndarray:
    if not waveforms:
        return np.zeros((0, 1), dtype=np.float32)
    non_empty = [waveform for waveform in waveforms if waveform.size > 0]
    if not non_empty:
        channel_count = int(waveforms[0].shape[1]) if waveforms[0].ndim == 2 and waveforms[0].shape[1] > 0 else 1
        return np.zeros((0, channel_count), dtype=np.float32)
    return np.concatenate(non_empty, axis=0)


def _reading_pause_ms(text: str) -> int:
    ending = text.rstrip("".join(CLOSING_PUNCTUATION))[-1:]
    if ending in CLAUSE_SPLIT_PUNCTUATION:
        return 150
    if ending in SENTENCE_END_PUNCTUATION:
        return 280
    return 80


def _emphasize_contrast(audio: np.ndarray, active: np.ndarray, sample_rate: int, fraction: float) -> np.ndarray:
    # shortcut: infer the clause boundary only from a nearby audible pause until word timing is available.
    if not np.any(active):
        return audio
    speech_start = int(active.argmax())
    speech_end = len(active) - int(active[::-1].argmax())
    speech_length = speech_end - speech_start
    if speech_length <= 0:
        return audio
    expected = speech_start + round(speech_length * fraction)
    edges = np.flatnonzero(np.diff(np.r_[True, active[speech_start:speech_end], True])) + speech_start
    candidates = [
        (int(start), int(end))
        for start, end in edges.reshape(-1, 2)
        if end - start >= sample_rate * 0.08
        and start >= speech_start + sample_rate * 0.18
        and end <= speech_end - sample_rate * 0.18
        and abs((start + end) / 2 - expected) <= speech_length * 0.18
    ]
    candidates.sort(key=lambda pause: abs(sum(pause) / 2 - expected))
    if not candidates:
        return audio
    start, end = candidates[0]
    if len(candidates) > 1:
        distance = abs((start + end) / 2 - expected)
        next_distance = abs(sum(candidates[1]) / 2 - expected)
        if next_distance - distance < sample_rate * 0.12:
            return audio
    peak = float(np.max(np.abs(audio[end:])))
    focus_gain = min(1.20, max(1.0, 0.98 / peak)) if peak else 1.0
    adjusted = audio.copy()
    adjusted[:start] *= 0.92
    fade = np.linspace(0.92, focus_gain, end - start, dtype=np.float32)
    adjusted[start:end] *= fade[:, None] if audio.ndim > 1 else fade
    adjusted[end:] *= focus_gain
    return adjusted


def _join_reading_audio(
    chunks: Sequence[ReadingChunk], waveforms: list[np.ndarray], sample_rate: int
) -> np.ndarray:
    if not waveforms:
        return np.zeros((0, 1), dtype=np.float32)
    if any(not audio.size for audio in waveforms):
        raise ValueError("No audio was generated for part of the text.")
    if len(chunks) != len(waveforms):
        raise ValueError("Mismatch between number of chunks and waveforms.")
    is_1d = all(audio.ndim == 1 for audio in waveforms)
    normalized_waveforms = [
        audio.reshape(-1, 1) if audio.ndim == 1 else audio
        for audio in waveforms
    ]
    edges: list[tuple[int, int]] = []
    adjusted: list[np.ndarray] = []
    for chunk, audio in zip(chunks, normalized_waveforms):
        if len(normalized_waveforms) == 1 and chunk.focus_fraction is None:
            return waveforms[0]
        levels = np.max(np.abs(audio), axis=1)
        peak = float(np.max(levels))
        active = levels > min(0.006, peak * 0.04)
        # Keep a short margin around quiet consonants and breaths when shortening a join.
        margin = sample_rate // 50
        if np.any(active):
            head_silence = max(0, int(active.argmax()) - margin)
            tail_silence = max(0, int(active[::-1].argmax()) - margin)
        else:
            head_silence = len(audio) // 2
            tail_silence = len(audio) - head_silence
        edges.append((head_silence, tail_silence))
        if chunk.focus_fraction is not None and peak > 0:
            audio = _emphasize_contrast(audio, active, sample_rate, chunk.focus_fraction)
        adjusted.append(audio)

    parts: list[np.ndarray] = []
    start = 0
    for index, audio in enumerate(adjusted):
        if index == len(adjusted) - 1:
            parts.append(audio[start:])
            break
        target = round(chunks[index].pause_ms * sample_rate / 1000)
        tail = edges[index][1]
        head = edges[index + 1][0]
        excess = max(0, head + tail - target)
        drop_tail = round(excess * tail / (head + tail)) if excess else 0
        parts.append(audio[start : len(audio) - drop_tail])
        missing = target - head - tail
        if missing > 0:
            parts.append(np.zeros((missing, audio.shape[1]), dtype=np.float32))
        start = excess - drop_tail
    result = _concat_waveforms(parts)
    return result.reshape(-1) if is_1d else result


def _write_waveform_to_wav(path: str | Path, waveform: np.ndarray, sample_rate: int) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = np.round(clipped * 32767.0).astype(np.int16)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(int(pcm16.shape[1]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())
    return output_path


class ReadingTextChunker:
    def count_text_tokens(self, text: str) -> int:
        raise NotImplementedError

    def split_text_by_token_budget(self, text: str, max_tokens: int) -> list[str]:
        remaining_text = str(text or "").strip()
        if not remaining_text:
            return []
        pieces: list[str] = []
        preferred_boundary_chars = set(CLAUSE_SPLIT_PUNCTUATION) | set(SENTENCE_END_PUNCTUATION) | {" "}
        while remaining_text:
            if self.count_text_tokens(remaining_text) <= max_tokens:
                pieces.append(remaining_text)
                break
            low = 1
            high = len(remaining_text)
            best_prefix_length = 1
            while low <= high:
                middle = (low + high) // 2
                candidate = remaining_text[:middle].strip()
                if not candidate:
                    low = middle + 1
                    continue
                if self.count_text_tokens(candidate) <= max_tokens:
                    best_prefix_length = middle
                    low = middle + 1
                else:
                    high = middle - 1
            cut_index = best_prefix_length
            prefix = remaining_text[:best_prefix_length]
            preferred_index = -1
            scan_min = max(-1, len(prefix) - 25)
            for scan_index in range(len(prefix) - 1, scan_min, -1):
                if prefix[scan_index] in preferred_boundary_chars:
                    preferred_index = scan_index + 1
                    break
            if preferred_index > 0:
                cut_index = preferred_index
            piece = remaining_text[:cut_index].strip()
            if not piece:
                piece = remaining_text[:best_prefix_length].strip()
                cut_index = best_prefix_length
            pieces.append(piece)
            remaining_text = remaining_text[cut_index:].strip()
        return pieces

    def split_voice_clone_text(self, text: str, max_tokens: int = 75) -> list[str]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return []
        safe_max_tokens = max(1, int(max_tokens))
        prepared_text = _prepare_text_for_sentence_chunking(normalized_text)
        sentence_candidates = _split_text_by_punctuation(prepared_text, SENTENCE_END_PUNCTUATION) or [prepared_text.strip()]
        sentence_slices: list[tuple[int, str]] = []
        for sentence_text in sentence_candidates:
            normalized_sentence = sentence_text.strip()
            if not normalized_sentence:
                continue
            sentence_token_count = self.count_text_tokens(normalized_sentence)
            if sentence_token_count <= safe_max_tokens:
                sentence_slices.append((sentence_token_count, normalized_sentence))
                continue
            clause_candidates = _split_text_by_punctuation(normalized_sentence, CLAUSE_SPLIT_PUNCTUATION)
            if len(clause_candidates) <= 1:
                clause_candidates = [normalized_sentence]
            for clause_text in clause_candidates:
                normalized_clause = clause_text.strip()
                if not normalized_clause:
                    continue
                clause_token_count = self.count_text_tokens(normalized_clause)
                if clause_token_count <= safe_max_tokens:
                    sentence_slices.append((clause_token_count, normalized_clause))
                    continue
                for piece in self.split_text_by_token_budget(normalized_clause, safe_max_tokens):
                    normalized_piece = piece.strip()
                    if normalized_piece:
                        sentence_slices.append((self.count_text_tokens(normalized_piece), normalized_piece))
        chunks: list[str] = []
        current_chunk = ""
        current_chunk_token_count = 0
        for sentence_token_count, sentence_text in sentence_slices:
            if not current_chunk:
                current_chunk = sentence_text
                current_chunk_token_count = sentence_token_count
                continue
            if current_chunk_token_count + sentence_token_count > safe_max_tokens:
                chunks.append(current_chunk.strip())
                current_chunk = sentence_text
                current_chunk_token_count = sentence_token_count
            else:
                current_chunk = _join_sentence_parts(current_chunk, sentence_text)
                current_chunk_token_count = self.count_text_tokens(current_chunk)
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks if len(chunks) > 1 else [normalized_text]

    def plan_reading_chunks(self, paragraphs: Sequence[str], max_tokens: int) -> list[ReadingChunk]:
        chunks: list[ReadingChunk] = []
        for paragraph in paragraphs:
            paragraph_chunks: list[ReadingChunk] = []
            start = 0
            for match in _SPOKEN_CONTRAST.finditer(paragraph):
                sentence = match.group("sentence")
                if self.count_text_tokens(sentence) > max_tokens:
                    continue
                paragraph_chunks.extend(
                    ReadingChunk(piece, _reading_pause_ms(piece))
                    for piece in self.split_voice_clone_text(paragraph[start : match.start()], max_tokens=max_tokens)
                )
                prefix = match.group("prefix")
                focus = match.group("focus")
                fraction = len(prefix) / (len(prefix) + len(focus))
                paragraph_chunks.append(ReadingChunk(sentence, _reading_pause_ms(sentence), fraction))
                start = match.end()
            paragraph_chunks.extend(
                ReadingChunk(piece, _reading_pause_ms(piece))
                for piece in self.split_voice_clone_text(paragraph[start:], max_tokens=max_tokens)
            )
            if paragraph_chunks:
                last = paragraph_chunks[-1]
                paragraph_chunks[-1] = ReadingChunk(last.text, 460, last.focus_fraction)
                chunks.extend(paragraph_chunks)
        return chunks


class OnnxTtsRuntime(ReadingTextChunker, OrtCpuRuntime):
    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        thread_count: int = 4,
        max_new_frames: int | None = None,
        do_sample: bool | None = None,
        sample_mode: str | None = None,
        execution_provider: str = EXECUTION_PROVIDER_CPU,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ) -> None:
        resolved_model_dir = ensure_browser_onnx_model_dir(model_dir)
        super().__init__(
            model_dir=resolved_model_dir,
            thread_count=thread_count,
            max_new_frames=max_new_frames,
            do_sample=do_sample,
            sample_mode=sample_mode,
            execution_provider=execution_provider,
        )
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_relative_path = str(self.manifest["model_files"].get("tokenizer_model", "tokenizer.model"))
        tokenizer_path = self.resolve_manifest_relative_path(tokenizer_relative_path)
        self.sp_model = spm.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())
        self._text_normalizer_manager: WeTextProcessingManager | None = None

    def _ensure_text_normalizer(self, enable_wetext: bool) -> WeTextProcessingManager | None:
        if not enable_wetext:
            return None
        if self._text_normalizer_manager is None:
            self._text_normalizer_manager = WeTextProcessingManager()
        snapshot = self._text_normalizer_manager.ensure_ready()
        if not snapshot.ready:
            raise RuntimeError(snapshot.error or snapshot.message)
        return self._text_normalizer_manager

    def encode_text(self, text: str) -> list[int]:
        return [int(token_id) for token_id in self.sp_model.encode(str(text or ""), out_type=int)]

    def count_text_tokens(self, text: str) -> int:
        return len(self.encode_text(text))

    def prepare_synthesis_text(
        self,
        *,
        text: str,
        voice: str = "",
        prompt_text: str = "",
        enable_wetext: bool = True,
        enable_normalize_tts_text: bool = True,
    ) -> dict[str, object]:
        text_normalizer_manager = self._ensure_text_normalizer(enable_wetext)
        return prepare_tts_request_texts(
            text=text,
            prompt_text=prompt_text,
            voice=voice,
            enable_wetext=bool(enable_wetext),
            enable_normalize_tts_text=bool(enable_normalize_tts_text),
            text_normalizer_manager=text_normalizer_manager,
        )

    def estimate_voice_clone_inter_chunk_pause_seconds(self, text_chunk: str) -> float:
        word_count = len([item for item in str(text_chunk or "").strip().split() if item])
        return (
            DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS
            if word_count <= 4
            else DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS
        )

    def _load_reference_audio(self, reference_audio_path: str | Path) -> np.ndarray:
        import torch
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(Path(reference_audio_path).expanduser().resolve()))
        waveform = waveform.to(torch.float32)
        target_sample_rate = int(self.codec_meta["codec_config"]["sample_rate"])
        target_channels = int(self.codec_meta["codec_config"]["channels"])
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        current_channels = int(waveform.shape[0])
        if current_channels == target_channels:
            pass
        elif current_channels == 1 and target_channels > 1:
            waveform = waveform.repeat(target_channels, 1)
        elif current_channels > 1 and target_channels == 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unsupported reference audio channel conversion: {current_channels} -> {target_channels}")
        return waveform.unsqueeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    def encode_reference_audio(self, reference_audio_path: str | Path) -> list[list[int]]:
        waveform = self._load_reference_audio(reference_audio_path)
        waveform_length = int(waveform.shape[-1])
        outputs = self.sessions["codec_encode"].run(
            None,
            {
                "waveform": waveform,
                "input_lengths": np.asarray([waveform_length], dtype=np.int32),
            },
        )
        output_names = [output.name for output in self.sessions["codec_encode"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        audio_codes = np.asarray(named_outputs["audio_codes"], dtype=np.int32)
        audio_code_lengths = np.asarray(named_outputs["audio_code_lengths"], dtype=np.int32)
        code_length = int(audio_code_lengths.reshape(-1)[0])
        num_quantizers = int(self.codec_meta["codec_config"]["num_quantizers"])
        prompt_audio_codes: list[list[int]] = []
        for frame_index in range(code_length):
            prompt_audio_codes.append(
                [int(audio_codes[0, frame_index, quantizer_index]) for quantizer_index in range(num_quantizers)]
            )
        return prompt_audio_codes

    def resolve_prompt_audio_codes(
        self,
        *,
        voice: str | None,
        prompt_audio_path: str | Path | None,
    ) -> list[list[int]]:
        if prompt_audio_path:
            return self.encode_reference_audio(prompt_audio_path)
        resolved_voice = str(voice or self.list_builtin_voices()[0]["voice"])
        voice_row = next((item for item in self.list_builtin_voices() if item["voice"] == resolved_voice), None)
        if voice_row is None:
            raise ValueError(f"Built-in voice not found: {resolved_voice}")
        return list(voice_row["prompt_audio_codes"])

    def decode_full_audio_safe(self, generated_frames: list[list[int]]) -> np.ndarray:
        try:
            channel_arrays, _audio_length = self.decode_full_audio(generated_frames)
            return _merge_audio_channels(channel_arrays)
        except Exception as exc:
            logging.warning("full codec decode failed, falling back to incremental decode: %s", exc)
            self.codec_streaming_session.reset()
            merged_by_channel: list[list[np.ndarray]] = [
                [] for _ in range(int(self.codec_meta["codec_config"]["channels"]))
            ]
            try:
                for start_index in range(0, len(generated_frames), 8):
                    frame_chunk = generated_frames[start_index : start_index + 8]
                    decoded = self.codec_streaming_session.run_frames(frame_chunk)
                    if decoded is None:
                        continue
                    audio, audio_length = decoded
                    if audio_length <= 0:
                        continue
                    for channel_index, channel in enumerate(audio[0, :, :audio_length]):
                        merged_by_channel[channel_index].append(np.asarray(channel, dtype=np.float32))
            finally:
                self.codec_streaming_session.reset()
            return _merge_audio_channels(
                [np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32) for chunks in merged_by_channel]
            )

    def synthesize_single_chunk(
        self,
        *,
        text: str,
        prompt_audio_codes: list[list[int]],
        streaming: bool,
    ) -> dict[str, Any]:
        text_token_ids = self.encode_text(text)
        request_rows = self.build_voice_clone_request_rows(prompt_audio_codes, text_token_ids)
        if not streaming:
            generated_frames = self.generate_audio_frames(request_rows)
            waveform = self.decode_full_audio_safe(generated_frames)
            return {
                "text": text,
                "text_token_ids": text_token_ids,
                "generated_frames": generated_frames,
                "waveform": waveform,
            }

        pending_decode_frames: list[list[int]] = []
        emitted_chunks: list[np.ndarray] = []
        emitted_samples_total = 0
        first_audio_emitted_at_perf: float | None = None
        self.codec_streaming_session.reset()

        def decode_pending_frames(force: bool) -> None:
            nonlocal emitted_samples_total, first_audio_emitted_at_perf
            pending_count = len(pending_decode_frames)
            if pending_count <= 0:
                return
            sample_rate = int(self.codec_meta["codec_config"]["sample_rate"])
            decode_budget = _resolve_stream_decode_frame_budget(
                emitted_samples_total,
                sample_rate,
                first_audio_emitted_at_perf,
            )
            if not force and pending_count < max(1, decode_budget):
                return
            frame_budget = pending_count if force else min(pending_count, max(1, decode_budget))
            frame_chunk = pending_decode_frames[:frame_budget]
            del pending_decode_frames[:frame_budget]
            decoded = self.codec_streaming_session.run_frames(frame_chunk)
            if decoded is None:
                return
            audio, audio_length = decoded
            if audio_length <= 0:
                return
            if first_audio_emitted_at_perf is None:
                first_audio_emitted_at_perf = time.perf_counter()
            emitted_samples_total += audio_length
            emitted_chunks.append(_merge_audio_channels([audio[0, channel_index, :audio_length] for channel_index in range(audio.shape[1])]))

        def on_frame(_generated_frames: list[list[int]], _step_index: int, frame: list[int]) -> None:
            pending_decode_frames.append(list(frame))
            decode_pending_frames(False)

        try:
            generated_frames = self.generate_audio_frames(request_rows, on_frame=on_frame)
            decode_pending_frames(True)
        finally:
            self.codec_streaming_session.reset()
        waveform = _concat_waveforms(emitted_chunks)
        return {
            "text": text,
            "text_token_ids": text_token_ids,
            "generated_frames": generated_frames,
            "waveform": waveform,
        }

    def synthesize(
        self,
        *,
        text: str,
        voice: str | None = None,
        prompt_audio_path: str | Path | None = None,
        output_audio_path: str | Path | None = None,
        sample_mode: str | None = None,
        do_sample: bool = True,
        streaming: bool = False,
        max_new_frames: int | None = None,
        voice_clone_max_text_tokens: int = 75,
        enable_wetext: bool = True,
        enable_normalize_tts_text: bool = True,
        seed: int | None = None,
        automatic_prosody: bool = False,
    ) -> dict[str, Any]:
        if max_new_frames is not None:
            self.manifest["generation_defaults"]["max_new_frames"] = int(max_new_frames)
        normalized_sample_mode = _normalize_sample_mode(sample_mode, do_sample)
        self.manifest["generation_defaults"]["sample_mode"] = normalized_sample_mode
        self.manifest["generation_defaults"]["do_sample"] = normalized_sample_mode != SAMPLE_MODE_GREEDY
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        # Blank lines must be read before the text normalizer flattens them.
        if automatic_prosody:
            raw_paragraphs = [
                p.strip() for p in _PARAGRAPH_BREAK.split(str(text or "")) if p.strip()
            ]
            if not raw_paragraphs:
                raw_paragraphs = [str(text or "")]
        else:
            raw_paragraphs = [text]
        prepared_paragraphs = [
            self.prepare_synthesis_text(
                text=paragraph,
                voice=str(voice or ""),
                enable_wetext=enable_wetext,
                enable_normalize_tts_text=enable_normalize_tts_text,
            )
            for paragraph in raw_paragraphs
        ]
        prepared_texts = prepared_paragraphs[0]
        paragraphs = [str(item["text"]) for item in prepared_paragraphs if str(item["text"]).strip()]
        max_tokens = int(voice_clone_max_text_tokens)
        if automatic_prosody:
            prepared_text = "\n\n".join(paragraphs)
            prepared_texts = prepared_texts | {"text": prepared_text, "normalized_text": prepared_text}
            reading_chunks = self.plan_reading_chunks(paragraphs, max_tokens)
            text_chunks = [chunk.text for chunk in reading_chunks]
        else:
            text_chunks = self.split_voice_clone_text(str(prepared_texts["text"]), max_tokens=max_tokens)
        prompt_audio_codes = self.resolve_prompt_audio_codes(voice=voice, prompt_audio_path=prompt_audio_path)
        all_waveforms: list[np.ndarray] = []
        all_generated_frames: list[list[int]] = []
        sample_rate = int(self.codec_meta["codec_config"]["sample_rate"])
        channels = int(self.codec_meta["codec_config"]["channels"])
        chunk_results: list[dict[str, Any]] = []
        for chunk_index, chunk_text in enumerate(text_chunks):
            chunk_result = self.synthesize_single_chunk(
                text=chunk_text,
                prompt_audio_codes=prompt_audio_codes,
                streaming=bool(streaming),
            )
            chunk_results.append(chunk_result)
            all_waveforms.append(np.asarray(chunk_result["waveform"], dtype=np.float32))
            all_generated_frames.extend(chunk_result["generated_frames"])
            if not automatic_prosody and chunk_index < len(text_chunks) - 1:
                pause_seconds = self.estimate_voice_clone_inter_chunk_pause_seconds(chunk_text)
                pause_samples = max(0, int(round(sample_rate * pause_seconds)))
                if pause_samples > 0:
                    all_waveforms.append(np.zeros((pause_samples, channels), dtype=np.float32))
        waveform = (
            _join_reading_audio(reading_chunks, all_waveforms, sample_rate)
            if automatic_prosody
            else _concat_waveforms(all_waveforms)
        )
        resolved_output_audio_path = (
            Path(output_audio_path).expanduser().resolve()
            if output_audio_path
            else (self.output_dir / DEFAULT_BROWSER_ONNX_OUTPUT_PATH.name).resolve()
        )
        audio_path = _write_waveform_to_wav(resolved_output_audio_path, waveform, sample_rate)
        return {
            "audio_path": str(audio_path),
            "waveform": waveform,
            "sample_rate": sample_rate,
            "audio_token_ids": np.asarray(all_generated_frames, dtype=np.int32),
            "text_chunks": text_chunks,
            "prepared_texts": prepared_texts,
            "sample_mode": normalized_sample_mode,
            "do_sample": normalized_sample_mode != SAMPLE_MODE_GREEDY,
            "streaming": bool(streaming),
            "chunk_results": chunk_results,
        }
