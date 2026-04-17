from __future__ import annotations

import argparse
import logging
import os
import queue
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import uvicorn

import app as legacy_app
from onnx_tts_runtime import (
    DEFAULT_BROWSER_ONNX_MODEL_DIR,
    OnnxTtsRuntime,
    _concat_waveforms,
    _merge_audio_channels,
    _write_waveform_to_wav,
)
from text_normalization_pipeline import WeTextProcessingManager

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR
from ort_cpu_runtime import _resolve_stream_decode_frame_budget

_LEGACY_RENDER_INDEX_HTML = legacy_app._render_index_html


class _CpuDeviceInfo:
    type = "cpu"

    def __str__(self) -> str:
        return "cpu"


class OnnxNanoTTSServiceAdapter:
    def __init__(
        self,
        *,
        model_dir: str | Path | None,
        output_dir: str | Path | None = None,
        cpu_threads: int = 4,
        max_new_frames: int = 375,
        text_normalizer_manager: WeTextProcessingManager | None = None,
    ) -> None:
        self.output_dir = Path(output_dir or (APP_DIR / "generated_audio")).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runtime = OnnxTtsRuntime(
            model_dir=model_dir,
            thread_count=max(1, int(cpu_threads)),
            max_new_frames=int(max_new_frames),
            output_dir=self.output_dir,
        )
        self.model_dir = self.runtime.model_dir
        self.runtime._text_normalizer_manager = text_normalizer_manager
        self.device = _CpuDeviceInfo()
        self.dtype = "float32"
        self.attn_implementation = "fixed"
        self._checkpoint_global_attn_implementation = "onnxruntime_cpu"
        self._checkpoint_local_attn_implementation = "onnxruntime_cpu"
        self._configured_global_attn_implementation = "onnxruntime_cpu"
        self._configured_local_attn_implementation = "onnxruntime_cpu"
        self.checkpoint_path = self.runtime.tts_meta_path.parent.resolve()
        self.audio_tokenizer_path = self.runtime.codec_meta_path.parent.resolve()
        self.thread_count = max(1, int(cpu_threads))

    def get_model(self) -> "OnnxNanoTTSServiceAdapter":
        return self

    def warmup(self) -> dict[str, object]:
        voice_name = str(self.runtime.list_builtin_voices()[0]["voice"])
        return self.synthesize(
            text="Warmup.",
            mode="voice_clone",
            voice=voice_name,
            prompt_audio_path=None,
            max_new_frames=min(16, int(self.runtime.manifest["generation_defaults"]["max_new_frames"])),
            voice_clone_max_text_tokens=75,
            do_sample=True,
            text_temperature=1.0,
            text_top_p=1.0,
            text_top_k=50,
            audio_temperature=0.8,
            audio_top_p=0.95,
            audio_top_k=25,
            audio_repetition_penalty=1.2,
            seed=1234,
        )

    def split_voice_clone_text(self, *, text: str, voice_clone_max_text_tokens: int) -> list[str]:
        return self.runtime.split_voice_clone_text(str(text or ""), max_tokens=int(voice_clone_max_text_tokens))

    def _apply_generation_options(
        self,
        *,
        sample_mode: str | None,
        max_new_frames: int,
        do_sample: bool,
        text_temperature: float,
        text_top_p: float,
        text_top_k: int,
        audio_temperature: float,
        audio_top_p: float,
        audio_top_k: int,
        audio_repetition_penalty: float,
        seed: int | None,
    ) -> None:
        resolved_sample_mode = self._resolve_sample_mode(sample_mode, do_sample=do_sample)
        generation_defaults = self.runtime.manifest["generation_defaults"]
        generation_defaults["max_new_frames"] = int(max_new_frames)
        generation_defaults["sample_mode"] = resolved_sample_mode
        generation_defaults["do_sample"] = resolved_sample_mode != "greedy"
        generation_defaults["text_temperature"] = float(text_temperature)
        generation_defaults["text_top_p"] = float(text_top_p)
        generation_defaults["text_top_k"] = int(text_top_k)
        generation_defaults["audio_temperature"] = float(audio_temperature)
        generation_defaults["audio_top_p"] = float(audio_top_p)
        generation_defaults["audio_top_k"] = int(audio_top_k)
        generation_defaults["audio_repetition_penalty"] = float(audio_repetition_penalty)
        if seed is not None:
            self.runtime.rng = np.random.default_rng(int(seed))

    @staticmethod
    def _resolve_sample_mode(raw_sample_mode: str | None, *, do_sample: bool) -> str:
        normalized = str(raw_sample_mode or "").strip().lower()
        if normalized in {"fixed", "full", "greedy"}:
            if normalized == "greedy":
                return "greedy"
            return normalized if bool(do_sample) else "greedy"
        return "fixed" if bool(do_sample) else "greedy"

    def _format_result_payload(
        self,
        *,
        waveform: np.ndarray,
        sample_rate: int,
        elapsed_seconds: float,
        audio_path: str,
        voice: str | None,
        prompt_audio_path: str | None,
        text_chunks: list[str],
    ) -> dict[str, object]:
        return {
            "audio_path": audio_path,
            "waveform_numpy": np.asarray(waveform, dtype=np.float32),
            "sample_rate": int(sample_rate),
            "elapsed_seconds": float(elapsed_seconds),
            "mode": "voice_clone",
            "voice": str(voice or ""),
            "prompt_audio_path": str(prompt_audio_path or ""),
            "voice_clone_text_chunks": list(text_chunks),
            "effective_global_attn_implementation": "onnxruntime_cpu",
            "effective_local_attn_implementation": "onnxruntime_cpu",
            "voice_clone_chunk_batch_size": 1,
            "voice_clone_codec_batch_size": 1,
        }

    def synthesize(
        self,
        *,
        text: str,
        mode: str,
        voice: str | None,
        prompt_audio_path: str | None,
        max_new_frames: int,
        voice_clone_max_text_tokens: int,
        tts_max_batch_size: int = 0,
        codec_max_batch_size: int = 0,
        attn_implementation: str = "model_default",
        do_sample: bool = True,
        text_temperature: float = 1.0,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 0.8,
        audio_top_p: float = 0.95,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.2,
        seed: int | None = None,
    ) -> dict[str, object]:
        del mode, tts_max_batch_size, codec_max_batch_size
        resolved_sample_mode = self._resolve_sample_mode(attn_implementation, do_sample=do_sample)
        self._apply_generation_options(
            sample_mode=resolved_sample_mode,
            max_new_frames=max_new_frames,
            do_sample=do_sample,
            text_temperature=text_temperature,
            text_top_p=text_top_p,
            text_top_k=text_top_k,
            audio_temperature=audio_temperature,
            audio_top_p=audio_top_p,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            seed=seed,
        )
        start_time = time.perf_counter()
        result = self.runtime.synthesize(
            text=str(text or ""),
            voice=voice,
            prompt_audio_path=prompt_audio_path,
            sample_mode=resolved_sample_mode,
            do_sample=resolved_sample_mode != "greedy",
            streaming=False,
            max_new_frames=int(max_new_frames),
            voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
            enable_wetext=False,
            enable_normalize_tts_text=False,
            seed=seed,
        )
        elapsed_seconds = time.perf_counter() - start_time
        waveform = np.asarray(result["waveform"], dtype=np.float32)
        return self._format_result_payload(
            waveform=waveform,
            sample_rate=int(result["sample_rate"]),
            elapsed_seconds=elapsed_seconds,
            audio_path=str(result["audio_path"]),
            voice=voice,
            prompt_audio_path=prompt_audio_path,
            text_chunks=[str(chunk).strip() for chunk in result.get("text_chunks", []) if str(chunk).strip()],
        )

    def synthesize_stream(
        self,
        *,
        text: str,
        mode: str,
        voice: str | None,
        prompt_audio_path: str | None,
        max_new_frames: int,
        voice_clone_max_text_tokens: int,
        tts_max_batch_size: int = 0,
        codec_max_batch_size: int = 0,
        attn_implementation: str = "model_default",
        do_sample: bool = True,
        text_temperature: float = 1.0,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 0.8,
        audio_top_p: float = 0.95,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.2,
        seed: int | None = None,
    ) -> Iterator[dict[str, object]]:
        del mode, tts_max_batch_size, codec_max_batch_size
        event_queue: "queue.Queue[dict[str, object] | None]" = queue.Queue(maxsize=128)

        def _worker() -> None:
            try:
                resolved_sample_mode = self._resolve_sample_mode(attn_implementation, do_sample=do_sample)
                self._apply_generation_options(
                    sample_mode=resolved_sample_mode,
                    max_new_frames=max_new_frames,
                    do_sample=do_sample,
                    text_temperature=text_temperature,
                    text_top_p=text_top_p,
                    text_top_k=text_top_k,
                    audio_temperature=audio_temperature,
                    audio_top_p=audio_top_p,
                    audio_top_k=audio_top_k,
                    audio_repetition_penalty=audio_repetition_penalty,
                    seed=seed,
                )
                start_time = time.perf_counter()
                prompt_audio_codes = self.runtime.resolve_prompt_audio_codes(voice=voice, prompt_audio_path=prompt_audio_path)
                text_chunks = self.runtime.split_voice_clone_text(str(text or ""), max_tokens=int(voice_clone_max_text_tokens))
                sample_rate = int(self.runtime.codec_meta["codec_config"]["sample_rate"])
                channels = int(self.runtime.codec_meta["codec_config"]["channels"])
                emitted_samples_total = 0
                first_audio_emitted_at_perf: float | None = None
                all_waveforms: list[np.ndarray] = []
                all_generated_frames: list[list[int]] = []

                for chunk_index, chunk_text in enumerate(text_chunks):
                    text_token_ids = self.runtime.encode_text(chunk_text)
                    request_rows = self.runtime.build_voice_clone_request_rows(prompt_audio_codes, text_token_ids)
                    pending_decode_frames: list[list[int]] = []
                    emitted_chunks: list[np.ndarray] = []
                    self.runtime.codec_streaming_session.reset()

                    def _emit_waveform(waveform: np.ndarray, *, is_pause: bool) -> None:
                        nonlocal emitted_samples_total, first_audio_emitted_at_perf
                        audio_length = int(waveform.shape[0])
                        if first_audio_emitted_at_perf is None and not is_pause:
                            first_audio_emitted_at_perf = time.perf_counter()
                        emitted_samples_total += audio_length
                        lead_seconds = 0.0
                        if first_audio_emitted_at_perf is not None:
                            elapsed_since_first_audio = max(0.0, time.perf_counter() - first_audio_emitted_at_perf)
                            lead_seconds = (emitted_samples_total / float(sample_rate)) - elapsed_since_first_audio
                        emitted_chunks.append(np.asarray(waveform, dtype=np.float32))
                        event_queue.put(
                            {
                                "type": "audio",
                                "waveform_numpy": np.asarray(waveform, dtype=np.float32),
                                "sample_rate": sample_rate,
                                "channels": channels,
                                "chunk_index": chunk_index,
                                "emitted_audio_seconds": emitted_samples_total / float(sample_rate),
                                "lead_seconds": lead_seconds,
                                "is_pause": bool(is_pause),
                            }
                        )

                    def _decode_pending(force: bool) -> None:
                        pending_count = len(pending_decode_frames)
                        if pending_count <= 0:
                            return
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
                        decoded = self.runtime.codec_streaming_session.run_frames(frame_chunk)
                        if decoded is None:
                            return
                        audio, audio_length = decoded
                        if audio_length <= 0:
                            return
                        waveform = _merge_audio_channels(
                            [audio[0, channel_index, :audio_length] for channel_index in range(audio.shape[1])]
                        )
                        _emit_waveform(waveform, is_pause=False)

                    def _on_frame(_generated_frames: list[list[int]], _step_index: int, frame: list[int]) -> None:
                        pending_decode_frames.append(list(frame))
                        _decode_pending(False)

                    try:
                        generated_frames = self.runtime.generate_audio_frames(request_rows, on_frame=_on_frame)
                        _decode_pending(True)
                    finally:
                        self.runtime.codec_streaming_session.reset()

                    chunk_waveform = _concat_waveforms(emitted_chunks)
                    all_waveforms.append(chunk_waveform)
                    all_generated_frames.extend(generated_frames)

                    if chunk_index < len(text_chunks) - 1:
                        pause_seconds = self.runtime.estimate_voice_clone_inter_chunk_pause_seconds(chunk_text)
                        pause_samples = max(0, int(round(sample_rate * pause_seconds)))
                        if pause_samples > 0:
                            pause_waveform = np.zeros((pause_samples, channels), dtype=np.float32)
                            _emit_waveform(pause_waveform, is_pause=True)
                            all_waveforms.append(pause_waveform)

                waveform = _concat_waveforms(all_waveforms)
                output_path = _write_waveform_to_wav(
                    self.output_dir / "app_onnx_stream_output.wav",
                    waveform,
                    sample_rate,
                )
                event_queue.put(
                    {
                        "type": "result",
                        **self._format_result_payload(
                            waveform=waveform,
                            sample_rate=sample_rate,
                            elapsed_seconds=time.perf_counter() - start_time,
                            audio_path=str(output_path),
                            voice=voice,
                            prompt_audio_path=prompt_audio_path,
                            text_chunks=text_chunks,
                        ),
                    }
                )
            except Exception as exc:
                event_queue.put({"type": "error", "error": str(exc)})
            finally:
                event_queue.put(None)

        worker = threading.Thread(target=_worker, name="onnx-synthesize-stream", daemon=True)
        worker.start()
        while True:
            item = event_queue.get()
            if item is None:
                break
            if str(item.get("type")) == "error":
                raise RuntimeError(str(item.get("error") or "Unknown ONNX streaming error"))
            yield item


class OnnxRequestRuntimeManager:
    _factory_model_dir: Path | None = None
    _factory_output_dir: Path | None = None
    _factory_max_new_frames: int = 375
    _factory_text_normalizer_manager: WeTextProcessingManager | None = None

    def __init__(self, default_runtime: OnnxNanoTTSServiceAdapter) -> None:
        self.default_runtime = default_runtime
        self.default_cpu_threads = max(1, int(os.cpu_count() or 1))
        self._lock = threading.Lock()
        self._execution_lock = threading.Lock()
        self._cpu_runtimes: dict[int, OnnxNanoTTSServiceAdapter] = {default_runtime.thread_count: default_runtime}

    @staticmethod
    def normalize_requested_execution_device(requested: str | None) -> str:
        del requested
        return "cpu"

    def is_dedicated_cpu_request(self, requested: str | None) -> bool:
        del requested
        return False

    def is_cpu_runtime_loaded(self) -> bool:
        with self._lock:
            return bool(self._cpu_runtimes)

    def _resolve_cpu_threads(self, cpu_threads: int | None) -> int:
        if cpu_threads is None:
            return self.default_cpu_threads
        try:
            normalized_threads = int(cpu_threads)
        except Exception:
            return self.default_cpu_threads
        if normalized_threads <= 0:
            return self.default_cpu_threads
        return max(1, normalized_threads)

    def _build_runtime_locked(self, cpu_threads: int) -> OnnxNanoTTSServiceAdapter:
        runtime = self._cpu_runtimes.get(cpu_threads)
        if runtime is not None:
            return runtime
        runtime = OnnxNanoTTSServiceAdapter(
            model_dir=self._factory_model_dir or self.default_runtime.model_dir,
            output_dir=self._factory_output_dir or self.default_runtime.output_dir,
            cpu_threads=cpu_threads,
            max_new_frames=self._factory_max_new_frames,
            text_normalizer_manager=self._factory_text_normalizer_manager,
        )
        self._cpu_runtimes[cpu_threads] = runtime
        return runtime

    def resolve_runtime(self, requested: str | None) -> tuple[OnnxNanoTTSServiceAdapter, str]:
        del requested
        return self.default_runtime, "cpu"

    @contextmanager
    def _locked_runtime(self, cpu_threads: int | None) -> Iterator[tuple[OnnxNanoTTSServiceAdapter, str, int]]:
        resolved_cpu_threads = self._resolve_cpu_threads(cpu_threads)
        with self._lock:
            runtime = self._build_runtime_locked(resolved_cpu_threads)
        with self._execution_lock:
            yield runtime, "cpu", resolved_cpu_threads

    def call_with_runtime(
        self,
        *,
        requested_execution_device: str | None,
        cpu_threads: int | None,
        callback,
    ) -> tuple[object, str, int]:
        del requested_execution_device
        with self._locked_runtime(cpu_threads) as (runtime, execution_device, resolved_cpu_threads):
            return callback(runtime), execution_device, resolved_cpu_threads

    def iter_with_runtime(
        self,
        *,
        requested_execution_device: str | None,
        cpu_threads: int | None,
        factory,
    ) -> Iterator[tuple[object, str, int]]:
        del requested_execution_device
        with self._locked_runtime(cpu_threads) as (runtime, execution_device, resolved_cpu_threads):
            for item in factory(runtime):
                yield item, execution_device, resolved_cpu_threads


def _render_index_html_onnx(
    *,
    request,
    runtime,
    demo_entries,
    warmup_status: str,
    text_normalization_status: str,
) -> str:
    html = _LEGACY_RENDER_INDEX_HTML(
        request=request,
        runtime=runtime,
        demo_entries=demo_entries,
        warmup_status=warmup_status,
        text_normalization_status=text_normalization_status,
    )
    html = html.replace("MOSS-TTS-Nano Demo", "MOSS-TTS-Nano ONNX Demo")
    html = html.replace(
        '<label for="attn-implementation">Attention Backend</label>\n'
        '              <select id="attn-implementation">\n'
        '                <option value="model_default">model_default</option>\n'
        '                <option value="sdpa">sdpa</option>\n'
        '                <option value="eager">eager</option>\n'
        '              </select>',
        '<label for="attn-implementation">Sampling Mode</label>\n'
        '              <select id="attn-implementation">\n'
        '                <option value="fixed">fixed</option>\n'
        '                <option value="full">full</option>\n'
        '                <option value="greedy">greedy</option>\n'
        '              </select>\n'
        '              <div id="onnx-sampling-mode-note" class="meta">fixed uses the baked ONNX sampling constants.</div>',
    )
    html = html.replace(
        '<label><input id="do-sample" type="checkbox" checked> Do Sample</label>',
        '<label><input id="do-sample" type="checkbox" checked disabled> Do Sample (derived from Sampling Mode)</label>',
    )
    html = html.replace(
        'This app is CPU-only. CPU Threads maps to torch.set_num_threads for that request.',
        'This app is CPU-only. CPU Threads selects the cached ONNX runtime instance for that request.',
    )
    html = html.replace(
        '</style>',
        '    .field.disabled-field {\n'
        '      opacity: 0.5;\n'
        '    }\n'
        '    .field.disabled-field input {\n'
        '      cursor: not-allowed;\n'
        '      background: #f4f6fb;\n'
        '    }\n'
        '</style>',
        1,
    )
    html = html.replace(
        '    document.getElementById("attn-implementation").value = DEFAULT_ATTN_IMPLEMENTATION;\n',
        '    document.getElementById("attn-implementation").value = DEFAULT_ATTN_IMPLEMENTATION;\n'
        '    const onnxSamplingModeSelect = document.getElementById("attn-implementation");\n'
        '    const onnxDoSampleToggle = document.getElementById("do-sample");\n'
        '    const onnxSamplingModeNote = document.getElementById("onnx-sampling-mode-note");\n'
        '    const onnxSamplingParamIds = [\n'
        '      "text-temperature",\n'
        '      "text-top-p",\n'
        '      "text-top-k",\n'
        '      "audio-temperature",\n'
        '      "audio-top-p",\n'
        '      "audio-top-k",\n'
        '      "audio-repetition-penalty"\n'
        '    ];\n'
        '    function syncOnnxSamplingUi() {\n'
        '      const mode = (onnxSamplingModeSelect && onnxSamplingModeSelect.value) || "fixed";\n'
        '      const samplingParamsEnabled = mode === "full";\n'
        '      if (onnxDoSampleToggle) {\n'
        '        onnxDoSampleToggle.checked = mode !== "greedy";\n'
        '      }\n'
        '      for (const id of onnxSamplingParamIds) {\n'
        '        const input = document.getElementById(id);\n'
        '        if (!input) continue;\n'
        '        input.disabled = !samplingParamsEnabled;\n'
        '        const field = input.closest(".field");\n'
        '        if (field) field.classList.toggle("disabled-field", !samplingParamsEnabled);\n'
        '      }\n'
        '      if (onnxSamplingModeNote) {\n'
        '        if (mode === "full") {\n'
        '          onnxSamplingModeNote.textContent = "full uses the current page sampling hyperparameters.";\n'
        '        } else if (mode === "fixed") {\n'
        '          onnxSamplingModeNote.textContent = "fixed uses the baked ONNX sampling constants and ignores the hyperparameter inputs below.";\n'
        '        } else {\n'
        '          onnxSamplingModeNote.textContent = "greedy disables sampling and ignores the hyperparameter inputs below.";\n'
        '        }\n'
        '      }\n'
        '    }\n'
        '    if (onnxSamplingModeSelect) {\n'
        '      onnxSamplingModeSelect.addEventListener("change", syncOnnxSamplingUi);\n'
        '      syncOnnxSamplingUi();\n'
        '    }\n',
        1,
    )
    return html


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOSS-TTS-Nano ONNX web demo")
    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "browser_onnx model directory. If omitted, the app uses "
            f"{DEFAULT_BROWSER_ONNX_MODEL_DIR} and auto-downloads the ONNX assets on first run."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(APP_DIR / "generated_audio"),
        help="Directory for generated wav files.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--cpu-threads", type=int, default=max(1, int(os.cpu_count() or 1)))
    parser.add_argument("--max-new-frames", type=int, default=375)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )

    text_normalizer_manager = WeTextProcessingManager()
    text_normalizer_manager.start()
    output_dir = Path(args.output_dir).expanduser().resolve()
    runtime = OnnxNanoTTSServiceAdapter(
        model_dir=args.model_dir,
        output_dir=output_dir,
        cpu_threads=args.cpu_threads,
        max_new_frames=args.max_new_frames,
        text_normalizer_manager=text_normalizer_manager,
    )
    warmup_manager = legacy_app.WarmupManager(runtime, text_normalizer_manager=text_normalizer_manager)
    warmup_manager.start()

    OnnxRequestRuntimeManager._factory_model_dir = runtime.model_dir
    OnnxRequestRuntimeManager._factory_output_dir = output_dir
    OnnxRequestRuntimeManager._factory_max_new_frames = int(args.max_new_frames)
    OnnxRequestRuntimeManager._factory_text_normalizer_manager = text_normalizer_manager
    legacy_app.RequestRuntimeManager = OnnxRequestRuntimeManager
    legacy_app._render_index_html = _render_index_html_onnx

    vscode_proxy_uri = os.getenv("VSCODE_PROXY_URI", "")
    root_path = legacy_app._resolve_vscode_root_path(vscode_proxy_uri, args.port)
    logging.info("root_path=%s", root_path)
    if args.share:
        logging.warning("--share is ignored by the FastAPI-based ONNX app.")

    app = legacy_app._build_app(runtime, warmup_manager, text_normalizer_manager, root_path)
    app.title = "MOSS-TTS-Nano ONNX Demo"
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        root_path=root_path or "",
    )


if __name__ == "__main__":
    main()
