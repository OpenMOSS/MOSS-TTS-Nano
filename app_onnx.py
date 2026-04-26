from __future__ import annotations

import argparse
import concurrent.futures
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
from ort_cpu_runtime import CodecStreamingDecodeSession
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
        # Build session pool for parallel chunk processing.
        # Each worker gets its own ONNX sessions to avoid contention.
        total_cpus = os.cpu_count() or 4
        per_worker_threads = max(1, int(cpu_threads))
        self._parallel_workers = max(1, total_cpus // per_worker_threads)
        self._parallel_sessions: list[dict[str, object]] = []
        self._parallel_rlocks: list[threading.RLock] = []
        if self._parallel_workers > 1:
            worker_threads = max(1, per_worker_threads)
            for _ in range(self._parallel_workers):
                sessions = self.runtime._create_sessions_with_threads(worker_threads)
                self._parallel_sessions.append(sessions)
                self._parallel_rlocks.append(threading.RLock())
            logging.info(
                "ONNX parallel pool workers=%d per_worker_threads=%d total_cpus=%d",
                self._parallel_workers, worker_threads, total_cpus,
            )

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
        max_new_frames: int = 375,
        voice_clone_max_text_tokens: int = 75,
        voice_clone_max_memory_per_sample_gb: float = 1.0,
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
        del mode, tts_max_batch_size, codec_max_batch_size, voice_clone_max_memory_per_sample_gb
        event_queue: "queue.Queue[dict[str, object] | None]" = queue.Queue(maxsize=128)

        def _process_single_chunk(
            chunk_index: int,
            chunk_text: str,
            total_chunks: int,
            prompt_audio_codes: list[list[int]],
            rng: np.random.Generator,
            pool_index: int,
        ) -> dict[str, object]:
            """Process one text chunk using pooled sessions for thread-safety."""
            _chunk_t0 = time.perf_counter()

            if pool_index >= 0 and self._parallel_sessions:
                # Acquire pool slot and swap sessions + rng atomically
                with self._parallel_rlocks[pool_index]:
                    original_sessions = self.runtime.sessions
                    original_rng = self.runtime.rng
                    self.runtime.sessions = self._parallel_sessions[pool_index]
                    self.runtime.rng = rng
                    try:
                        return _run_chunk(chunk_index, chunk_text, total_chunks, _chunk_t0, pool_index, prompt_audio_codes)
                    finally:
                        self.runtime.sessions = original_sessions
                        self.runtime.rng = original_rng
            else:
                # No pool — use runtime sessions directly (single-threaded path)
                original_rng = self.runtime.rng
                self.runtime.rng = rng
                try:
                    return _run_chunk(chunk_index, chunk_text, total_chunks, _chunk_t0, -1, prompt_audio_codes)
                finally:
                    self.runtime.rng = original_rng

        def _run_chunk(
            chunk_index: int,
            chunk_text: str,
            total_chunks: int,
            _chunk_t0: float,
            pool_index: int,
            prompt_audio_codes: list[list[int]],
        ) -> dict[str, object]:
            text_token_ids = self.runtime.encode_text(chunk_text)
            request_rows = self.runtime.build_voice_clone_request_rows(prompt_audio_codes, text_token_ids)
            codec_session = CodecStreamingDecodeSession(
                codec_meta=self.runtime.codec_meta,
                session=self.runtime.sessions["codec_decode_step"],
            )
            generated_frames: list[list[int]] = []

            def _on_frame(_gf: list[list[int]], _si: int, frame: list[int]) -> None:
                generated_frames.append(list(frame))

            _gen_t0 = time.perf_counter()
            all_frames = self.runtime.generate_audio_frames(request_rows, on_frame=_on_frame)
            _gen_elapsed = time.perf_counter() - _gen_t0

            _dec_t0 = time.perf_counter()
            decoded_waveforms: list[np.ndarray] = []
            if all_frames:
                decoded = codec_session.run_frames(all_frames)
                if decoded is not None:
                    audio, audio_length = decoded
                    if audio_length > 0:
                        waveform = _merge_audio_channels(
                            [audio[0, ch, :audio_length] for ch in range(audio.shape[1])]
                        )
                        decoded_waveforms.append(waveform)
            _decode_elapsed = time.perf_counter() - _dec_t0
            _chunk_elapsed = time.perf_counter() - _chunk_t0

            chunk_waveform = _concat_waveforms(decoded_waveforms) if decoded_waveforms else np.zeros((0,), dtype=np.float32)
            logging.info(
                "ONNX timing chunk=%d/%d generate=%.3fs decode=%.3fs frames=%d total=%.3fs pool=%d text=%r",
                chunk_index + 1, total_chunks, _gen_elapsed, _decode_elapsed,
                len(all_frames), _chunk_elapsed, pool_index, chunk_text,
            )
            return {
                "chunk_index": chunk_index,
                "chunk_text": chunk_text,
                "waveform": chunk_waveform,
                "frames": len(all_frames),
            }

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
                _t0 = time.perf_counter()
                prompt_audio_codes = self.runtime.resolve_prompt_audio_codes(voice=voice, prompt_audio_path=prompt_audio_path)
                logging.info("ONNX timing resolve_prompt_audio_codes %.3fs", time.perf_counter() - _t0)
                _t0 = time.perf_counter()
                text_chunks = self.runtime.split_voice_clone_text(str(text or ""), max_tokens=int(voice_clone_max_text_tokens))
                logging.info("ONNX timing split_voice_clone_text %.3fs count=%d chunks=%r", time.perf_counter() - _t0, len(text_chunks), text_chunks)
                sample_rate = int(self.runtime.codec_meta["codec_config"]["sample_rate"])
                channels = int(self.runtime.codec_meta["codec_config"]["channels"])

                num_chunks = len(text_chunks)
                logging.info(
                    "ONNX dispatch num_chunks=%d parallel_workers=%d pool_sessions=%d",
                    num_chunks, self._parallel_workers, len(self._parallel_sessions),
                )
                # Parallel chunk processing when multiple chunks
                if num_chunks > 1 and self._parallel_workers > 1:
                    max_workers = min(num_chunks, self._parallel_workers)
                    logging.info(
                        "ONNX parallel chunks=%d max_workers=%d pool_size=%d",
                        num_chunks, max_workers, len(self._parallel_sessions),
                    )
                    # Create per-chunk RNGs from the current rng
                    base_seed = int.from_bytes(self.runtime.rng.bytes(4), "little")
                    chunk_rngs = [np.random.default_rng(base_seed + i) for i in range(num_chunks)]

                    # Simple round-robin pool slot assignment
                    _pool_counter = {"value": 0}
                    _pool_lock = threading.Lock()

                    def _next_pool_index() -> int:
                        with _pool_lock:
                            idx = _pool_counter["value"] % len(self._parallel_sessions)
                            _pool_counter["value"] += 1
                            return idx

                    chunk_results: list[dict[str, object] | None] = [None] * num_chunks
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(
                                _process_single_chunk,
                                i, text_chunks[i], num_chunks,
                                prompt_audio_codes, chunk_rngs[i],
                                _next_pool_index(),
                            ): i
                            for i in range(num_chunks)
                        }
                        for future in concurrent.futures.as_completed(futures):
                            idx = futures[future]
                            try:
                                chunk_results[idx] = future.result()
                            except Exception as exc:
                                chunk_results[idx] = {"chunk_index": idx, "error": str(exc)}

                    # Emit results in original chunk order
                    all_waveforms: list[np.ndarray] = []
                    emitted_samples_total = 0
                    for idx in range(num_chunks):
                        result = chunk_results[idx]
                        if result is None or "error" in result:
                            error_msg = result.get("error", "unknown") if result else "no result"
                            logging.error("ONNX chunk=%d failed: %s", idx + 1, error_msg)
                            continue
                        waveform = result["waveform"]
                        if waveform.shape[0] > 0:
                            all_waveforms.append(waveform)
                            event_queue.put({
                                "type": "audio",
                                "waveform_numpy": np.asarray(waveform, dtype=np.float32),
                                "sample_rate": sample_rate,
                                "channels": channels,
                                "chunk_index": idx,
                                "emitted_audio_seconds": (emitted_samples_total + waveform.shape[0]) / float(sample_rate),
                                "lead_seconds": 0.0,
                                "is_pause": False,
                            })
                            emitted_samples_total += waveform.shape[0]
                        # Inter-chunk pause
                        if idx < num_chunks - 1:
                            pause_seconds = self.runtime.estimate_voice_clone_inter_chunk_pause_seconds(text_chunks[idx])
                            pause_samples = max(0, int(round(sample_rate * pause_seconds)))
                            if pause_samples > 0:
                                pause_waveform = np.zeros((pause_samples, channels), dtype=np.float32)
                                all_waveforms.append(pause_waveform)
                    waveform = _concat_waveforms(all_waveforms)
                else:
                    # Single chunk or no pool: sequential path
                    emitted_samples_total = 0
                    all_waveforms: list[np.ndarray] = []
                    all_generated_frames: list[list[int]] = []
                    for chunk_index, chunk_text in enumerate(text_chunks):
                        pool_idx = 0 if self._parallel_sessions else -1
                        chunk_result = _process_single_chunk(
                            chunk_index, chunk_text, num_chunks,
                            prompt_audio_codes,
                            np.random.default_rng(int.from_bytes(self.runtime.rng.bytes(4), "little") + chunk_index),
                            pool_idx,
                        )
                        if "error" in chunk_result:
                            logging.error("ONNX chunk=%d failed: %s", chunk_index + 1, chunk_result.get("error"))
                            continue
                        waveform = chunk_result["waveform"]
                        if waveform.shape[0] > 0:
                            all_waveforms.append(waveform)
                            event_queue.put({
                                "type": "audio",
                                "waveform_numpy": np.asarray(waveform, dtype=np.float32),
                                "sample_rate": sample_rate,
                                "channels": channels,
                                "chunk_index": chunk_index,
                                "emitted_audio_seconds": waveform.shape[0] / float(sample_rate),
                                "lead_seconds": 0.0,
                                "is_pause": False,
                            })
                            emitted_samples_total += waveform.shape[0]
                        if chunk_index < num_chunks - 1:
                            pause_seconds = self.runtime.estimate_voice_clone_inter_chunk_pause_seconds(chunk_text)
                            pause_samples = max(0, int(round(sample_rate * pause_seconds)))
                            if pause_samples > 0:
                                pause_waveform = np.zeros((pause_samples, channels), dtype=np.float32)
                                all_waveforms.append(pause_waveform)
                    waveform = _concat_waveforms(all_waveforms)
                    if "error" not in chunk_result:
                        waveform = chunk_result["waveform"]
                        if waveform.shape[0] > 0:
                            all_waveforms.append(waveform)
                            event_queue.put({
                                "type": "audio",
                                "waveform_numpy": np.asarray(waveform, dtype=np.float32),
                                "sample_rate": sample_rate,
                                "channels": channels,
                                "chunk_index": 0,
                                "emitted_audio_seconds": waveform.shape[0] / float(sample_rate),
                                "lead_seconds": 0.0,
                                "is_pause": False,
                            })
                            emitted_samples_total = waveform.shape[0]
                    waveform = _concat_waveforms(all_waveforms)

                _total_elapsed = time.perf_counter() - start_time
                logging.info(
                    "ONNX timing total %.3fs chunks=%d audio_seconds=%.2fs",
                    _total_elapsed, num_chunks, emitted_samples_total / float(sample_rate) if sample_rate else 0,
                )
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
                logging.exception("ONNX synthesize_stream error")
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
        self.default_cpu_threads = default_runtime.thread_count
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


def _patch_torchaudio_backend() -> None:
    """Patch torchaudio to avoid the SoX backend, which segfaults on some systems."""
    try:
        import torchaudio
        _original_load = torchaudio.load
        _original_save = torchaudio.save

        def _load_with_soundfile(uri, *args, backend=None, **kwargs):
            if backend is None:
                backend = "soundfile"
            return _original_load(uri, *args, backend=backend, **kwargs)

        def _save_with_soundfile(uri, src, sample_rate, *args, backend=None, **kwargs):
            if backend is None:
                backend = "soundfile"
            return _original_save(uri, src, sample_rate, *args, backend=backend, **kwargs)

        torchaudio.load = _load_with_soundfile
        torchaudio.save = _save_with_soundfile
    except ImportError:
        pass


def main(argv: Optional[Sequence[str]] = None) -> None:
    _patch_torchaudio_backend()

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
