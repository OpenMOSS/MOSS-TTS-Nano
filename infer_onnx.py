from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from onnx_tts_runtime import (
    DEFAULT_BROWSER_ONNX_MODEL_DIR,
    DEFAULT_BROWSER_ONNX_OUTPUT_PATH,
    OnnxTtsRuntime,
)


def set_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native onnxruntime CPU inference on browser_onnx exported assets.")
    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "browser_onnx model directory. If omitted, the script uses "
            f"{DEFAULT_BROWSER_ONNX_MODEL_DIR} and auto-downloads the ONNX assets on first run."
        ),
    )
    parser.add_argument(
        "--output-audio-path",
        default=str(DEFAULT_BROWSER_ONNX_OUTPUT_PATH),
        help="Where to save the generated waveform.",
    )
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to synthesize.")
    text_group.add_argument("--text-file", help="Path to a UTF-8 text file to synthesize.")
    parser.add_argument(
        "--voice",
        default="Junhao",
        help="Built-in voice preset name used only when no reference audio path is provided.",
    )
    parser.add_argument(
        "--prompt-audio-path",
        "--reference-audio-path",
        dest="prompt_audio_path",
        default=None,
        help="Local reference audio path used directly for voice cloning. When provided, it overrides --voice.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("greedy", "fixed", "full"),
        default="fixed",
        help="greedy=do_sample false, fixed=fixed hyperparameter sampled frame, full=host sampled full frame.",
    )
    parser.add_argument(
        "--do-sample",
        type=int,
        nargs="?",
        const=1,
        default=1,
        choices=[0, 1],
        help="Whether to sample. If 0, sample_mode is forced to greedy.",
    )
    parser.add_argument(
        "--realtime-streaming-decode",
        type=int,
        nargs="?",
        const=1,
        default=1,
        choices=[0, 1],
        help="Use codec streaming decode path internally instead of full decode.",
    )
    parser.add_argument("--cpu-threads", type=int, default=4, help="onnxruntime intra-op thread count.")
    parser.add_argument("--max-new-frames", type=int, default=375, help="Maximum generated audio frames.")
    parser.add_argument("--voice-clone-max-text-tokens", type=int, default=75, help="Chunk long text by token budget.")
    parser.add_argument("--text-temperature", type=float, default=1.0, help="Text-layer sampling temperature.")
    parser.add_argument("--text-top-p", type=float, default=1.0, help="Text-layer top-p sampling.")
    parser.add_argument("--text-top-k", type=int, default=50, help="Text-layer top-k sampling.")
    parser.add_argument("--audio-temperature", type=float, default=0.8, help="Audio-layer sampling temperature.")
    parser.add_argument("--audio-top-p", type=float, default=0.95, help="Audio-layer top-p sampling.")
    parser.add_argument("--audio-top-k", type=int, default=25, help="Audio-layer top-k sampling.")
    parser.add_argument(
        "--audio-repetition-penalty",
        type=float,
        default=1.2,
        help="Audio-layer repetition penalty.",
    )
    parser.add_argument(
        "--enable-wetext-processing",
        type=int,
        nargs="?",
        const=1,
        default=1,
        choices=[0, 1],
        help="Enable WeTextProcessing text normalization before inference.",
    )
    parser.add_argument(
        "--disable-wetext-processing",
        action="store_true",
        help="Disable WeTextProcessing even if enabled above.",
    )
    parser.add_argument(
        "--enable-normalize-tts-text",
        dest="enable_normalize_tts_text",
        action="store_true",
        default=True,
        help="Enable normalize_tts_text robust cleanup before inference.",
    )
    parser.add_argument(
        "--disable-normalize-tts-text",
        dest="disable_normalize_tts_text",
        action="store_true",
        help="Disable normalize_tts_text robust cleanup before inference.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--print-voice-clone-text-chunks",
        action="store_true",
        help="Print the effective chunked text before synthesis.",
    )
    return parser.parse_args(argv)


def resolve_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return str(args.text)
    return Path(args.text_file).read_text(encoding="utf-8")


def maybe_print_voice_clone_text_chunks(runtime: OnnxTtsRuntime, text: str, max_tokens: int) -> None:
    chunks = runtime.split_voice_clone_text(text, max_tokens=max_tokens)
    effective_chunks = chunks if len(chunks) > 1 else [text]
    print("Voice clone text chunks")
    print("----------------------")
    print(f"max_tokens={max_tokens} chunks={len(effective_chunks)}")
    for chunk_index, chunk_text in enumerate(effective_chunks, start=1):
        print(f"[chunk {chunk_index}]")
        print(chunk_text)
        print()


def main(argv: Optional[Sequence[str]] = None) -> dict[str, object]:
    set_logging()
    args = parse_args(argv)
    runtime = OnnxTtsRuntime(
        model_dir=args.model_dir,
        thread_count=args.cpu_threads,
        max_new_frames=args.max_new_frames,
        do_sample=bool(args.do_sample),
        sample_mode=args.sample_mode,
    )
    generation_defaults = runtime.manifest["generation_defaults"]
    generation_defaults["text_temperature"] = float(args.text_temperature)
    generation_defaults["text_top_p"] = float(args.text_top_p)
    generation_defaults["text_top_k"] = int(args.text_top_k)
    generation_defaults["audio_temperature"] = float(args.audio_temperature)
    generation_defaults["audio_top_p"] = float(args.audio_top_p)
    generation_defaults["audio_top_k"] = int(args.audio_top_k)
    generation_defaults["audio_repetition_penalty"] = float(args.audio_repetition_penalty)
    raw_text = resolve_text(args)
    enable_wetext = bool(args.enable_wetext_processing) and not bool(args.disable_wetext_processing)
    enable_normalize_tts_text = bool(args.enable_normalize_tts_text) and not bool(args.disable_normalize_tts_text)
    prepared = runtime.prepare_synthesis_text(
        text=raw_text,
        voice=str(args.voice or ""),
        enable_wetext=enable_wetext,
        enable_normalize_tts_text=enable_normalize_tts_text,
    )
    prepared_text = str(prepared["text"])
    logging.info(
        "text normalization method=%s language=%s text_chars=%d",
        prepared["normalization_method"],
        prepared["text_normalization_language"] or "n/a",
        len(prepared_text),
    )
    if args.print_voice_clone_text_chunks:
        maybe_print_voice_clone_text_chunks(runtime, prepared_text, args.voice_clone_max_text_tokens)
    if args.prompt_audio_path:
        logging.info("using direct reference audio path for voice cloning: %s", args.prompt_audio_path)
    else:
        logging.info("using built-in voice preset: %s", args.voice)
    result = runtime.synthesize(
        text=raw_text,
        voice=args.voice,
        prompt_audio_path=args.prompt_audio_path,
        output_audio_path=args.output_audio_path,
        sample_mode=args.sample_mode,
        do_sample=bool(args.do_sample),
        streaming=bool(args.realtime_streaming_decode),
        max_new_frames=args.max_new_frames,
        voice_clone_max_text_tokens=args.voice_clone_max_text_tokens,
        enable_wetext=enable_wetext,
        enable_normalize_tts_text=enable_normalize_tts_text,
        seed=args.seed,
    )
    logging.info(
        "saved generated audio to %s sample_rate=%s frames=%s sample_mode=%s streaming=%s",
        result["audio_path"],
        result["sample_rate"],
        int(result["audio_token_ids"].shape[0]),
        result["sample_mode"],
        result["streaming"],
    )
    return result


if __name__ == "__main__":
    main()
