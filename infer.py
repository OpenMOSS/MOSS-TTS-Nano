from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import torch
from transformers import AutoModelForCausalLM

from moss_tts_nano.defaults import (
    DEFAULT_AUDIO_TOKENIZER_PATH,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_OUTPUT_DIR,
)
from text_normalization_pipeline import WeTextProcessingManager, prepare_tts_request_texts

MOSS_AUDIO_TOKENIZER_TYPE = "moss-audio-tokenizer-nano"
DEFAULT_AUDIO_TOKENIZER_PRETRAINED_NAME_OR_PATH = DEFAULT_AUDIO_TOKENIZER_PATH
DEFAULT_OUTPUT_AUDIO_PATH = DEFAULT_OUTPUT_DIR / "infer_output.wav"


def set_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )


def waiting_for_debug(ip: str, port: int) -> None:
    import debugpy

    logging.info("waiting for debugger attach at %s:%s", ip, port)
    debugpy.listen((ip, port))
    debugpy.wait_for_client()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MOSS-TTS-Nano inference from a HF-style checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Checkpoint directory loadable by from_pretrained().",
    )
    parser.add_argument(
        "--output-audio-path",
        default=str(DEFAULT_OUTPUT_AUDIO_PATH),
        help="Where to save the generated waveform.",
    )

    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to synthesize.")
    text_group.add_argument("--text-file", help="Path to a UTF-8 text file to synthesize.")

    prompt_text_group = parser.add_mutually_exclusive_group(required=False)
    prompt_text_group.add_argument("--prompt-text", help="Reference transcript used by continuation mode.")
    prompt_text_group.add_argument("--prompt-text-file", help="UTF-8 reference transcript file used by continuation mode.")

    parser.add_argument("--text-tokenizer-path", default=None, help="Override the checkpoint-bundled text tokenizer.")
    parser.add_argument(
        "--audio-tokenizer-pretrained-name-or-path",
        default=DEFAULT_AUDIO_TOKENIZER_PRETRAINED_NAME_OR_PATH,
        help="HF path or repo id for the audio tokenizer. Defaults to OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano.",
    )
    parser.add_argument(
        "--mode",
        default="voice_clone",
        choices=("continuation", "voice_clone"),
        help=(
            "Inference mode. continuation: plain TTS or prompt_text + prompt_speech continuation; "
            "voice_clone: prompt_speech + target_text."
        ),
    )
    parser.add_argument(
        "--prompt-audio-path",
        default=None,
        help="Reference speech used by continuation-with-prompt or voice_clone mode.",
    )
    parser.add_argument(
        "--reference-audio-path",
        default=None,
        help="Compatibility alias for --prompt-audio-path.",
    )
    parser.add_argument("--device", default="auto", help="Device to run on, for example auto/cpu/cuda/cuda:0.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Weights dtype after loading.",
    )
    parser.add_argument(
        "--nq",
        type=int,
        default=None,
        help="Only use the first nq RVQ layers for prompt encode, model decoding, and audio decode.",
    )
    parser.add_argument("--max-new-frames", type=int, default=375, help="Maximum number of audio frames to generate.")
    parser.add_argument(
        "--voice-clone-max-text-tokens",
        type=int,
        default=75,
        help=(
            "Only for voice_clone mode: pocket-tts style sentence chunking target token budget. "
            "Set <= 0 to disable chunking and keep single-pass generation."
        ),
    )
    parser.add_argument(
        "--voice-clone-max-memory-per-sample-gb",
        type=float,
        default=1.0,
        help=(
            "Only for chunked voice_clone mode on CUDA: maximum free-memory budget used to size each sample "
            "when auto-selecting the chunk batch size."
        ),
    )
    parser.add_argument(
        "--print-voice-clone-text-chunks",
        action="store_true",
        help="Print the effective voice_clone text chunks before generation.",
    )
    parser.add_argument(
        "--do-sample",
        type=int,
        nargs="?",
        const=1,
        default=1,
        choices=[0, 1],
        help="Sample instead of greedy decoding. Accepts bare --do-sample or --do-sample 0/1.",
    )
    parser.add_argument("--text-temperature", type=float, default=None, help="Text-layer sampling temperature. Default: 1.5.")
    parser.add_argument("--text-top-p", type=float, default=None, help="Text-layer top-p sampling. Default: 1.0.")
    parser.add_argument("--text-top-k", type=int, default=None, help="Text-layer top-k sampling. Default: 50.")
    parser.add_argument("--audio-temperature", type=float, default=None, help="Audio-layer sampling temperature. Default: 1.7.")
    parser.add_argument("--audio-top-p", type=float, default=None, help="Audio-layer top-p sampling. Default: 0.8.")
    parser.add_argument("--audio-top-k", type=int, default=None, help="Audio-layer top-k sampling. Default: 25.")
    parser.add_argument(
        "--audio-repetition-penalty",
        type=float,
        default=None,
        help="Audio-layer repetition penalty. Default: 1.0.",
    )
    parser.add_argument(
        "--enable-wetext-processing",
        type=int,
        nargs="?",
        const=1,
        default=1,
        choices=[0, 1],
        help="Enable WeTextProcessing normalization before inference.",
    )
    parser.add_argument(
        "--disable-wetext-processing",
        action="store_true",
        help="Disable WeTextProcessing normalization even if --enable-wetext-processing 1 is set.",
    )
    parser.add_argument(
        "--enable-normalize-tts-text",
        "--enable-robust-text-normalization",
        dest="enable_normalize_tts_text",
        action="store_true",
        default=True,
        help="Enable the repository's normalize_tts_text robust cleanup before inference.",
    )
    parser.add_argument(
        "--disable-normalize-tts-text",
        "--disable-robust-text-normalization",
        dest="disable_normalize_tts_text",
        action="store_true",
        help="Disable the repository's normalize_tts_text robust cleanup before inference.",
    )
    parser.add_argument("--temperature", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--top-k", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--top-p", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--repetition-penalty", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for sampling.")

    parser.add_argument("--debug_ip", type=str, default="localhost")
    parser.add_argument("--debug_port", type=int, default=32431)
    parser.add_argument("--debug", type=int, default=0, help="Run with debug-friendly settings.")
    return parser.parse_args(argv)


def resolve_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    return Path(args.text_file).read_text(encoding="utf-8")


def resolve_prompt_text(args: argparse.Namespace) -> Optional[str]:
    if args.prompt_text is not None:
        return args.prompt_text
    if args.prompt_text_file is not None:
        return Path(args.prompt_text_file).read_text(encoding="utf-8")
    return None


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


def load_model(checkpoint: str, device: torch.device, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=True,
    )
    model.to(device=device, dtype=dtype)
    model._set_attention_implementation("sdpa")
    model.eval()
    return model


def resolve_sampling_kwargs(args: argparse.Namespace) -> dict[str, object]:
    text_temperature = 1.0 if args.text_temperature is None else float(args.text_temperature)
    text_top_p = 1.0 if args.text_top_p is None else float(args.text_top_p)
    text_top_k = 50 if args.text_top_k is None else int(args.text_top_k)
    audio_temperature = 0.8 if args.audio_temperature is None else float(args.audio_temperature)
    audio_top_p = 0.95 if args.audio_top_p is None else float(args.audio_top_p)
    audio_top_k = 25 if args.audio_top_k is None else int(args.audio_top_k)
    audio_repetition_penalty = (
        1.2 if args.audio_repetition_penalty is None else float(args.audio_repetition_penalty)
    )

    if args.temperature is not None:
        if args.text_temperature is None:
            text_temperature = float(args.temperature)
        if args.audio_temperature is None:
            audio_temperature = float(args.temperature)
    if args.top_p is not None:
        if args.text_top_p is None:
            text_top_p = float(args.top_p)
        if args.audio_top_p is None:
            audio_top_p = float(args.top_p)
    if args.top_k is not None:
        if args.text_top_k is None:
            text_top_k = int(args.top_k)
        if args.audio_top_k is None:
            audio_top_k = int(args.top_k)
    if args.repetition_penalty is not None and args.audio_repetition_penalty is None:
        audio_repetition_penalty = float(args.repetition_penalty)

    return {
        "text_temperature": text_temperature,
        "text_top_p": text_top_p,
        "text_top_k": text_top_k,
        "audio_temperature": audio_temperature,
        "audio_top_p": audio_top_p,
        "audio_top_k": audio_top_k,
        "audio_repetition_penalty": audio_repetition_penalty,
    }


def maybe_print_voice_clone_text_chunks(
    *,
    model,
    args: argparse.Namespace,
    text: str,
) -> None:
    if args.mode != "voice_clone" or not args.print_voice_clone_text_chunks:
        return

    text_tokenizer = model._load_text_tokenizer(
        text_tokenizer=None,
        text_tokenizer_path=args.text_tokenizer_path,
    )
    split_chunks = model._split_text_into_best_sentences(
        text_tokenizer=text_tokenizer,
        text=text,
        max_tokens=args.voice_clone_max_text_tokens,
    )
    effective_chunks = split_chunks if len(split_chunks) > 1 else [text]

    print("Voice clone text chunks")
    print("----------------------")
    print(
        f"max_tokens={args.voice_clone_max_text_tokens} "
        f"split_chunks={len(split_chunks)} effective_chunks={len(effective_chunks)}"
    )
    for chunk_index, chunk_text in enumerate(effective_chunks, start=1):
        print(f"[chunk {chunk_index}]")
        print(chunk_text)
        print()


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


def main(argv: Optional[Sequence[str]] = None) -> dict[str, object]:
    _patch_torchaudio_backend()
    set_logging()
    args = parse_args(argv)
    if args.debug == 1:
        waiting_for_debug(args.debug_ip, args.debug_port)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    model = load_model(args.checkpoint, device=device, dtype=dtype)
    sampling_kwargs = resolve_sampling_kwargs(args)
    raw_text = resolve_text(args)
    raw_prompt_text = resolve_prompt_text(args) or ""
    enable_wetext_processing = bool(args.enable_wetext_processing) and not bool(args.disable_wetext_processing)
    enable_normalize_tts_text = bool(args.enable_normalize_tts_text) and not bool(args.disable_normalize_tts_text)
    text_normalizer_manager = None
    if enable_wetext_processing:
        text_normalizer_manager = WeTextProcessingManager()
        snapshot = text_normalizer_manager.ensure_ready()
        if not snapshot.ready:
            raise RuntimeError(snapshot.error or snapshot.message)
        logging.info("WeTextProcessing ready for infer.py status=%s", snapshot.message)
    prepared_texts = prepare_tts_request_texts(
        text=raw_text,
        prompt_text=raw_prompt_text,
        voice="",
        enable_wetext=enable_wetext_processing,
        enable_normalize_tts_text=enable_normalize_tts_text,
        text_normalizer_manager=text_normalizer_manager,
    )
    text = str(prepared_texts["text"])
    prompt_text = str(prepared_texts["prompt_text"]).strip() or None
    logging.info(
        "text normalization method=%s language=%s text_chars=%d prompt_chars=%d",
        prepared_texts["normalization_method"],
        prepared_texts["text_normalization_language"] or "n/a",
        len(text),
        len(prompt_text or ""),
    )
    maybe_print_voice_clone_text_chunks(model=model, args=args, text=text)
    logging.info("running inference mode=%s", args.mode)
    result = model.inference(
        text=text,
        output_audio_path=args.output_audio_path,
        mode=args.mode,
        prompt_text=prompt_text,
        prompt_audio_path=args.prompt_audio_path,
        reference_audio_path=args.reference_audio_path,
        text_tokenizer_path=args.text_tokenizer_path,
        audio_tokenizer_type=MOSS_AUDIO_TOKENIZER_TYPE,
        audio_tokenizer_pretrained_name_or_path=args.audio_tokenizer_pretrained_name_or_path,
        device=device,
        nq=args.nq,
        max_new_frames=args.max_new_frames,
        voice_clone_max_text_tokens=args.voice_clone_max_text_tokens,
        voice_clone_max_memory_per_sample_gb=args.voice_clone_max_memory_per_sample_gb,
        do_sample=bool(args.do_sample),
        use_kv_cache=True,
        **sampling_kwargs,
    )
    logging.info(
        "saved generated audio to %s sample_rate=%s frames=%s",
        result["audio_path"],
        result["sample_rate"],
        int(result["audio_token_ids"].shape[0]),
    )
    return result


if __name__ == "__main__":
    main()
