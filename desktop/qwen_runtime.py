from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import soundfile
import soxr

from onnx_tts_runtime import ReadingTextChunker, _join_reading_audio, _PARAGRAPH_BREAK


SAMPLE_AUDIO = Path(__file__).resolve().parents[1] / "assets" / "audio"
SAMPLE_RATE = 24000
MODEL_REVISIONS = {
    "0.6B": "5d83992436eae1d760afd27aff78a71d676296fc",
    "1.7B": "fd4b254389122332181a7c3db7f27e918eec64e3",
}
MODEL_FILES = (
    "config.json", "generation_config.json", "merges.txt", "model.safetensors",
    "preprocessor_config.json", "tokenizer_config.json", "vocab.json",
    "speech_tokenizer/config.json", "speech_tokenizer/model.safetensors",
    "speech_tokenizer/preprocessor_config.json",
)
PRESET_VOICES = {
    "Junhao": "zh_1.wav",
    "Zhiming": "zh_3.wav",
    "Weiguo": "zh_10.wav",
    "Xiaoyu": "zh_11.wav",
    "Yuewen": "zh_4.wav",
    "Lingyu": "zh_6.wav",
    "Ava": "en_2.wav",
    "Bella": "en_3.wav",
    "Adam": "en_4.wav",
    "Nathan": "en_8.wav",
    "Saki": "jp_2.wav",
}


def model_name(edition: str) -> str:
    return f"Qwen3-TTS-12Hz-{edition}-Base"


def model_ready(directory: Path) -> bool:
    return all((directory / file).is_file() for file in MODEL_FILES)


def download_model(edition: str, directory: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=f"Qwen/{model_name(edition)}",
        revision=MODEL_REVISIONS[edition],
        local_dir=directory,
        allow_patterns=MODEL_FILES,
        max_workers=2,
        token=False,
    )


class QwenRuntime(ReadingTextChunker):
    def __init__(self, model_dir: Path, output_dir: Path) -> None:
        if not model_ready(model_dir):
            raise FileNotFoundError(
                f"Qwen3-TTS 模型文件不完整。\nThe Qwen3-TTS model files are incomplete.\n{model_dir}"
            )

        import torch
        from qwen_tts import Qwen3TTSModel

        torch.set_num_threads(min(4, os.cpu_count() or 1))
        device, dtype = "cpu", torch.float32
        if torch.cuda.is_available():
            device = "cuda:0"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logging.info("Loading Qwen3-TTS from %s on %s (%s)", model_dir, device, dtype)
        self.model = Qwen3TTSModel.from_pretrained(
            str(model_dir), device_map=device, dtype=dtype, attn_implementation="sdpa", local_files_only=True
        )
        self.output_dir = output_dir

    def list_builtin_voices(self) -> list[dict[str, str]]:
        return [{"voice": name} for name in PRESET_VOICES]

    def count_text_tokens(self, text: str) -> int:
        return len(self.model.processor.tokenizer.encode(text, add_special_tokens=False))

    def synthesize_to_file(self, text: str, voice: str, reference: str, output: Path) -> np.ndarray:
        paragraphs = [p.strip() for p in _PARAGRAPH_BREAK.split(text) if p.strip()]
        chunks = self.plan_reading_chunks(paragraphs, max_tokens=75)
        if not chunks:
            return np.empty(0, dtype=np.float32)

        source = Path(reference) if reference else SAMPLE_AUDIO / PRESET_VOICES[voice or next(iter(PRESET_VOICES))]
        samples, sample_rate = soundfile.read(str(source), dtype="float32", always_2d=True)
        if not len(samples) or samples.shape[1] not in (1, 2):
            raise ValueError("Choose a non-empty mono or stereo reference recording.")
        mono = samples.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            mono = soxr.resample(mono, sample_rate, SAMPLE_RATE)
        prompt = self.model.create_voice_clone_prompt(
            ref_audio=(mono, SAMPLE_RATE), x_vector_only_mode=True
        )

        waveforms = []
        for chunk in chunks:
            wavs, output_rate = self.model.generate_voice_clone(
                text=chunk.text, language="Auto", voice_clone_prompt=prompt, non_streaming_mode=True
            )
            if not wavs or not wavs[0].size or not np.isfinite(wavs[0]).all():
                raise ValueError("Qwen3-TTS returned invalid audio.")
            waveforms.append(wavs[0])

        waveform = _join_reading_audio(chunks, waveforms, output_rate)
        soundfile.write(str(output), waveform, output_rate, format="WAV", subtype="PCM_16")
        return waveform
