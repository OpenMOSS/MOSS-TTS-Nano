# Third-party components

The MOSS-TTS Nano source and both OpenMOSS model exports use Apache-2.0.

- [MOSS-TTS-Nano-100M-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX)
- [MOSS-Audio-Tokenizer-Nano-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX)

This distribution uses dynamically linked [Qt and PySide6](https://www.qt.io/licensing/open-source-lgpl-obligations). The Qt and PySide6 libraries in `_internal` can be replaced with compatible builds. Their source is available from [Qt releases](https://download.qt.io/archive/qt/) and [Qt for Python releases](https://download.qt.io/official_releases/QtForPython/).

[libsndfile](https://github.com/libsndfile/libsndfile) is distributed with SoundFile. [libsoxr](https://sourceforge.net/projects/soxr/) is used by python-soxr. The package license and attribution files are included under `licenses/`.

The Qwen edition uses the Apache-2.0 [Qwen3-TTS runtime](https://github.com/QwenLM/Qwen3-TTS) and either the [0.6B Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) or [1.7B Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) model. Preset voice samples come from the OpenMOSS source assets.
