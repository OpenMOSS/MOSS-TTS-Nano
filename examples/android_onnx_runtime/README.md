# Android ONNX Runtime Example

This example is a small Android project that runs the MOSS-TTS-Nano ONNX Runtime path on device and writes a WAV file.

It intentionally stays minimal:

- no product UI
- no server calls
- no model files committed to git
- no app-specific business logic

The demo synthesizes two pre-tokenized prompts so the ONNX path can be tested without adding a large SentencePiece JNI dependency to the first Android example.

## Model Files

Download the ONNX assets:

```bash
huggingface-cli download OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX \
  --local-dir MOSS-TTS-Nano-100M-ONNX

huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX \
  --local-dir MOSS-Audio-Tokenizer-Nano-ONNX
```

Copy both directories to the app external files directory:

```text
Android/data/com.openmoss.ttsnano.onnxruntime/files/moss_tts_onnx/
  MOSS-TTS-Nano-100M-ONNX/
    browser_poc_manifest.json
    tts_browser_onnx_meta.json
    tokenizer.model
    moss_tts_prefill.onnx
    moss_tts_decode_step.onnx
    moss_tts_local_fixed_sampled_frame.onnx
    *.data
  MOSS-Audio-Tokenizer-Nano-ONNX/
    codec_browser_onnx_meta.json
    moss_audio_tokenizer_decode_full.onnx
    *.data
```

You can use Android Studio Device Explorer or `adb push` after launching the app once:

```bash
adb shell mkdir -p \
  /sdcard/Android/data/com.openmoss.ttsnano.onnxruntime/files/moss_tts_onnx/

adb push MOSS-TTS-Nano-100M-ONNX \
  /sdcard/Android/data/com.openmoss.ttsnano.onnxruntime/files/moss_tts_onnx/

adb push MOSS-Audio-Tokenizer-Nano-ONNX \
  /sdcard/Android/data/com.openmoss.ttsnano.onnxruntime/files/moss_tts_onnx/
```

## Run

Open `examples/android_onnx_runtime` in Android Studio, connect a device, and run the `app` configuration.

Tap either demo button. The app writes a WAV file to its cache directory and prints the output path on screen.

The sample uses:

- `moss_tts_prefill.onnx`
- `moss_tts_decode_step.onnx`
- `moss_tts_local_fixed_sampled_frame.onnx`
- `moss_audio_tokenizer_decode_full.onnx`

## Custom Text

For custom text input, tokenize with `tokenizer.model` using the same SentencePiece model used by the Python ONNX runtime, then pass the resulting token ids into `MossOnnxDemoEngine.synthesize`.

For a production Android app, add one of the following tokenizer paths:

- a small SentencePiece JNI wrapper
- a pre-tokenization service or build step
- another Android-compatible SentencePiece implementation

The ONNX Runtime code is independent from the tokenizer as long as it receives the correct `IntArray` token ids.

## Notes

- Start with `cpuThreads = 2` or `cpuThreads = 4`; device thermal behavior varies.
- The demo caps generation to `maxFrames = 160` for faster smoke testing.
- The decoded ONNX codec output is stereo; this example averages channels and writes a mono WAV for simplicity.
- Keep the model files outside the APK for local testing. Bundling them into app assets is possible but increases APK size substantially.
