# Android ONNX Runtime Example

This example is a small Android project that runs the MOSS-TTS-Nano ONNX Runtime path on device and writes a WAV file.

It intentionally stays minimal:

- no product UI
- no server calls
- no model files committed to git
- no app-specific business logic

The demo includes a small pure Kotlin tokenizer for `tokenizer.model`, so you can synthesize custom text directly on Android without adding a SentencePiece JNI dependency.

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

Type custom text and tap `Generate custom text WAV`, or tap either pre-tokenized demo button. The app writes a WAV file to its cache directory and prints the output path on screen.

The sample uses:

- `moss_tts_prefill.onnx`
- `moss_tts_decode_step.onnx`
- `moss_tts_local_fixed_sampled_frame.onnx`
- `moss_audio_tokenizer_decode_full.onnx`

## Custom Text

Custom text is handled by `SimpleSentencePieceTokenizer`, which reads the exported `tokenizer.model` and returns the text token ids used by `MossOnnxDemoEngine.synthesize`.

You can also call the engine directly:

```kotlin
MossOnnxDemoEngine(
    modelRoot = modelRoot,
    outputDir = cacheDir,
).use { engine ->
    engine.synthesizeText(
        text = "Hello world!",
        outputFile = File(cacheDir, "custom.wav"),
    )
}
```

The tokenizer intentionally implements only the inference-time pieces needed by the exported Nano `tokenizer.model`: NFKC normalization, whitespace escaping, Unigram segmentation, and BPE merge ranking. If you replace the tokenizer model with a different SentencePiece configuration, compare its output against the Python tokenizer first.

## Notes

- Start with `cpuThreads = 2` or `cpuThreads = 4`; device thermal behavior varies.
- The demo caps generation to `maxFrames = 160` for faster smoke testing.
- The decoded ONNX codec output is stereo; this example averages channels and writes a mono WAV for simplicity.
- Keep the model files outside the APK for local testing. Bundling them into app assets is possible but increases APK size substantially.
- Unit tests use a handcrafted tokenizer fixture by default. To compare against a real Nano tokenizer locally, set `MOSS_TOKENIZER_MODEL=/path/to/tokenizer.model` before running `:app:testDebugUnitTest`.
