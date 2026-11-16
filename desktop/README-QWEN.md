# NanoVoice Qwen3-TTS Windows 版

从 [GitHub Releases](https://github.com/12-test-12/MOSS-TTS-Nano/releases) 选择 Qwen3-TTS 发布，下载 0.6B 或 1.7B 版。完整解压 ZIP 后双击 `NanoVoice-Qwen.exe`。

| 版本 | Windows x64 文件 | 首次模型下载 |
| --- | --- | --- |
| 0.6B Base | `NanoVoice-Qwen-0.6B-Windows-x86_64.zip` | 约 2.5 GB |
| 1.7B Base | `NanoVoice-Qwen-1.7B-Windows-x86_64.zip` | 约 4.5 GB |

两个包均自带 CPU 运行环境，支持 11 个预设音色、参考音频克隆和 WAV 导出。首次启动从 Hugging Face 下载对应模型，下载完成后可离线合成。参考音频保存在 `data/voices`，生成的 WAV 在 `data/outputs`。

如果下载受阻，可以通过 [Qwen 官方下载说明](https://github.com/QwenLM/Qwen3-TTS#released-models-description-and-download)使用 ModelScope 获取同一型号的 Base 模型，并将完整目录放进 `data/models`。保留其中的 `speech_tokenizer` 子目录。重新打开程序即可加载。

## 构建

在 Windows x64 的 Python 3.12 虚拟环境内运行以下命令。

```powershell
python -m pip install torch==2.8.0 torchaudio==2.8.0
python -m pip install -r desktop/requirements-qwen.txt
python -m desktop.build_qwen
```

构建会用真实模型逐一验证解压后的 EXE，再把两个 ZIP 和 SHA-256 校验文件写入 `dist`。已有本地模型时，可以用 `--model-06` 和 `--model-17` 分别指定完整模型目录。
