# MOSS-TTS Nano 便携桌面版

## 使用

从当前仓库的 Releases 下载对应系统的桌面包，完整解压到可写目录，再双击启动程序。

| 系统 | 下载文件 | 启动程序 |
| --- | --- | --- |
| Windows x86-64 | `MOSS-TTS-Nano-Windows-x86_64.zip` | `MOSS-TTS.exe` |
| Linux x86-64 | `MOSS-TTS-Nano-Linux-x86_64.tar.gz` | `MOSS-TTS` |

选择内置音色，或导入 WAV、MP3、FLAC、OGG、Opus 参考音频。输入文字后点击「生成语音」。应用可以直接播放、另存 WAV 或打开输出文件夹。导入的音色和生成结果分别在同级 `data/voices` 和 `data/outputs`。模型放在 `data/models`，启动和生成日志放在 `data/logs`。

## 构建和发布

Windows 和 Linux 都使用 Python 3.12 构建。Linux 发布包以 Ubuntu 22.04 x86-64 为构建基线。安装 [CI 所用的系统库](../.github/workflows/desktop.yml)后，在目标系统的独立构建环境中从仓库根目录运行以下命令。构建阶段需要访问 PyPI 和 Hugging Face。两套模型都使用固定提交。

```text
python -m pip install --only-binary=:all: -r desktop/requirements.txt
python -m desktop.build
```

构建产物位于 `dist/`。已有本地 ONNX 模型时，可用 `--models-dir` 指向包含 `MOSS-TTS-Nano-100M-ONNX` 和 `MOSS-Audio-Tokenizer-Nano-ONNX` 两个子目录的目录。

[GitHub Actions 流水线](../.github/workflows/desktop.yml)可手动运行，也会在相关变更和 `desktop-v*` 标签上触发。普通运行把压缩包留在 Actions Artifacts。标签运行在双端构建和 Linux 兼容性检查通过后，把两个包和 SHA-256 校验文件附到 Release 草稿。

## 平台验收

流水线解压正式产物到含中文和空格的目录，在窗口中分别触发预设音色合成与参考音频克隆，并验证 WAV。Linux 还用同一个包在禁用网络的 Ubuntu 24.04、26.04、26.10、Debian 13、Fedora 45、Arch Linux 容器运行图形冒烟测试。Ubuntu 26.10 使用 Wayland，其他容器使用 X11。

正式发布前，在目标系统的干净桌面会话中使用默认文件管理器验证图形解压、双击启动、音频播放、保存和移动目录。Linux Mint 和 Windows 10/11 的实际桌面体验纳入手工验收。
