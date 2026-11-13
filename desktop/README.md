# NanoVoice 便携桌面版

## 使用

从 [GitHub Releases](https://github.com/12-test-12/MOSS-TTS-Nano/releases) 下载对应系统的桌面包，完整解压到可写目录，再双击启动程序。

| 系统 | 下载文件 | 启动程序 |
| --- | --- | --- |
| Windows x86-64 | `NanoVoice-Windows-x86_64.zip` | `NanoVoice.exe` |
| Linux x86-64 | `NanoVoice-Linux-x86_64.tar.gz` | `NanoVoice` |

选择内置音色，或导入 WAV、MP3、FLAC、OGG、Opus 参考音频。输入文字后点击「生成语音」。应用可以直接播放、另存 WAV 或打开输出文件夹。导入的音色和生成结果分别在同级 `data/voices` 和 `data/outputs`。模型放在 `data/models`，启动和生成日志放在 `data/logs`。

朗读会自动按句子和空行处理停顿。遇到「不是……而是……」「虽然……但是……」这类明确的对比或转折，且能在音频里定位分句边界时，会轻微突出后半句。

## 构建和发布

Windows 和 Linux 都使用 Python 3.12 构建。Linux 发布包以 Ubuntu 22.04 x86-64 为构建基线。安装 [CI 所用的系统库](../.github/workflows/desktop.yml)后，在目标系统的独立构建环境中从仓库根目录运行以下命令。构建阶段需要访问 PyPI 和 Hugging Face。两套模型都使用固定提交。

```text
python -m pip install --only-binary=:all: -r desktop/requirements.txt
python -m desktop.build
```

构建产物位于 `dist/`。已有本地 ONNX 模型时，可用 `--models-dir` 指向包含 `MOSS-TTS-Nano-100M-ONNX` 和 `MOSS-Audio-Tokenizer-Nano-ONNX` 两个子目录的目录。

[GitHub Actions 流水线](../.github/workflows/desktop.yml)会在 `main` 上的相关变更、`v*` 或 `desktop-v*` 标签推送后运行。双端构建和 Linux 兼容性检查通过后，流水线自动公开发布两个便携包及 `SHA256SUMS.txt`。`main` 运行发布为 `desktop-<运行 ID>`，标签运行沿用该标签。手动运行默认发布，可取消勾选 `publish_release` 仅构建和验证。PR 运行可从 Actions 下载测试包。

Windows 与 Linux 共用按 [模型版本](models.json) 锁定的 Actions 缓存。命中后直接使用缓存模型。两个平台以较快的压缩级别生成安装包，并直接传输成品归档。

## 平台验收

流水线解压正式产物到含中文和空格的目录，在窗口中分别触发预设音色合成与参考音频克隆，并验证 WAV。Linux 还用同一个包在禁用网络的 Ubuntu 24.04、26.04、26.10、Debian 13、Fedora 45、Arch Linux 容器运行图形冒烟测试。Ubuntu 26.10 使用 Wayland，其他容器使用 X11。

正式发布前，在目标系统的干净桌面会话中使用默认文件管理器验证图形解压、双击启动、音频播放、保存和移动目录。Linux Mint 和 Windows 10/11 的实际桌面体验纳入手工验收。
