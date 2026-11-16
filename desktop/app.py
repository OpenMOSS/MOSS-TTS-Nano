from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile
import soxr
from PySide6.QtCore import QLocale, QObject, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from desktop.qwen_runtime import (
    QwenRuntime,
    SAMPLE_AUDIO,
    SAMPLE_RATE as QWEN_SAMPLE_RATE,
    download_model,
    model_name,
    model_ready,
)
from onnx_tts_runtime import OnnxTtsRuntime


AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".opus"}


def label(zh: str, en: str) -> str:
    return zh if QLocale.system().language() == QLocale.Language.Chinese else en


class DesktopRuntime(OnnxTtsRuntime):
    def synthesize_to_file(self, text: str, voice: str, reference: str, output: Path) -> np.ndarray:
        result = self.synthesize(
            text=text,
            voice=voice or None,
            prompt_audio_path=reference or None,
            output_audio_path=output,
            enable_wetext=False,
            automatic_prosody=True,
        )
        return result["waveform"]

    def _load_reference_audio(self, reference_audio_path: str | Path) -> np.ndarray:
        samples, rate = soundfile.read(str(reference_audio_path), dtype="float32", always_2d=True)
        if not len(samples):
            raise ValueError(label("参考音频为空。", "The reference audio is empty."))
        codec = self.codec_meta["codec_config"]
        if rate != codec["sample_rate"]:
            samples = soxr.resample(samples, rate, codec["sample_rate"])
        channels = codec["channels"]
        if channels == 1 and samples.shape[1] > 1:
            samples = samples.mean(axis=1, keepdims=True)
        elif channels > 1 and samples.shape[1] == 1:
            samples = np.repeat(samples, channels, axis=1)
        if samples.shape[1] != channels:
            raise ValueError(label("请使用单声道或立体声参考音频。", "Use a mono or stereo recording."))
        return np.ascontiguousarray(samples.T[None, :, :], dtype=np.float32)


def load_runtime(data_dir: Path) -> DesktopRuntime:
    return DesktopRuntime(
        model_dir=data_dir / "models",
        output_dir=data_dir / "outputs",
        thread_count=min(4, os.cpu_count() or 1),
        max_new_frames=375,
    )


def generate_audio(runtime: DesktopRuntime | QwenRuntime, text: str, voice: str, reference: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = runtime.output_dir / f"NanoVoice_{stamp}_{uuid4().hex[:8]}.wav"
    temporary = output.with_suffix(".part")
    try:
        if not runtime.synthesize_to_file(text, voice, reference, temporary).size:
            raise ValueError(label("没有生成音频。请换一段文字再试。", "No audio was generated. Try different text."))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


class EngineWorker(QObject):
    loaded = Signal(list)
    generated = Signal(str)
    failed = Signal(str)
    status_changed = Signal(str)

    def __init__(self, data_dir: Path, qwen_model: Path | None, download_edition: str | None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.qwen_model = qwen_model
        self.download_edition = download_edition
        self.runtime: DesktopRuntime | QwenRuntime | None = None

    @Slot()
    def load(self) -> None:
        try:
            if self.qwen_model is None:
                self.runtime = load_runtime(self.data_dir)
            else:
                if self.download_edition and not model_ready(self.qwen_model):
                    self.status_changed.emit(label(
                        f"首次启动正在下载 Qwen3-TTS {self.download_edition} 模型…",
                        f"Downloading Qwen3-TTS {self.download_edition} on first launch…",
                    ))
                    download_model(self.download_edition, self.qwen_model)
                    self.status_changed.emit(label("正在加载语音模型…", "Loading voice models…"))
                self.runtime = QwenRuntime(self.qwen_model, self.data_dir / "outputs")
            self.loaded.emit([row["voice"] for row in self.runtime.list_builtin_voices()])
        except Exception as exc:
            logging.exception("Could not load the speech model")
            self.failed.emit(str(exc))

    @Slot(str, str, str)
    def generate(self, text: str, voice: str, reference: str) -> None:
        try:
            if self.runtime is None:
                raise RuntimeError("The voice models have not been loaded.")
            self.generated.emit(str(generate_audio(self.runtime, text, voice, reference)))
        except Exception as exc:
            logging.exception("Speech generation failed")
            self.failed.emit(str(exc))


class DesktopWindow(QWidget):
    def __init__(self, data_dir: Path, *, qwen_model: Path | None = None,
                 download_edition: str | None = None, testing: bool = False) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.qwen_model = qwen_model
        self.ready = False
        self.busy = True
        self.testing = testing
        self.output: Path | None = None
        self.player = QSoundEffect(self)

        title = "NanoVoice · Qwen3-TTS" if qwen_model else "NanoVoice"
        self.setWindowTitle(f"{title} {download_edition}" if download_edition else title)
        self.resize(700, 475)
        self.setMinimumSize(560, 355)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        text_label = QLabel(label("合成文字(&T)", "&Text to speak"))
        layout.addWidget(text_label)
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText(label("输入或粘贴要朗读的文字", "Enter or paste text to speak"))
        text_label.setBuddy(self.text)
        layout.addWidget(self.text, 1)

        self.voice = QComboBox()
        self.voice.setAccessibleName(label("音色", "Voice"))
        self.import_button = QPushButton(label("导入参考音频…", "Import reference audio…"))
        self.import_button.clicked.connect(self._import_reference)
        voice_row = QHBoxLayout()
        voice_row.addWidget(self.voice, 1)
        voice_row.addWidget(self.import_button)
        voice_form = QFormLayout()
        voice_form.addRow(label("音色", "Voice"), voice_row)
        layout.addLayout(voice_form)

        actions = QHBoxLayout()
        self.generate_button = QPushButton(label("生成语音", "Generate speech"))
        self.generate_button.clicked.connect(self._generate)
        self.play_button = QPushButton()
        self.play_button.clicked.connect(self._toggle_playback)
        self.player.statusChanged.connect(self._update_play_button)
        self.player.playingChanged.connect(self._update_play_button)
        self.save_button = QPushButton(label("另存 WAV…", "Save WAV as…"))
        self.save_button.clicked.connect(self._save_audio)
        folder_button = QPushButton(label("打开输出文件夹", "Open output folder"))
        folder_button.clicked.connect(self._open_output_folder)
        for button in (self.generate_button, self.play_button, self.save_button, folder_button):
            actions.addWidget(button)
        actions.addStretch()
        layout.addLayout(actions)

        self.status = QLabel(label("正在加载语音模型…", "Loading voice models…"))
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.text.textChanged.connect(self._update_controls)
        self._update_controls()
        self.text.setFocus()

        # PyTorch tensor operations can stall on a QThread on Windows.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="NanoVoice")
        self.worker = EngineWorker(data_dir, qwen_model, download_edition)
        self.worker.status_changed.connect(self.status.setText, Qt.ConnectionType.QueuedConnection)
        self.worker.loaded.connect(self._loaded, Qt.ConnectionType.QueuedConnection)
        self.worker.generated.connect(self._generated, Qt.ConnectionType.QueuedConnection)
        self.worker.failed.connect(self._failed, Qt.ConnectionType.QueuedConnection)

    def _update_controls(self) -> None:
        self.generate_button.setEnabled(self.ready and not self.busy and bool(self.text.toPlainText().strip()))
        self.voice.setEnabled(self.ready and not self.busy)
        self.import_button.setEnabled(self.ready and not self.busy)
        self.save_button.setEnabled(self.output is not None)
        self._update_play_button()

    @Slot()
    def _update_play_button(self) -> None:
        self.play_button.setEnabled(self.player.status() == QSoundEffect.Status.Ready)
        self.play_button.setText(
            label("停止播放", "Stop") if self.player.isPlaying() else label("播放", "Play")
        )

    @Slot(list)
    def _loaded(self, voices: list[str]) -> None:
        for voice in voices:
            self.voice.addItem(voice, (voice, ""))
        voices_dir = self.data_dir / "voices"
        if voices_dir.is_dir():
            for path in sorted(voices_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                    self._add_reference(path)
        self.ready = True
        self.busy = False
        self.status.setText(label("输入文字后即可生成。", "Enter text to generate speech."))
        self._update_controls()

    def _add_reference(self, path: Path) -> None:
        self.voice.addItem(label(f"参考音频 · {path.stem}", f"Reference · {path.stem}"), ("", str(path)))

    @Slot()
    def _import_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            label("选择参考音频", "Choose reference audio"),
            filter=label("音频", "Audio") + " (*.wav *.mp3 *.flac *.ogg *.opus)",
        )
        if not path:
            return
        source = Path(path)
        if source.suffix.lower() not in AUDIO_SUFFIXES:
            self._show_error(label("请选择 WAV、MP3、FLAC、OGG 或 Opus。", "Choose WAV, MP3, FLAC, OGG or Opus."))
            return
        voices_dir = self.data_dir / "voices"
        voices_dir.mkdir(parents=True, exist_ok=True)
        destination = voices_dir / source.name
        if destination.exists():
            destination = destination.with_name(f"{source.stem}_{uuid4().hex[:8]}{source.suffix}")
        temporary = destination.with_name(f".{uuid4().hex}.part")
        try:
            info = soundfile.info(str(source))
            if not info.frames or info.channels not in (1, 2):
                raise ValueError(label("请选择有内容的单声道或立体声音频。", "Choose a non-empty mono or stereo recording."))
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        except (OSError, ValueError, soundfile.SoundFileError) as exc:
            self._show_error(str(exc))
            return
        finally:
            temporary.unlink(missing_ok=True)
        self._add_reference(destination)
        self.voice.setCurrentIndex(self.voice.count() - 1)

    @Slot()
    def _generate(self) -> None:
        text = self.text.toPlainText().strip()
        if self.busy or not text:
            return
        voice, reference = self.voice.currentData() or ("", "")
        self.player.stop()
        self.busy = True
        self.status.setText(label("正在生成语音…", "Generating speech…"))
        self._update_controls()
        self.executor.submit(self.worker.generate, text, voice, reference)

    @Slot(str)
    def _generated(self, path: str) -> None:
        self.output = Path(path)
        self.player.setSource(QUrl.fromLocalFile(path))
        self.busy = False
        self.status.setText(label(f"已保存 · {self.output.name}", f"Saved · {self.output.name}"))
        self._update_controls()

    @Slot(str)
    def _failed(self, error: str) -> None:
        self.busy = False
        if self.ready:
            self.status.setText(label("生成失败。", "Speech generation failed."))
        elif self.qwen_model:
            self.status.setText(label(
                "Qwen3-TTS 加载失败。请检查模型目录和运行环境。",
                "Qwen3-TTS could not load. Check the model folder and Python environment.",
            ))
        else:
            self.status.setText(label("模型加载失败。请重新解压完整的软件包。", "Model loading failed. Extract the complete package again."))
        self._update_controls()
        self._show_error(error)

    def _show_error(self, error: str) -> None:
        if self.testing:
            return
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("NanoVoice")
        dialog.setTextFormat(Qt.TextFormat.PlainText)
        dialog.setText(error)
        dialog.exec()

    @Slot()
    def _toggle_playback(self) -> None:
        if self.player.isPlaying():
            self.player.stop()
        else:
            self.player.play()

    @Slot()
    def _open_output_folder(self) -> None:
        folder = self.data_dir / "outputs"
        folder.mkdir(parents=True, exist_ok=True)
        if sys.platform == "linux" and getattr(sys, "frozen", False):
            # Launch the desktop file manager with the host's shared libraries.
            env = os.environ.copy()
            env.pop("LD_LIBRARY_PATH", None)
            original = env.pop("LD_LIBRARY_PATH_ORIG", None)
            if original is not None:
                env["LD_LIBRARY_PATH"] = original
            try:
                subprocess.Popen(["xdg-open", str(folder)], env=env)
            except OSError as exc:
                self._show_error(f"{folder}\n{exc}")
        elif not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            self._show_error(str(folder))

    @Slot()
    def _save_audio(self) -> None:
        if self.output is None:
            return
        dialog = QFileDialog(self, label("另存语音", "Save speech"), str(self.output), "WAV (*.wav)")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setDefaultSuffix("wav")
        if dialog.exec():
            destination = Path(dialog.selectedFiles()[0])
            try:
                if destination.resolve() != self.output.resolve():
                    shutil.copyfile(self.output, destination)
            except OSError as exc:
                self._show_error(str(exc))

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.busy:
            self.status.setText(label("当前任务结束后即可关闭。", "You can close the window when the current task finishes."))
            event.ignore()
            return
        self.executor.shutdown()
        super().closeEvent(event)


def self_test(window: DesktopWindow) -> None:
    reference = window.data_dir / "voices" / "测试 reference.wav"
    if window.qwen_model:
        shutil.copyfile(SAMPLE_AUDIO / "en_2.wav", reference)
    else:
        sample_rate = 16000
        t = np.arange(sample_rate * 2) / sample_rate
        soundfile.write(reference, 0.15 * np.sin(2 * np.pi * 220 * t), sample_rate)

    def wait_for_worker() -> None:
        timeout = 180 if window.qwen_model else 90
        deadline = time.monotonic() + timeout
        while window.busy:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"The synthesis worker did not finish in {timeout} seconds.")
            QApplication.processEvents()
            time.sleep(0.01)

    window.show()
    window.executor.submit(window.worker.load)
    wait_for_worker()
    if not window.ready or window.voice.count() < 2:
        raise RuntimeError("The desktop window could not load preset and reference voices.")
    window.text.setPlainText("你好，欢迎使用语音合成。")
    expected_rate = QWEN_SAMPLE_RATE if window.qwen_model else 48000
    previous = None
    for index in (0, window.voice.count() - 1):
        window.voice.setCurrentIndex(index)
        window.generate_button.click()
        wait_for_worker()
        output = window.output
        if output is None or output == previous:
            raise RuntimeError("The desktop window did not generate a new WAV file.")
        with wave.open(str(output)) as audio:
            if audio.getnframes() <= 0 or audio.getframerate() != expected_rate:
                raise RuntimeError(f"Invalid synthesized WAV: {output}")
        previous = output
    (window.data_dir / "self-test.ok").write_text("Preset and reference synthesis passed.\n", encoding="utf-8")


def main(*, qwen_edition: str | None = None) -> int:
    root = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    parser = argparse.ArgumentParser(description="Launch NanoVoice.")
    parser.add_argument("--data-dir", type=Path, default=root / "data", help="Folder for models, voices, outputs and logs.")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    if qwen_edition or not getattr(sys, "frozen", False):
        parser.add_argument("--qwen-model", type=Path, help="Use a local Qwen3-TTS Base model folder.")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    qwen_model = getattr(args, "qwen_model", None)
    download_edition = qwen_edition if qwen_model is None else None
    if download_edition:
        qwen_model = data_dir / "models" / model_name(download_edition)
    if qwen_model is not None:
        qwen_model = qwen_model.expanduser().resolve()
    app = QApplication([sys.argv[0]])
    app.setApplicationName("NanoVoice")
    testing = args.self_test
    try:
        for folder in ("outputs", "voices", "logs"):
            (data_dir / folder).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=data_dir / "logs" / "desktop.log", encoding="utf-8", level=logging.INFO)
        window = DesktopWindow(data_dir, qwen_model=qwen_model, download_edition=download_edition, testing=testing)
        if testing:
            try:
                self_test(window)
            finally:
                window.executor.shutdown()
            return 0
        window.show()
        window.executor.submit(window.worker.load)
        return app.exec()
    except Exception as exc:
        logging.exception("Could not start NanoVoice")
        if not testing:
            dialog = QMessageBox()
            dialog.setIcon(QMessageBox.Icon.Critical)
            dialog.setWindowTitle("NanoVoice")
            dialog.setText(label("启动失败。请将软件解压到可写入的文件夹。", "Could not start. Extract the package into a writable folder."))
            dialog.setDetailedText(str(exc))
            dialog.exec()
        return 1


if __name__ == "__main__":
    sys.exit(main())
