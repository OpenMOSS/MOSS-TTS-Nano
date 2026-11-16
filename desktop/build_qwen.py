from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from desktop.build import DIST, ROOT, WORK, archive_package, copy_notices
from desktop.qwen_runtime import MODEL_FILES, MODEL_REVISIONS, download_model, model_name, model_ready


APP_NAME = "NanoVoice-Qwen"
QWEN_WORK = WORK / "qwen"


def verify(archive: Path, edition: str, model: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="验证 Qwen ", dir=QWEN_WORK) as staging:
        shutil.unpack_archive(str(archive), staging)
        package = Path(staging) / f"{APP_NAME}-{edition}"
        data = package / "data"
        model_target = data / "models" / model_name(edition)
        for name in MODEL_FILES:
            target = model_target / name
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(model / name, target)
            except OSError:
                shutil.copy2(model / name, target)
        env = os.environ | {"HF_HUB_OFFLINE": "1", "QT_QPA_PLATFORM": "offscreen"}
        try:
            subprocess.run([str(package / f"{APP_NAME}.exe"), "--self-test"], env=env, check=True, timeout=600)
            if not (data / "self-test.ok").is_file():
                raise RuntimeError(f"The {edition} executable did not finish its synthesis checks.")
        except Exception:
            log = data / "logs" / "desktop.log"
            if log.is_file():
                print(log.read_text(encoding="utf-8", errors="replace"), file=sys.stderr)
            raise


def build(models: dict[str, Path]) -> list[Path]:
    if sys.platform != "win32" or sys.maxsize <= 2**32:
        raise SystemExit("Build on 64-bit Windows.")
    QWEN_WORK.mkdir(parents=True, exist_ok=True)
    DIST.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stage-", dir=QWEN_WORK) as staging:
        stage = Path(staging)
        command = [
            sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir", "--noupx", "--windowed",
            "--name", APP_NAME, "--paths", str(ROOT), "--distpath", str(stage),
            "--workpath", str(QWEN_WORK / "pyinstaller"), "--specpath", str(QWEN_WORK),
            "--add-data", f"{ROOT / 'assets' / 'audio'}:assets/audio",
            "--collect-data", "qwen_tts", "--collect-all", "librosa",
            "--exclude-module", "gradio", "--exclude-module", "gradio_client", "--exclude-module", "torchvision",
            "--recursive-copy-metadata", "qwen-tts", str(ROOT / "desktop" / "qwen_app.py"),
        ]
        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        env = os.environ | {
            "PYINSTALLER_CONFIG_DIR": str(QWEN_WORK / "pyinstaller-cache"),
            "PATH": os.pathsep.join(map(str, (Path(sys.executable).parent, sys.base_prefix, system_root / "System32", system_root))),
        }
        subprocess.run(command, cwd=ROOT, env=env, check=True)

        core = stage / APP_NAME
        copy_notices(core)
        shutil.copy2(ROOT / "desktop" / "START-QWEN.txt", core / "START-HERE.txt")
        for filename in ("icuuc.dll", "icuin.dll"):
            (core / "_internal" / filename).unlink(missing_ok=True)
            (core / filename).unlink(missing_ok=True)
        archives = []
        for edition, model in models.items():
            package = stage / f"{APP_NAME}-{edition}"
            shutil.copytree(core, package, copy_function=os.link)
            (package / "_internal" / "qwen-edition.txt").write_text(edition, encoding="utf-8")
            archive = archive_package(package, windows=True)
            if archive.stat().st_size >= 2 * 1024**3:
                raise ValueError(f"{archive.name} exceeds GitHub's release asset size limit.")
            verify(archive, edition, model)
            archives.append(archive.replace(DIST / archive.name))
    return archives


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and verify both Windows Qwen3-TTS editions.")
    parser.add_argument("--model-06", type=Path, help="Use a local 0.6B Base model for verification.")
    parser.add_argument("--model-17", type=Path, help="Use a local 1.7B Base model for verification.")
    args = parser.parse_args()
    models = {}
    for edition, override in zip(MODEL_REVISIONS, (args.model_06, args.model_17)):
        model = override or QWEN_WORK / "models" / model_name(edition)
        if override is None and not model_ready(model):
            download_model(edition, model)
        if not model_ready(model):
            parser.error(f"Incomplete {edition} model at {model}")
        models[edition] = model.resolve()
    checksums = []
    for archive in build(models):
        with archive.open("rb") as file:
            checksums.append(f"{hashlib.file_digest(file, 'sha256').hexdigest()}  {archive.name}\n")
        print(archive)
    (DIST / "SHA256SUMS-Qwen.txt").write_text("".join(checksums), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
