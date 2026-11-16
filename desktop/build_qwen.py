from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import site
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

from desktop.build import DIST, ROOT, WORK, archive_package, copy_notices
from desktop.qwen_runtime import MODEL_FILES, MODEL_REVISIONS, download_model, model_name, model_ready


APP_NAME = "NanoVoice-Qwen"
QWEN_WORK = WORK / "qwen"
EMBED_ARCHIVE = "python-3.12.10-embed-amd64.zip"
EMBED_URL = f"https://www.python.org/ftp/python/3.12.10/{EMBED_ARCHIVE}"
EMBED_SHA256 = "4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3"
APP_MODULES = ("onnx_tts_runtime.py", "ort_cpu_runtime.py", "text_normalization_pipeline.py", "tts_robust_normalizer_single_script.py")


def link_or_copy(source: str | Path, target: str | Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def stage_runtime(package: Path) -> None:
    runtime = package / "runtime"
    bundle = QWEN_WORK / EMBED_ARCHIVE
    if not bundle.is_file():
        temporary = bundle.with_suffix(".part")
        urlretrieve(EMBED_URL, temporary)
        temporary.replace(bundle)
    with bundle.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != EMBED_SHA256:
        bundle.unlink(missing_ok=True)
        raise ValueError("The embedded Python archive failed its SHA-256 check. Run the build again to redownload it.")
    shutil.unpack_archive(str(bundle), runtime)
    (runtime / "python312._pth").write_text(
        "python312.zip\n.\napp\nLib/site-packages\nimport site\n", encoding="utf-8"
    )
    shutil.copytree(
        site.getsitepackages()[-1], runtime / "Lib" / "site-packages",
        ignore=shutil.ignore_patterns("__pycache__", "PyInstaller", "_pyinstaller_hooks_contrib"),
        copy_function=link_or_copy,
    )
    app = runtime / "app"
    (app / "desktop").mkdir(parents=True)
    for name in APP_MODULES:
        shutil.copy2(ROOT / name, app / name)
    for name in ("app.py", "qwen_app.py", "qwen_runtime.py"):
        shutil.copy2(ROOT / "desktop" / name, app / "desktop" / name)
    shutil.copytree(ROOT / "moss_tts_nano", app / "moss_tts_nano", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "assets" / "audio", app / "assets" / "audio")


def verify(archive: Path, edition: str, model: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="验证 Qwen ", dir=QWEN_WORK) as staging:
        shutil.unpack_archive(str(archive), staging)
        package = Path(staging) / f"{APP_NAME}-{edition}"
        data = package / "data"
        model_target = data / "models" / model_name(edition)
        for name in MODEL_FILES:
            target = model_target / name
            target.parent.mkdir(parents=True, exist_ok=True)
            link_or_copy(model / name, target)
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
    if sys.platform != "win32" or sys.maxsize <= 2**32 or sys.version_info[:2] != (3, 12):
        raise SystemExit("Build with Python 3.12 on 64-bit Windows.")
    QWEN_WORK.mkdir(parents=True, exist_ok=True)
    DIST.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stage-", dir=QWEN_WORK) as staging:
        stage = Path(staging)
        command = [
            sys.executable, "-m", "PyInstaller", "--noconfirm", "--onefile", "--noupx", "--windowed",
            "--name", APP_NAME, "--paths", str(ROOT), "--distpath", str(stage),
            "--workpath", str(QWEN_WORK / "pyinstaller"), "--specpath", str(QWEN_WORK),
            "--exclude-module", "desktop", str(ROOT / "desktop" / "qwen_app.py"),
        ]
        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        env = os.environ | {
            "PYINSTALLER_CONFIG_DIR": str(QWEN_WORK / "pyinstaller-cache"),
            "PATH": os.pathsep.join(map(str, (Path(sys.executable).parent, sys.base_prefix, system_root / "System32", system_root))),
        }
        subprocess.run(command, cwd=ROOT, env=env, check=True)

        core = stage / APP_NAME
        core.mkdir()
        (stage / f"{APP_NAME}.exe").replace(core / f"{APP_NAME}.exe")
        stage_runtime(core)
        copy_notices(core)
        shutil.copy2(ROOT / "desktop" / "START-QWEN.txt", core / "START-HERE.txt")
        archives = []
        for edition, model in models.items():
            package = stage / f"{APP_NAME}-{edition}"
            shutil.copytree(core, package, copy_function=os.link)
            (package / "runtime" / "qwen-edition.txt").write_text(edition, encoding="utf-8")
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
