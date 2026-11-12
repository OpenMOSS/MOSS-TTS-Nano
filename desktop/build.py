from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "build" / "desktop"
DIST = ROOT / "dist"
APP_NAME = "MOSS-TTS"
MODEL_REVISIONS = {
    "MOSS-TTS-Nano-100M-ONNX": "f52645cb467506d8e18e746ddd59482685b74e58",
    "MOSS-Audio-Tokenizer-Nano-ONNX": "ceff0d0749bfb3fa2d61149794ec6feef0d1e1ae",
}
MODEL_PATTERNS = ("*.onnx", "*.data", "*.json", "tokenizer.model", "README.md", "LICENSE*")


def fetch_models(destination: Path) -> None:
    from huggingface_hub import snapshot_download

    for name, revision in MODEL_REVISIONS.items():
        snapshot_download(
            repo_id=f"OpenMOSS-Team/{name}",
            revision=revision,
            local_dir=destination / name,
            allow_patterns=MODEL_PATTERNS,
            max_workers=4,
        )


def stage_models(source: Path, destination: Path) -> None:
    for name in MODEL_REVISIONS:
        model = source / name
        files = {path for pattern in MODEL_PATTERNS for path in model.glob(pattern) if path.is_file()}
        if not any(path.suffix == ".onnx" for path in files):
            raise FileNotFoundError(f"ONNX files missing from {model}")
        (destination / name).mkdir(parents=True, exist_ok=True)
        for path in sorted(files):
            if path.stat().st_size < 256 and path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"Download the Git LFS contents for {path}")
            shutil.copy2(path, destination / name / path.name)


def copy_notices(destination: Path) -> None:
    shutil.copy2(ROOT / "LICENSE", destination / "LICENSE")
    shutil.copy2(ROOT / "desktop" / "THIRD-PARTY.md", destination / "THIRD-PARTY.md")
    for package in importlib.metadata.distributions():
        for entry in package.files or ():
            if not entry.parts[0].endswith(".dist-info"):
                continue
            if "licenses" not in entry.parts and not entry.name.upper().startswith(("LICENSE", "COPYING", "NOTICE")):
                continue
            source = package.locate_file(entry)
            if source.is_file():
                target = destination / "licenses" / entry
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)


def build(models: Path, *, fetch: bool) -> Path:
    windows = sys.platform == "win32"
    if sys.platform not in {"win32", "linux"} or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise SystemExit("Build on a 64-bit x86 Windows or Linux host.")

    if fetch:
        fetch_models(models)
    WORK.mkdir(parents=True, exist_ok=True)
    DIST.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stage-", dir=WORK) as staging:
        stage = Path(staging)
        command = [
            sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", "--onedir", "--noupx",
            "--name", APP_NAME, "--paths", str(ROOT), "--distpath", str(stage),
            "--workpath", str(WORK / "pyinstaller"), "--specpath", str(WORK),
            "--exclude-module", "torch", "--exclude-module", "torchaudio",
            "--exclude-module", "transformers", "--exclude-module", "huggingface_hub",
        ]
        env = os.environ | {"PYINSTALLER_CONFIG_DIR": str(WORK / "pyinstaller-cache")}
        if windows:
            command.append("--windowed")
            # Qt imports Windows ICU. A different ICU on the build host's PATH can shadow it in the bundle.
            system_root = Path(os.environ.get("SystemRoot") or os.environ.get("windir") or "C:\\Windows")
            env["PATH"] = os.pathsep.join(map(str, (
                Path(sys.executable).parent, sys.base_prefix, system_root / "System32", system_root,
            )))
        subprocess.run([*command, str(ROOT / "desktop" / "app.py")], cwd=ROOT, env=env, check=True)

        package = stage / APP_NAME
        if windows:
            for system_dll in ("icuuc.dll", "icuin.dll"):
                (package / "_internal" / system_dll).unlink(missing_ok=True)
                (package / system_dll).unlink(missing_ok=True)
        stage_models(models, package / "data" / "models")
        for folder in ("outputs", "voices", "logs"):
            (package / "data" / folder).mkdir(parents=True, exist_ok=True)
        copy_notices(package)
        shutil.copy2(ROOT / "desktop" / "START-HERE.txt", package)

        archive_name = f"MOSS-TTS-Nano-{'Windows' if windows else 'Linux'}-x86_64"
        archive = Path(shutil.make_archive(str(stage / archive_name), "zip" if windows else "gztar", stage, APP_NAME))
        return archive.replace(DIST / archive.name)


def verify(archive: Path) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="验证 Portable ", dir=WORK) as extracted:
        kwargs = {}
        if sys.version_info >= (3, 12) and (archive.suffix == ".gz" or str(archive).endswith(".tar.gz")):
            kwargs["filter"] = "data"
        shutil.unpack_archive(str(archive), extracted, **kwargs)
        package = Path(extracted) / APP_NAME
        binary = package / (f"{APP_NAME}.exe" if sys.platform == "win32" else APP_NAME)
        if sys.platform != "win32":
            binary.chmod(binary.stat().st_mode | 0o755)
        env = os.environ | {"HF_HUB_OFFLINE": "1"}
        try:
            subprocess.run([str(binary), "--self-test"], cwd=extracted, env=env, check=True, timeout=300)
            if not (package / "data" / "self-test.ok").is_file():
                raise RuntimeError("The packaged application did not complete its self-test.")
        except Exception:
            log = package / "data" / "logs" / "desktop.log"
            if log.is_file():
                print(log.read_text(encoding="utf-8", errors="replace"), file=sys.stderr)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify a MOSS-TTS Nano portable desktop package.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--models-dir", type=Path, help="Use pre-downloaded ONNX model folders.")
    mode.add_argument("--verify", type=Path, metavar="ARCHIVE", help="Run the packaged application's smoke test.")
    args = parser.parse_args()
    if args.verify:
        verify(args.verify.resolve())
        print(f"Verified {args.verify}")
        return
    print(build(args.models_dir or (WORK / "models"), fetch=args.models_dir is None))


if __name__ == "__main__":
    main()
