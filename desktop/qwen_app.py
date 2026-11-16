import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    executable = Path(sys.executable).resolve()
    if getattr(sys, "frozen", False):
        from ctypes import windll

        windll.kernel32.SetDllDirectoryW(None)
        python = executable.parent / "runtime" / "pythonw.exe"
        return subprocess.run([str(python), "-B", "-m", "desktop.qwen_app", *sys.argv[1:]], check=False).returncode

    from desktop.app import main as run_desktop
    from desktop.qwen_runtime import MODEL_REVISIONS

    # Libraries that report progress expect writable streams, even in a windowed executable.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    edition = (executable.parent / "qwen-edition.txt").read_text(encoding="utf-8").strip()
    if edition not in MODEL_REVISIONS:
        raise ValueError(f"Unknown Qwen edition: {edition}")
    sys.argv[1:1] = ["--data-dir", str(executable.parent.parent / "data")]
    return run_desktop(qwen_edition=edition)


if __name__ == "__main__":
    sys.exit(main())
