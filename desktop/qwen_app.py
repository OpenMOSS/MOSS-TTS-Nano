import sys
from pathlib import Path

from desktop.app import main
from desktop.qwen_runtime import MODEL_REVISIONS


if __name__ == "__main__":
    edition = (Path(sys._MEIPASS) / "qwen-edition.txt").read_text(encoding="utf-8").strip()
    if edition not in MODEL_REVISIONS:
        raise ValueError(f"Unknown Qwen edition: {edition}")
    sys.exit(main(qwen_edition=edition))
