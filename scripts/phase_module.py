"""Thin current-phase module entry. Install editable package or use this wrapper."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arc_model.experiments.phase_cli import main

if __name__ == "__main__":
    main()
