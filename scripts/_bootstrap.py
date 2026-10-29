"""Run project commands from any working directory without installing the package."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_dataset(config):
    from arc_model.data.dataset import PreparedDataset

    return PreparedDataset(config.path("prepared"), config.path("manifests"))
