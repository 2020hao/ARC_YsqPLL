"""Local YAML reader; this implementation imports no peer model package."""

from pathlib import Path
import yaml

CONFIG_KEY = "ArcFENet_DualBranch"


def load_config(path=None):
    p = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[4] / "configs/models.yaml"
    )
    with p.open() as f:
        document = yaml.safe_load(f)
    if not isinstance(document, dict) or CONFIG_KEY not in document:
        raise ValueError("Missing model config: " + CONFIG_KEY)
    cfg = document[CONFIG_KEY]
    if cfg["implementation"] != "dual_arc_fenet":
        raise ValueError("Config implementation mismatch")
    return cfg
