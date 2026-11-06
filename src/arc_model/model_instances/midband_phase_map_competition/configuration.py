"""Read this instance's complete YAML without another model's defaults."""
from pathlib import Path
import yaml

def load_config(path=None):
    p = Path(path) if path is not None else Path(__file__).with_name("model.yaml")
    cfg = yaml.safe_load(p.read_text())
    if cfg.get("model_id") != 'midband_phase_map_competition':
        raise ValueError("Model configuration identity mismatch")
    return cfg
