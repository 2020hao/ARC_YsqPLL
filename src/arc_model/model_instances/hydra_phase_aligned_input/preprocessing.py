"""Local author-FD extraction; learned scaling belongs to this model's fit."""
from .features import paper_transform
from .configuration import load_config

def extract_features(waves, *, device="cpu"):
    cfg = load_config()["network"]["fixed_kwargs"]
    return paper_transform(waves, device=device, k=cfg["k"], g=cfg["g"],
                           kernel_length=cfg["kernel_length"], stride=cfg["stride"],
                           seed=cfg["transform_seed"])
