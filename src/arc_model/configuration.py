"""Validated project paths and per-model experiment configuration."""

from dataclasses import dataclass
from pathlib import Path
import yaml

MODELS = ("ArcFENet", "ArcFusionNet", "HYDRA")
MODES = ("base", "raw", "enhanced", "enhanced_low")


@dataclass
class ExperimentConfig:
    root: Path
    paths: dict
    model: str
    variant: int
    mode: str
    seed: int
    threads: int
    device: str
    training: dict

    def path(self, name: str) -> Path:
        value = Path(self.paths[name]).expanduser()
        return value if value.is_absolute() else self.root / value

    def validate(self):
        if self.model not in MODELS or self.mode not in MODES:
            raise ValueError(f"Unknown model/mode: {self.model}/{self.mode}")
        if self.variant not in (0, 1):
            raise ValueError("variant must be 0 (fixed) or 1 (PLL)")
        if self.mode == "enhanced_low" and self.variant != 1:
            raise ValueError("enhanced_low is a PLL-only control in this experiment")
        if self.seed not in (0, 1, 2) or self.threads < 1:
            raise ValueError("Use prepared seeds 0,1,2 and a positive thread count")
        if self.model != "HYDRA":
            if self.training.get("epochs", 0) < 1 or self.training.get("batch_size", 0) < 1:
                raise ValueError("Positive epochs and batch_size are required")
        return self


def load_config(model_path: str | Path, pipeline_path: str | Path | None = None):
    model_path = Path(model_path).resolve()
    pipeline = (
        Path(pipeline_path).resolve()
        if pipeline_path
        else model_path.parent.parent / "pipeline.yaml"
    )
    common = yaml.safe_load(pipeline.read_text())
    if common.get("scope") != "real_appliances_only":
        raise ValueError("This project accepts real-appliance evaluation only")
    model = yaml.safe_load(model_path.read_text())
    return ExperimentConfig(
        root=pipeline.parent.parent,
        paths=common["paths"],
        model=model["model"],
        variant=common["variant"],
        mode=common["mode"],
        seed=common["seed"],
        threads=common["threads"],
        device=common["device"],
        training=model["training"],
    ).validate()
