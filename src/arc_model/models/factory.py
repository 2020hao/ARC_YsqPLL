"""Explicit neural-model registry. HYDRA has a separate feature/classifier API."""

from .standalone.arc_fenet import FENet
from .standalone.arc_fusion import FusionNet


def make_model(name: str, extra: int = 0):
    registry = {"ArcFENet": FENet, "ArcFusionNet": FusionNet}
    if name not in registry:
        raise ValueError(f"{name!r} is not a neural model; HYDRA uses models.hydra")
    if not isinstance(extra, int) or isinstance(extra, bool) or not 0 <= extra <= 42:
        raise ValueError("Extra features must be an explicit count between 0 and 42")
    return registry[name](extra)
