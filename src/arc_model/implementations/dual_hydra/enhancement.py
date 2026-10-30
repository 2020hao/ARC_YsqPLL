"""Independent score-correction inference; parent binding is mandatory."""

import numpy as np
from scipy.special import expit
from .dynamic_correction import replay


class Correction:
    def __init__(self, state, *, baseline_id, score_kind):
        self.state = {k: np.asarray(v).copy() for k, v in state.items()}
        if str(self.state.get("baseline_id", "")) != baseline_id:
            raise ValueError("Correction was not fitted on this locked baseline")
        if str(self.state.get("score_kind", "")) != score_kind:
            raise ValueError("Score convention mismatch; old C0 is not a new SVM head")
        for k in [
            "keep",
            "mu",
            "sd",
            "coef",
            "pca_components",
            "pca_mean",
            "alpha",
            "gate_threshold",
        ]:
            if k not in self.state:
                raise ValueError("Missing correction state: " + k)

    def delta(self, dynamic):
        x = np.asarray(dynamic)
        if x.ndim != 2 or x.shape[1] != 36 or not np.isfinite(x).all():
            raise ValueError("Finite [N,36] dynamic features required")
        r, d = replay(self.state, x)
        t = float(self.state["gate_threshold"])
        if t <= 0:
            raise ValueError("Positive training-only gate threshold required")
        return float(self.state["alpha"]) * expit(-(d - t) / (0.15 * t + 1e-8)) * r
