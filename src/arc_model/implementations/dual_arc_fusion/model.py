"""Independent original network plus36D dynamic score-correction branch."""

import torch
from .network import Model as Original
from .enhancement import Correction


class Model(Original):
    def attach_correction(self, state):
        self.correction = Correction(
            state, baseline_id="B_FUSION_CURRENT_fixed20_v2", score_kind="logit_raw"
        )

    def forward(self, x, spectral=None, *, dynamic=None, enabled=False):
        logits = super().forward(x, spectral)
        if not enabled:
            return logits
        if self.training:
            raise ValueError(
                "Fit correction from grouped OOF via training.fit_from_oof; backbone remains its own fixed model"
            )
        if not hasattr(self, "correction"):
            raise ValueError(
                "No fitted correction on this locked parent; no legacy-score inheritance"
            )
        delta = self.correction.delta(dynamic)
        result = logits.clone()
        result[:, 1] += torch.as_tensor(delta, dtype=logits.dtype, device=logits.device)
        return result

    def save_correction(self, path):
        import numpy as np

        if not hasattr(self, "correction"):
            raise ValueError("No fitted correction")
        state = dict(self.correction.state)
        np.savez(path, **state)
