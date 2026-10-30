"""Independent original network plus36D dynamic score-correction branch."""

import torch
from .network import Model as Original
from .enhancement import Correction


class Model(Original):
    def attach_correction(self, state, *, calibration=None):
        """Attach a separately fitted correction; off preserves the original forward."""
        if calibration is None:
            calibration = dict(slope=1.0, intercept=0.0, fit_indices=[])
        if not {"slope", "intercept", "fit_indices"}.issubset(calibration):
            raise ValueError("Training-only affine calibration receipt required")
        if float(calibration["slope"]) <= 0:
            raise ValueError("Calibration must preserve score ordering")
        self.calibration = calibration
        self.correction = Correction(
            state, baseline_id="B_FENET_fixed20_v1", score_kind="logit_raw"
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
        result = logits.float().clone()
        margin = logits[:, 1].float() - logits[:, 0].float()
        calibrated = float(self.calibration["slope"]) * margin + float(self.calibration["intercept"])
        result[:, 1] = result[:, 0] + calibrated + torch.as_tensor(
            delta, dtype=result.dtype, device=result.device
        )
        return result

    def save_correction(self, path):
        import numpy as np

        if not hasattr(self, "correction"):
            raise ValueError("No fitted correction")
        state = dict(self.correction.state)
        state.update(
            calibration_slope=np.asarray(self.calibration["slope"]),
            calibration_intercept=np.asarray(self.calibration["intercept"]),
            calibration_fit_indices=np.asarray(self.calibration["fit_indices"]),
        )
        np.savez(path, **state)
