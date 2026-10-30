"""SVM+dynamic correction candidate. New calibrated head must be fitted."""

import numpy as np
from .network import Model as Original
from .enhancement import Correction


class Model(Original):
    def attach_correction(self, state, *, calibration):
        if calibration is None or not {"slope", "intercept", "fit_indices"}.issubset(calibration):
            raise ValueError(
                "Training-only margin calibration and matched calibration-only control required"
            )
        self.calibration = calibration
        self.correction = Correction(
            state, baseline_id="B_HYDRA_FD_fixed20_v1", score_kind="calibrated_svm_logit"
        )

    def decision_function(self, features, *, dynamic=None, enabled=False):
        margin = super().decision_function(features)
        if not enabled:
            return margin
        if not hasattr(self, "correction"):
            raise ValueError(
                "No fitted correction for the new frozen SVM; legacy C0 weights are incompatible"
            )
        z = float(self.calibration["slope"]) * margin + float(self.calibration["intercept"])
        return z + self.correction.delta(dynamic)

    def predict(self, features, **kwargs):
        return self.decision_function(features, **kwargs) >= 0

    def save_correction(self, path):
        if not hasattr(self, "correction"):
            raise ValueError("No fitted correction")
        state = dict(self.correction.state)
        state.update(
            calibration_slope=np.asarray(self.calibration["slope"]),
            calibration_intercept=np.asarray(self.calibration["intercept"]),
            calibration_fit_indices=np.asarray(self.calibration["fit_indices"]),
        )
        np.savez(path, **state)
