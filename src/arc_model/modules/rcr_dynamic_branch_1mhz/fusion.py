"""Versioned phase_dynamic_v2 fusion; raw SVM margins stay in margin space."""

import numpy as np
from scipy.special import expit

from .fitting import replay


def corrected_margin(state, raw, features):
    raw = np.asarray(raw)
    if str(state["fusion"]) == "whole_path" and float(state["strength"]) == 0:
        return raw.copy()
    delta, distance = replay(state, features)
    threshold = float(state["gate_threshold"])
    gate = expit(-(distance - threshold) / (0.15 * threshold + 1e-8))
    calibrated = float(state["calibration_slope"]) * raw + float(state["calibration_intercept"])
    strength = float(state["strength"])
    if str(state["fusion"]) == "old_residual":
        return calibrated + strength * gate * delta
    if str(state["fusion"]) == "whole_path":
        return raw + strength * gate * (calibrated - raw + delta)
    raise ValueError("Unknown fusion; select old_residual or whole_path explicitly")
