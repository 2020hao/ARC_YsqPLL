"""Array contracts for phase features; no datasets, file paths, labels or models."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

BANDS_HZ = ((10_000, 100_000), (100_000, 400_000))


@dataclass(frozen=True)
class PhaseFeatureBatch:
    values: np.ndarray
    names: tuple[str, ...]
    valid: np.ndarray
    phase_valid: np.ndarray
    mode: str

    def __post_init__(self):
        if self.values.ndim != 2 or self.values.shape[1] != len(self.names):
            raise ValueError("Feature values and names must align")
        if self.valid.shape != self.values.shape:
            raise ValueError("Per-column validity must align with values")
        if self.phase_valid.shape != (len(self.values),):
            raise ValueError("Phase validity must have one value per window")
        if not np.isfinite(self.values).all():
            raise ValueError("Features must be finite, with explicit invalid masks")
