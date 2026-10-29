"""Train-only, mask-aware feature scaling with serializable fit provenance."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from arc_model.methods.phase_representation import PhaseFeatureBatch


@dataclass(frozen=True)
class PhaseFeatureScaler:
    mean: np.ndarray
    scale: np.ndarray
    names: tuple[str, ...]
    fit_indices: np.ndarray

    @classmethod
    def fit(cls, batch: PhaseFeatureBatch, train_indices):
        ids = np.asarray(train_indices)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids):
            raise ValueError("Nonempty integer training indices required")
        if len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(batch.values):
            raise ValueError("Invalid or duplicate training indices")
        x, valid = batch.values[ids], batch.valid[ids]
        count = valid.sum(0)
        mean = np.divide(
            np.where(valid, x, 0.0).sum(0), count, out=np.zeros(x.shape[1]), where=count > 0
        )
        var = np.divide(
            np.where(valid, (x - mean) ** 2, 0.0).sum(0),
            count,
            out=np.ones(x.shape[1]),
            where=count > 0,
        )
        return cls(mean, np.maximum(np.sqrt(var), 1e-6), batch.names, ids.copy())

    def transform(self, batch: PhaseFeatureBatch):
        if batch.names != self.names:
            raise ValueError("Feature schema differs from fitted scaler")
        return np.where(batch.valid, (batch.values - self.mean) / self.scale, 0.0).astype(
            np.float32
        )

    def state(self):
        return dict(
            mean=self.mean,
            scale=self.scale,
            names=np.asarray(self.names),
            fit_indices=self.fit_indices,
        )

    @classmethod
    def from_state(cls, state):
        return cls(
            state["mean"], state["scale"], tuple(state["names"].tolist()), state["fit_indices"]
        )
