"""Typed boundaries between acquisition, the optional branch and classifiers."""

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class CurrentChunk:
    values: np.ndarray
    sample_rate_hz: int
    record_id: str
    start_sample: int
    origin_seconds: float = 0.0
    channel: str = "trunk_current"
    unit: str = "a.u."

    def __post_init__(self):
        values = np.asarray(self.values)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("A nonempty finite single-channel current chunk is required")
        if not isinstance(self.sample_rate_hz, Integral) or isinstance(self.sample_rate_hz, bool) or self.sample_rate_hz not in (1_000_000, 2_000_000):
            raise ValueError("Current input must be 1 or 2 MHz; 5 kHz is reference-only")
        if not isinstance(self.start_sample, Integral) or isinstance(self.start_sample, bool):
            raise ValueError("start_sample must be an integer acquisition index")
        if not self.record_id or self.start_sample < 0 or not np.isfinite(self.origin_seconds):
            raise ValueError("Record identity, nonnegative sample index and finite origin required")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class ScorePacket:
    values: np.ndarray
    kind: str = "logit_margin"
    threshold: float | None = None
    positive_class: str = "arc"

    def __post_init__(self):
        values = np.asarray(self.values)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Scores must be a finite vector")
        if self.kind not in ("logit_margin", "svm_margin", "probability"):
            raise ValueError("Declare logit_margin, svm_margin or probability explicitly")
        if self.kind == "probability" and ((values < 0).any() or (values > 1).any()):
            raise ValueError("Probability scores must be in [0,1]")
        if self.threshold is None:
            object.__setattr__(self, "threshold", 0.5 if self.kind == "probability" else 0.0)
        if not np.isfinite(self.threshold) or self.positive_class != "arc":
            raise ValueError("Finite threshold and positive class arc required")
        object.__setattr__(self, "values", values)

    @property
    def decisions(self):
        return self.values >= self.threshold


@dataclass(frozen=True)
class FitIdentity:
    model_id: str
    parent_checkpoint_sha256: str
    dataset_id: str
    split_sha256: str
    fit_indices_sha256: str
    train_groups_sha256: str
    seed: int
    score_kind: str
    feature_version: str = "native_dynamic36_v1"
    reference_version: str = "rcr_cycle_v2_20261104"
    sampling_version: str = "causal_aa_1mhz_v1"

    def __post_init__(self):
        for name in ("model_id", "parent_checkpoint_sha256", "dataset_id", "split_sha256",
                     "fit_indices_sha256", "train_groups_sha256"):
            if not getattr(self, name):
                raise ValueError(f"Missing fit identity field: {name}")
        if self.score_kind not in ("logit_margin", "svm_margin", "probability"):
            raise ValueError("Unknown fit score convention")
        if not isinstance(self.seed, Integral) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("Fit seed must be a nonnegative integer")
        object.__setattr__(self, "seed", int(self.seed))

    def as_dict(self):
        return asdict(self)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class BranchEvent:
    record_id: str
    start_sample: int
    stop_sample_exclusive: int
    sample_rate_hz: int
    origin_seconds: float
    available_at_seconds: float
    coordinate: str
    segment: int
    epoch: int
    phase_status: str
    features: np.ndarray
    history_mask: np.ndarray

    def __post_init__(self):
        if not self.record_id or not np.isfinite(self.origin_seconds):
            raise ValueError("Record identity and finite time origin required")
        if self.sample_rate_hz != 1_000_000:
            raise ValueError("Branch event sample coordinates must be 1 MHz")
        if not 0 <= self.start_sample < self.stop_sample_exclusive:
            raise ValueError("Nonempty causal support required")
        end = self.origin_seconds + self.stop_sample_exclusive / self.sample_rate_hz
        if end > self.available_at_seconds + 1e-12:
            raise ValueError("An event cannot be available before its current support ends")
        if self.coordinate not in ("native", "fallback"):
            raise ValueError("Native and nominal coordinates must remain explicit")
        if np.asarray(self.features).shape != (36,) or np.asarray(self.history_mask).shape != (9,):
            raise ValueError("Expected 36 features and nine history flags")
        if not np.isfinite(self.available_at_seconds) or not np.isfinite(self.features).all():
            raise ValueError("Finite event timing and features required")
        if not np.isin(self.history_mask, [0, 1]).all():
            raise ValueError("Binary causal history availability required")
