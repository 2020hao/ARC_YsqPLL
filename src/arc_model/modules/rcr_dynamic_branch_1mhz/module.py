"""Optional current-only dynamic branch; disabled operation is an exact identity.

Raw preprocessing is shared, but each model/data/fold/seed owns its own fitted
state. Extraction preserves old equations; no historical cohort metric is
assigned to this newly packaged 1 MHz correction route.
"""

import numpy as np
from numbers import Integral
from scipy.special import expit, logit

from arc_model.contracts import BranchEvent, CurrentChunk, FitIdentity, ScorePacket
from .activity import CycleStream
from .features import native_dynamic_summary
from .fitting import fit as fit_linear
from .fusion import corrected_margin
from .sampling import CausalOneMHz
from .state import groups_digest, indices_digest, load_bundle, save_bundle, validate_state


class RCRDynamicBranch1MHz:
    status = "implemented_unvalidated_on_current462_correction"

    def __init__(self, state=None, identity=None):
        if identity is not None and not isinstance(identity, FitIdentity):
            raise TypeError("identity must be a FitIdentity")
        self.identity = identity
        self.state = None
        if state is not None:
            validate_state(state)
            self.state = {key: np.asarray(value).copy() for key, value in state.items()}
        self.reset()

    def reset(self):
        self._record_id = None
        self._source_next = None
        self._origin = None
        self._sampling = None
        self._stream = None
        self._pending = np.empty(0, dtype=float)

    @staticmethod
    def transform(raw, mask):
        import torch

        x = raw if torch.is_tensor(raw) else torch.as_tensor(np.asarray(raw))
        m = mask if torch.is_tensor(mask) else torch.as_tensor(np.asarray(mask))
        if not torch.isfinite(x).all() or not torch.isfinite(m).all():
            raise ValueError("Finite raw profiles and history mask required")
        if not ((m == 0) | (m == 1)).all():
            raise ValueError("History availability must be binary")
        with torch.no_grad():
            return native_dynamic_summary(x, m).cpu().numpy()

    def observe(self, chunk):
        if not isinstance(chunk, CurrentChunk):
            raise TypeError("Pass CurrentChunk with explicit acquisition and timing metadata")
        if self._record_id != chunk.record_id:
            self.reset()
            self._record_id = chunk.record_id
            self._source_next = chunk.start_sample
            self._origin = chunk.origin_seconds + chunk.start_sample / chunk.sample_rate_hz
            self._source_origin = chunk.origin_seconds
            self._sampling = CausalOneMHz(chunk.sample_rate_hz)
            self._stream = CycleStream(1_000_000, origin_seconds=self._origin)
        if chunk.sample_rate_hz != self._sampling.source_rate_hz:
            raise ValueError("Sampling rate changed inside a recording")
        if chunk.origin_seconds != self._source_origin:
            raise ValueError("Time origin changed inside a recording")
        if chunk.start_sample != self._source_next:
            raise ValueError("Current chunks must be consecutive; reset explicitly at a gap")
        self._source_next += len(chunk.values)
        adapted = self._sampling.update(chunk.values)
        self._pending = np.r_[self._pending, adapted]
        events = []
        size = self._stream.chunk_samples
        while len(self._pending) >= size:
            rows, _ = self._stream.update(self._pending[:size])
            self._pending = self._pending[size:]
            for row in rows:
                features = self.transform(row["input"][None], row["mask"][None])[0]
                events.append(BranchEvent(
                    record_id=chunk.record_id,
                    start_sample=row["start_sample"], stop_sample_exclusive=row["stop_sample_exclusive"],
                    sample_rate_hz=1_000_000, origin_seconds=self._origin,
                    available_at_seconds=self._origin + row["available_seconds"],
                    coordinate=row["coordinate"], segment=row["segment"], epoch=row["epoch"],
                    phase_status=row["phase_status"], features=features, history_mask=row["mask"].copy(),
                ))
        return events

    @staticmethod
    def _margin(scores):
        if scores.kind == "probability":
            if scores.threshold != 0.5:
                raise ValueError("Enabled probability correction requires the declared 0.5 convention")
            return logit(np.clip(scores.values, 1e-12, 1 - 1e-12))
        if scores.threshold != 0:
            raise ValueError("Enabled margin correction requires the declared zero convention")
        return scores.values

    def correct(self, scores, features=None, *, enabled=False, expected_identity=None):
        if not isinstance(scores, ScorePacket):
            raise TypeError("Use ScorePacket to declare score semantics and threshold")
        if not enabled:
            return scores
        if self.state is None:
            raise ValueError("Fit or load a branch state before enabling correction")
        if self.identity is None:
            raise ValueError("Enabled correction requires a bound FitIdentity")
        if expected_identity is not None:
            if not isinstance(expected_identity, FitIdentity):
                raise TypeError("expected_identity must be a FitIdentity")
            if self.identity != expected_identity:
                raise ValueError("Wrong model/checkpoint/data/split/seed identity")
        if self.identity.score_kind != scores.kind:
            raise ValueError("Fitted and current score conventions differ")
        if str(self.state["fusion"]) == "whole_path" and float(self.state["strength"]) == 0:
            return scores
        x = np.asarray(features)
        if x.shape != (len(scores.values), 36) or not np.isfinite(x).all():
            raise ValueError("Finite aligned [N,36] dynamics required")
        result = corrected_margin(self.state, self._margin(scores), x)
        if scores.kind == "probability":
            result = expit(result)
        return ScorePacket(result, kind=scores.kind, threshold=scores.threshold)

    def predict(self, raw_scores, dynamic=None, *, enabled=False,
                score_kind="logit_margin", threshold=None, expected_identity=None):
        packet = ScorePacket(raw_scores, kind=score_kind, threshold=threshold)
        return self.correct(packet, dynamic, enabled=enabled,
                            expected_identity=expected_identity).values

    score = predict

    def fit(self, features, labels, scores, train_indices, groups, *, identity,
            heldout_indices, honest_oof=False, calibration_slope=1.0,
            calibration_intercept=0.0, calibration_fit_indices=None,
            fusion="whole_path", strength=1.0, quantile=0.95, epochs=80):
        """Explicit offline operation; never called by observe/correct.

        ``scores`` must be honest group-OOF scores within training groups. This
        method validates declared support and group isolation; callers retain
        the producer checkpoints and OOF receipts as independent evidence.
        """
        if not isinstance(identity, FitIdentity) or not isinstance(scores, ScorePacket):
            raise TypeError("FitIdentity and ScorePacket are required")
        x, y, g = np.asarray(features), np.asarray(labels), np.asarray(groups)
        def integer_indices(values, name):
            values = np.asarray(values)
            if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"{name} must contain integer indices without truncation")
            return values.astype(np.int64)

        ids = integer_indices(train_indices, "Training indices")
        held = integer_indices(heldout_indices, "Held-out indices")
        if x.ndim != 2 or x.shape[1] != 36 or not np.isfinite(x).all():
            raise ValueError("Expected finite [N,36] features")
        if y.shape != (len(x),) or g.shape != y.shape or scores.values.shape != y.shape:
            raise ValueError("Features, labels, groups and scores must align")
        if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
            raise ValueError("Unique nonempty training indices required")
        if held.ndim != 1 or not len(held) or ids.min() < 0 or ids.max() >= len(x) or held.min() < 0 or held.max() >= len(x):
            raise ValueError("Valid explicit training and held-out indices required")
        for value in g[np.r_[ids, held]]:
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError("Training and held-out group identities cannot be empty")
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                raise ValueError("Training and held-out group identities must be finite")
        if np.intersect1d(ids, held).size or np.intersect1d(g[ids], g[held]).size:
            raise ValueError("Training and held-out groups overlap")
        if not honest_oof:
            raise ValueError("Declare and preserve honest group-OOF parent scores")
        if set(y[ids]) != {0, 1} or (y[ids] == 0).sum() < 2:
            raise ValueError("Both training classes and at least two normal examples required")
        if identity.score_kind != scores.kind or identity.fit_indices_sha256 != indices_digest(ids):
            raise ValueError("Training identity or score convention mismatch")
        if identity.train_groups_sha256 != groups_digest(g[ids]):
            raise ValueError("Training group identity mismatch")
        cal_ids = ids if calibration_fit_indices is None else integer_indices(calibration_fit_indices, "Calibration indices")
        if not len(cal_ids):
            raise ValueError("Nonempty calibration fit support is required")
        if not set(cal_ids).issubset(set(ids)):
            raise ValueError("Calibration must be fitted inside training groups")
        if not 0 < quantile < 1 or not isinstance(epochs, Integral) or isinstance(epochs, bool) or epochs < 0:
            raise ValueError("Invalid predeclared fit settings")
        if calibration_slope <= 0 or not np.isfinite([calibration_slope, calibration_intercept]).all():
            raise ValueError("Finite positive-slope calibration required")
        base = self._margin(scores)
        calibrated = calibration_slope * base + calibration_intercept
        _, distance, state = fit_linear(x, y, calibrated, ids, np.arange(36), identity.seed,
                                        device="cpu", epochs=epochs)
        normal = ids[y[ids] == 0]
        state.update(dimension=np.asarray(36), fusion=np.asarray(fusion), strength=np.asarray(strength),
                     calibration_slope=np.asarray(calibration_slope),
                     calibration_intercept=np.asarray(calibration_intercept),
                     calibration_fit_indices=cal_ids.copy(),
                     gate_threshold=np.asarray(max(float(np.quantile(distance[normal], quantile)), 1e-6)))
        validate_state(state)
        self.state, self.identity = state, identity
        return self

    def save(self, path):
        if self.state is None:
            raise ValueError("No fitted state to save")
        save_bundle(path, self.state, self.identity)

    @classmethod
    def load(cls, path, *, expected_identity=None):
        state, identity = load_bundle(path, expected_identity)
        return cls(state, identity)
