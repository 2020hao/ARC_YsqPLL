"""Current-synchronized, causal cross-cycle representations.

This module owns feature semantics only. It does not load datasets, select folds,
train backbones, write artifacts, or compute paper statistics.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np

DEFAULT_LAGS: tuple[int, ...] = (1, 2, 4, 8)
_STATISTICS_PER_BAND = 4


def cycle_distributions(
    energy: np.ndarray,
    *,
    phase_free: bool = False,
    circular_shifts: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize band/bin energy into per-cycle distributions.

    ``energy`` has shape ``[cycle, band, phase_bin]``. Sorting produces the
    phase-position-free control; deterministic shifts produce the destroyed
    phase control while preserving every cycle's values.
    """

    values = np.asarray(energy, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("energy must have shape [cycle, band, phase_bin]")
    total = np.maximum(values.sum(axis=2, keepdims=True), 1e-12)
    distributions = values / total
    if circular_shifts is not None:
        shifts = np.asarray(circular_shifts, dtype=int)
        if shifts.shape != (len(distributions),):
            raise ValueError("circular_shifts must contain one shift per cycle")
        distributions = np.stack(
            [np.roll(cycle, int(shift), axis=1) for cycle, shift in zip(distributions, shifts)]
        )
    if phase_free:
        distributions = np.sort(distributions, axis=2)
    return distributions


def deterministic_phase_shifts(dataset: str, count: int, bins: int) -> np.ndarray:
    """Return reproducible per-cycle shifts for the phase-destroyed control."""

    return np.array(
        [
            int.from_bytes(
                hashlib.sha256(f"cycle-dynamics|{dataset}|{index}".encode()).digest()[:4],
                "little",
            )
            % bins
            for index in range(count)
        ],
        dtype=int,
    )


def dynamic_features(
    distributions: np.ndarray,
    records: Sequence[Mapping[str, object]],
    *,
    lags: Sequence[int] = DEFAULT_LAGS,
) -> np.ndarray:
    """Compare each cycle with strictly earlier cycles from the same record.

    The last ``len(lags)`` columns are availability masks. Missing history is
    represented by zeros and never filled from future cycles.
    """

    values = np.asarray(distributions, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != len(records):
        raise ValueError("distributions and records must align row-for-row")
    bands = values.shape[1]
    per_lag = bands * _STATISTICS_PER_BAND
    output = np.zeros((len(values), len(lags) * per_lag + len(lags)), dtype=np.float64)
    lookup = {(str(row["record_id"]), int(row["cycle"])): i for i, row in enumerate(records)}
    for i, row in enumerate(records):
        features: list[float] = []
        available: list[float] = []
        current = values[i]
        for lag in lags:
            previous_index = lookup.get((str(row["record_id"]), int(row["cycle"]) - int(lag)))
            available.append(float(previous_index is not None))
            if previous_index is None:
                features.extend([0.0] * per_lag)
                continue
            previous = values[previous_index]
            for band in range(bands):
                u, v = current[band], previous[band]
                midpoint = 0.5 * (u + v)
                features.extend(
                    [
                        float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)),
                        float(np.mean(np.abs(u - v))),
                        float(
                            0.5 * np.sum(u * np.log((u + 1e-12) / (midpoint + 1e-12)))
                            + 0.5 * np.sum(v * np.log((v + 1e-12) / (midpoint + 1e-12)))
                        ),
                        float((np.max(u) - np.max(v)) / (np.max(u) + np.max(v) + 1e-12)),
                    ]
                )
        output[i] = features + available
    return output


def confidence_weights(
    distance: np.ndarray,
    normal_indices: np.ndarray,
    *,
    strength: float,
    quantile: float,
    mode: str,
    softness: float = 0.15,
) -> np.ndarray:
    """Convert train-normal distance into residual-fusion weights."""

    distance = np.asarray(distance, dtype=np.float64)
    normal_indices = np.asarray(normal_indices, dtype=int)
    if len(normal_indices) == 0:
        raise ValueError("normal_indices cannot be empty")
    if mode == "ungated":
        return np.full(len(distance), float(strength))
    threshold = max(float(np.quantile(distance[normal_indices], quantile)), 1e-6)
    if mode == "hard":
        return float(strength) * (distance <= threshold)
    if mode != "soft":
        raise ValueError(f"unknown confidence mode: {mode}")
    scale = softness * threshold + 1e-8
    return float(strength) / (1.0 + np.exp(np.clip((distance - threshold) / scale, -40, 40)))
