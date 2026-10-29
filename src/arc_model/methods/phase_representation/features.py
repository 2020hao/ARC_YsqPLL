"""Fixed 10D candidate and capacity-matched phase-free/phase-shift controls."""

from __future__ import annotations
import numpy as np
from .contracts import PhaseFeatureBatch


def phase_features(
    energy,
    *,
    phase_valid=None,
    mode="phase",
    shifts=None,
    half_width_degrees=18.0,
    energy_floor=1e-20,
):
    """Use two native-rate band energies in equal reference-phase bins.

    ``energy`` is [windows, 2, even_bins]. Coordinate availability is returned
    separately: fixed20 controls remain numerically defined even when no native
    PLL reference exists. The caller owns any predeclared gating/intersection.
    The floor is a numerical floor, not a calibrated physical noise threshold.
    """
    e = np.asarray(energy, dtype=np.float64)
    if e.ndim != 3 or e.shape[1] != 2 or e.shape[2] < 4 or e.shape[2] % 2:
        raise ValueError("Expected [windows,2,even_bins>=4] energy")
    if not np.isfinite(e).all() or (e < 0).any():
        raise ValueError("Energy must be finite and nonnegative")
    if mode not in ("phase", "phase_free", "shifted"):
        raise ValueError("Unknown phase feature mode")
    if not 0 < half_width_degrees < 90 or energy_floor <= 0:
        raise ValueError("Invalid neighborhood width or numerical floor")
    native = np.ones(len(e), bool) if phase_valid is None else np.asarray(phase_valid, bool)
    if native.shape != (len(e),):
        raise ValueError("phase_valid must be one-dimensional")
    total = e.sum(-1)
    ok = total > energy_floor
    p = np.divide(e, total[..., None], out=np.zeros_like(e), where=ok[..., None])
    if mode == "shifted":
        if shifts is None:
            raise ValueError("Shift control needs explicit deterministic per-window shifts")
        shifts = np.asarray(shifts)
        if shifts.shape != (len(e),) or shifts.dtype.kind not in "iu":
            raise ValueError("Integer shifts must align with windows")
        p = np.stack([np.roll(a, int(s), axis=-1) for a, s in zip(p, shifts)]) if len(e) else p
    elif shifts is not None:
        raise ValueError("Shifts are only accepted by the shifted control")
    bins = e.shape[-1]
    theta = (np.arange(bins) + 0.5) * 2 * np.pi / bins
    neighborhood = (
        np.abs((theta + np.pi / 2) % np.pi - np.pi / 2) <= np.deg2rad(half_width_degrees) + 1e-12
    )
    if not neighborhood.any():
        raise ValueError("Neighborhood is narrower than the phase-bin resolution")
    values = [
        np.where(ok[:, b], np.log(np.maximum(total[:, b], energy_floor)), 0.0) for b in range(2)
    ]
    valid = [ok[:, b] for b in range(2)]
    names = ["log_energy_low", "log_energy_high"]
    for b, band in enumerate(("low", "high")):
        u = p[:, b]
        c1, s1 = u @ np.cos(theta), u @ np.sin(theta)
        c2, s2 = u @ np.cos(2 * theta), u @ np.sin(2 * theta)
        if mode == "phase_free":
            stats = [
                -(u * np.log(np.maximum(u, 1e-30))).sum(-1) / np.log(bins),
                u.max(-1),
                np.hypot(c1, s1),
                np.hypot(c2, s2),
            ]
            labels = ("entropy", "peak_fraction", "concentration1", "concentration2")
        else:
            stats = [
                u[:, neighborhood].sum(-1),
                abs(u[:, : bins // 2].sum(-1) - u[:, bins // 2 :].sum(-1)),
                c2,
                s2,
            ]
            labels = ("zero_neighborhood", "half_asymmetry", "moment2_cos", "moment2_sin")
        values.extend(np.where(ok[:, b], a, 0.0) for a in stats)
        valid.extend([ok[:, b]] * 4)
        names.extend(f"{band}_{label}" for label in labels)
    return PhaseFeatureBatch(
        np.column_stack(values), tuple(names), np.column_stack(valid), native, mode
    )
