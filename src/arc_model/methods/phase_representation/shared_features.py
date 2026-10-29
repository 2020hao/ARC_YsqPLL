"""One frozen eight-column definition for every backbone and both datasets."""

import numpy as np

NAMES = (
    "relative_energy_10_100k",
    "relative_energy_100_400k",
    "zero_fraction_10_100k",
    "moment2_cos_10_100k",
    "moment2_sin_10_100k",
    "zero_fraction_100_400k",
    "moment2_cos_100_400k",
    "moment2_sin_100_400k",
)


def shared_eight(wave, energy, duration, *, shifts=None):
    """Return relative strength (2) and phase-location statistics (6).

    LF denominator is duration times mean squared *phase-grid* waveform: an
    energy proxy, not a calibrated physical integral on a nonuniform time grid.
    Near-zero reference: numerical floor only, no label-dependent gating.
    Native-rate HF energies are integrated before the call. Shifts rotate only
    their normalized position distributions, retaining the same total energies.
    """
    w, e, t = np.asarray(wave, float), np.asarray(energy, float), np.asarray(duration, float)
    if w.ndim != 2 or w.shape[1] != 1000 or e.shape != (len(w), 2, 20):
        raise ValueError("Expected wave[N,1000], energy[N,2,20]")
    if t.shape != (len(w),) or not np.isfinite(t).all() or (t <= 0).any():
        raise ValueError("Positive finite window duration required")
    if not np.isfinite(w).all() or not np.isfinite(e).all() or (e < 0).any():
        raise ValueError("Finite current and nonnegative energy required")
    total = e.sum(-1)
    lf_proxy = t * np.mean(w * w, axis=1)
    relative = np.log1p(total / np.maximum(lf_proxy[:, None], 1e-20))
    p = np.divide(e, total[..., None], out=np.zeros_like(e), where=total[..., None] > 1e-20)
    if shifts is not None:
        shifts = np.asarray(shifts)
        if shifts.shape != (len(w),) or shifts.dtype.kind not in "iu":
            raise ValueError("One integer position shift per window required")
        p = np.stack([np.roll(row, int(k), axis=-1) for row, k in zip(p, shifts)])
    theta = (np.arange(20) + .5) * 2 * np.pi / 20
    zero = np.abs((theta + np.pi / 2) % np.pi - np.pi / 2) <= np.pi / 10 + 1e-12
    parts = [relative]
    for b in range(2):
        parts.append(np.column_stack([p[:, b, zero].sum(-1),
                                      p[:, b] @ np.cos(2 * theta),
                                      p[:, b] @ np.sin(2 * theta)]))
    return np.concatenate(parts, axis=1).astype(np.float32)
