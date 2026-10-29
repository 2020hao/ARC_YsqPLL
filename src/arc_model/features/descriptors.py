"""Extend the 26 physical descriptors to the fixed 42-dimensional vector."""

import numpy as np
from .schema import NAMES, GROUPS


def extend(waves, physical):
    x = np.asarray(waves, dtype=np.float64)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-12)
    z = x / rms
    spec = np.abs(np.fft.rfft(z, axis=-1)) ** 2
    p = spec[..., 1:] + 1e-20
    prob = p / p.sum(-1, keepdims=True)
    harmonic = np.log1p(np.sqrt(spec[..., 2:10] / (spec[..., 1:2] + 1e-12)))
    dx = np.diff(z, axis=-1)
    length = z.shape[-1]
    phase = 2 * np.pi * np.arange(length) / length
    # Fixed phase-coordinate mask; it is only a waveform-phase mask after PLL.
    mask = np.abs(np.sin(phase)) < np.sin(np.pi / 8)
    ex = np.stack(
        [
            np.log1p(np.sqrt(spec[..., 2:21].sum(-1) / (spec[..., 1] + 1e-12))),
            -(prob * np.log(prob)).sum(-1) / np.log(prob.shape[-1]),
            np.mean((z[..., : length // 2] + z[..., length // 2 :]) ** 2, axis=-1),
            np.mean(z > 0, axis=-1),
            np.max(np.abs(dx), axis=-1) / np.sqrt(np.mean(dx * dx, axis=-1) + 1e-12),
            np.exp(np.mean(np.log(p), axis=-1)) / np.mean(p, axis=-1),
            np.sort(z * z, axis=-1)[..., -max(1, length // 20) :].sum(-1)
            / (z * z).sum(-1).clip(1e-12),
            np.mean(z[..., mask] ** 2, axis=-1) / (np.mean(z[..., ~mask] ** 2, axis=-1) + 1e-12),
        ],
        axis=-1,
    )
    result = np.concatenate([physical, harmonic, ex], axis=-1).astype("float32")
    assert result.shape[-1] == len(NAMES) == len(GROUPS)
    assert np.isfinite(result).all()
    return result
