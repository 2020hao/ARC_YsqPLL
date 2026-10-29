"""Legacy-compatible paired nominal-anchor windows and cycle dispersion.

Aligned cycles can repeat when adjacent nominal anchors select the same PLL turn.
The refactor preserves this experiment contract; it does not silently change
the cycle-selection protocol used by existing checkpoints.
"""

import numpy as np
from .constants import FS
from arc_model.features.physical import descriptors


def windows(low, envelope, phase, count=32, length=1000):
    """Paired fixed-time and PLL-cycle windows, label-free deterministic times.

    Records >=0.9 s: 32 anchors from 0.20 s at 20 ms increments.
    Short records: anchors from 0.05 s. Each aligned cycle starts at the
    first phase wrap after its fixed-window anchor; offset is recorded.
    Both use 1000 input points; PLL cycles explicitly use linear phase
    resampling. This is interpolation, not zero-padding or native-only cuts.
    """
    n = len(low)
    start = 0.2 if n / FS >= 0.9 else 0.05
    grid = np.arange(n)
    valid = np.isfinite(phase)
    vgrid = grid[valid]
    vphase = phase[valid]
    fixed = []
    aligned = []
    pf = []
    pa = []
    meta = []
    for j in range(count if start == 0.2 else max(1, int((n / FS - start) / 0.02))):
        a = int(round((start + j * 0.02) * FS))
        if a + length > n or a >= n or not np.isfinite(phase[a]):
            break
        turn = np.ceil(phase[a] / (2 * np.pi)) * 2 * np.pi
        if turn + 2 * np.pi > vphase[-1]:
            break
        # Uniform phase grid excludes endpoint, just as an ordinary time block.
        positions = np.interp(turn + 2 * np.pi * np.arange(length) / length, vphase, vgrid)
        end = float(np.interp(turn + 2 * np.pi, vphase, vgrid))
        w0 = low[a : a + length]
        w1 = np.interp(positions, grid, low)
        e0 = envelope[:, a : a + length]
        e1 = np.array([np.interp(positions, grid, e) for e in envelope])
        fixed.append(w0)
        aligned.append(w1)
        pf.append(descriptors(w0, e0))
        pa.append(descriptors(w1, e1))
        meta.append(
            dict(
                fixed_start_sample=a,
                pll_start_sample=float(positions[0]),
                pll_end_sample=end,
                pll_original_samples=end - positions[0],
                start_offset_ms=(positions[0] - a) / FS * 1000,
            )
        )
    return (
        np.asarray(fixed, np.float32),
        np.asarray(aligned, np.float32),
        np.asarray(pf),
        np.asarray(pa),
        meta,
    )


def dispersion(waves):
    """RMS spread of per-cycle RMS-normalized waveforms around their mean."""
    z = waves / np.sqrt(np.mean(waves**2, axis=1, keepdims=True) + 1e-12)
    return float(np.sqrt(np.mean((z - z.mean(axis=0)) ** 2)))
