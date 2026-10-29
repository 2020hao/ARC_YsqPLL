"""26 waveform, energy, phase-bin and transient descriptors."""

import numpy as np


def descriptors(wave, envelope):
    """26 descriptors: six waveform/global-energy + 2x8 energy shapes + 4 cues.

    Same computations for fixed-time and phase-resampled windows.
    """
    rms2 = np.mean(wave**2) + 1e-12
    norm = wave / np.sqrt(rms2)
    diff = np.diff(norm)
    global_e = envelope.mean(axis=1)
    bins = np.stack([part.mean(axis=1) for part in np.array_split(envelope, 8, axis=1)], axis=1)
    shapes = np.log1p(bins / (global_e[:, None] + 1e-20))
    tkeo = norm[1:-1] ** 2 - norm[:-2] * norm[2:]
    return np.r_[
        np.log10(rms2),
        np.max(np.abs(norm)),
        np.mean(norm**4),
        np.mean(np.abs(norm) < 0.1),
        np.log10(global_e / rms2 + 1e-20),
        shapes.ravel(),
        np.mean(np.abs(diff)),
        np.mean(diff**2),
        np.mean(np.abs(tkeo)),
        np.mean(norm),
    ].astype(np.float32)
