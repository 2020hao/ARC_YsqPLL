"""Parameter-free single-input candidate; no auxiliary model branch.

Output is a serialized representation, NOT a reconstructed physical waveform.
The first 500 positions carry low-frequency current; two 250-position blocks
carry interpolated log RMS band-energy profiles. The original 20 phase bins
are not claimed to contain 250 independently measured positions.
"""

import numpy as np
from scipy.signal import resample_poly


def pack_current_energy(wave, energy, duration):
    wave, energy = np.asarray(wave), np.asarray(energy)
    duration = np.asarray(duration)
    if wave.ndim != 2 or wave.shape[1] != 1000:
        raise ValueError("Expected wave [N,1000]")
    if energy.shape != (len(wave), 2, 20) or duration.shape != (len(wave),):
        raise ValueError("Expected energy [N,2,20] and duration [N]")
    if not np.isfinite(wave).all() or not np.isfinite(energy).all():
        raise ValueError("Finite wave and energy required")
    if (energy < 0).any() or not np.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError("Nonnegative energy and positive duration required")
    low = resample_poly(wave, 1, 2, axis=-1)
    # Bins span equal phase, not necessarily equal time; this is a nominal-bin
    # RMS proxy. No claim of calibrated physical RMS under phase-rate variation.
    profile = np.log1p(np.sqrt(energy / (duration[:, None, None] / 20)))
    x = (np.arange(20) + .5) / 20
    xp = (np.arange(250) + .5) / 250
    out = np.stack([np.interp(xp, x, row) for row in profile.reshape(-1, 20)])
    return np.concatenate([low, out.reshape(len(wave), 500)], axis=1).astype(np.float32)
