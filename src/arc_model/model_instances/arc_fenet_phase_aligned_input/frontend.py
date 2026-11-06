"""Pure input switch. No auxiliary feature branch or original-parameter mutation."""

import numpy as np
from scipy.signal import firwin, upfirdn, resample_poly

BACKBONE = "arc_fenet"


def aligned_wave(raw, fs, reference_times, *, origin_seconds=0.0, anchor_seconds):
    x = np.asarray(raw)
    t = np.asarray(reference_times, dtype=float)
    if fs not in [1000000, 2000000] or x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Native1/2MHz finite current required")
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or not (np.diff(t) > 0).all():
        raise ValueError("Monotonic observed phase-grid times required")
    if (
        t[0] < origin_seconds
        or t[-1] > anchor_seconds + 1e-9
        or t[-1] > origin_seconds + len(x) / fs
    ):
        raise ValueError("Incomplete/future cycle support")
    rate = 250000 if BACKBONE == "arc_fenet" else 50000
    cutoff = 112500 if BACKBONE == "arc_fenet" else 20000
    signal = np.abs(x) if BACKBONE == "hydra" else x
    low = upfirdn(
        firwin(801 if fs == 2000000 else 401, cutoff, fs=fs, window=("kaiser", 7.5)),
        signal,
        down=fs // rate,
    )[: (len(x) + fs // rate - 1) // (fs // rate)]
    n = 5000 if BACKBONE == "arc_fenet" else 1000
    positions = (np.interp(np.arange(n) / n, np.linspace(0, 1, len(t)), t) - origin_seconds) * rate
    if positions[-1] > len(low) - 1:
        raise ValueError("Last interpolation sample not observed")
    w = np.interp(positions, np.arange(len(low)), low).astype("float32")
    if BACKBONE == "arc_fusion":
        a = resample_poly(w, 1, 4).astype("float32")
        d = np.gradient(a)
        return dict(
            temporal=np.stack([a, d, np.gradient(d)])[None],
            spectrum=np.abs(np.fft.rfft(a))[1:].astype("float32"),
        )
    return w
