"""Causal raw-current decimation and independent high-frequency envelopes."""

import numpy as np
from scipy.signal import upfirdn, sosfilt, butter, firwin
from .constants import FS, FIR, HF_SOS, ENV_SOS
from .pll import robust_pll


def condition(raw, bands=(0, 1), sample_rate_hz=2_000_000):
    """Causal FIR decimation, no advance/group-delay compensation.

    FIR group delay is 0.2 ms. No full-record mean subtraction or filtfilt.
    High-band envelope branches remain separate from the low-band detector.
    """
    raw = np.asarray(raw, float)
    if sample_rate_hz not in (1_000_000, 2_000_000):
        raise ValueError("Registered frontends support 1 MHz and 2 MHz only")
    down = sample_rate_hz // FS
    fir = FIR if sample_rate_hz == 2_000_000 else firwin(401, 20_000, fs=sample_rate_hz, window=("kaiser", 7.5))
    hf_sos = HF_SOS if sample_rate_hz == 2_000_000 else [butter(3, b, fs=sample_rate_hz, btype="bandpass", output="sos") for b in [(10_000,100_000),(100_000,400_000)]]
    env_sos = ENV_SOS if sample_rate_hz == 2_000_000 else butter(2,8_000,fs=sample_rate_hz,output="sos")
    n = (len(raw) + down-1) // down
    low = upfirdn(fir, raw, down=down)[:n]
    env = []
    if any(b not in (0, 1) for b in bands) or len(set(bands)) != len(bands):
        raise ValueError('High-frequency bands must be unique indices 0 or 1')
    for band in bands:
        sos = hf_sos[band]
        hf = sosfilt(sos, raw)
        power = sosfilt(env_sos, hf * hf)
        env.append(np.maximum(upfirdn(fir, power, down=down)[:n], 0))
    # A separate causal anti-alias filter for the synchronizer.
    lp = sosfilt(butter(2, 800, fs=FS, output="sos"), low)
    ph5, fr5 = robust_pll(lp[::10])
    # At a detector sample, use only the most recent PLL phase and rate.
    ix = np.minimum(np.arange(n) // 10, len(ph5) - 1)
    ph = ph5[ix] + (np.arange(n) - ix * 10) / FS * 2 * np.pi * fr5[ix]
    return low, np.array(env), ph, fr5
