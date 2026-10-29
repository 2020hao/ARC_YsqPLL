"""Current-only causal robust SOGI-PLL; 40 ms startup."""

import numpy as np
from .constants import FS_PLL


def robust_pll(x, fs=FS_PLL, robust=True):
    """Trapezoidal SOGI + bounded PI phase loop, sequential after 40 ms init.

    Startup least squares estimates DC and harmonics 1,3,5 at nominal 50 Hz.
    PI uses nominal natural frequency 3 Hz, damping .707. Frequency-integrator
    bounds 45--55 Hz; instantaneous phase rate bounded 40--60 Hz.
    """
    x = np.asarray(x, float)
    init = min(int(0.04 * fs), len(x) - 1)
    tt = np.arange(init) / fs
    design = np.column_stack(
        [np.ones(init)] + [f(2 * np.pi * 50 * h * tt) for h in (1, 3, 5) for f in (np.sin, np.cos)]
    )
    coef = np.linalg.lstsq(design, x[:init], rcond=None)[0]
    dc = coef[0]
    amp = max(np.hypot(coef[1], coef[2]), 1e-8)
    phi0 = np.arctan2(coef[2], coef[1])
    phase = np.full(len(x), np.nan)
    freq = np.full(len(x), np.nan)
    theta = 2 * np.pi * 50 * init / fs + phi0
    a, b = amp * np.sin(theta), -amp * np.cos(theta)
    prev = float(x[init - 1] - dc)
    integ = 0.0
    wn = 2 * np.pi * 3
    kp, ki = 2 * 0.707 * wn, wn**2
    dt = 1 / fs
    phase[init] = theta
    freq[init] = 50.0
    for n in range(init + 1, len(x)):
        u = float(x[n] - dc)
        # Bound large amplitude outliers using past SOGI amplitude.
        if robust:
            bound = 4 * max(np.hypot(a, b), 0.1 * amp)
            u = max(-bound, min(bound, u))
        omega = 2 * np.pi * 50 + integ
        w = 0.5 * dt * omega
        k = np.sqrt(2.0)
        r1 = (1 - k * w) * a - w * b + k * w * (u + prev)
        r2 = w * a + b
        a = (r1 - w * r2) / (1 + k * w + w * w)
        b = r2 + w * a
        # Predicted angle at the present sample.
        pred = theta + dt * (2 * np.pi * freq[n - 1])
        err = np.arctan2(a * np.cos(pred) + b * np.sin(pred), a * np.sin(pred) - b * np.cos(pred))
        if robust:
            err = max(-0.35, min(0.35, err))
        integ = max(-2 * np.pi * 5, min(2 * np.pi * 5, integ + ki * dt * err))
        rate = max(2 * np.pi * 40, min(2 * np.pi * 60, 2 * np.pi * 50 + integ + kp * err))
        theta += rate * dt
        phase[n] = theta
        freq[n] = rate / (2 * np.pi)
        prev = u
    return phase, freq
