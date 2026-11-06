"""Causal 2 MHz -> 1 MHz adaptation; native 1 MHz is an exact identity path."""

import numpy as np
from scipy.signal import firwin, kaiserord, upfirdn


def antialias_coefficients():
    """Exact coefficients used by prepare_self462_1m_signals_20261105.py."""
    taps, beta = kaiserord(80, (500_000 - 400_000) / 1_000_000)
    taps += 1 - taps % 2
    return firwin(taps, 450_000, fs=2_000_000, window=("kaiser", beta))


class CausalOneMHz:
    def __init__(self, source_rate_hz):
        if source_rate_hz not in (1_000_000, 2_000_000):
            raise ValueError("Supported input rates: 1 and 2 MHz")
        self.source_rate_hz = int(source_rate_hz)
        self.input_samples = 0
        self.coefficients = antialias_coefficients() if source_rate_hz == 2_000_000 else None
        self.history = np.zeros(102, dtype=np.float64)

    @property
    def delay_seconds(self):
        return 0.0 if self.coefficients is None else 25.5e-6

    def update(self, values):
        values = np.asarray(values)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Finite single-channel samples required")
        if self.coefficients is None:
            self.input_samples += len(values)
            return values.copy()
        if not len(values):
            return np.empty(0, dtype=float)
        start = self.input_samples
        joined = np.r_[self.history, values]
        # Align the polyphase calculation to absolute even input indices even
        # when the caller supplies odd-sized chunks. No future padding is read.
        pad = start % 2
        padded = np.r_[np.zeros(pad), joined]
        first = start + start % 2
        count = max(0, (start + len(values) - 1 - first) // 2 + 1)
        offset = (first - (start - len(self.history) - pad)) // 2
        result = upfirdn(self.coefficients, padded, down=2)[offset:offset + count]
        self.history = joined[-102:].copy()
        self.input_samples += len(values)
        return result
