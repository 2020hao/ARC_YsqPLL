"""Chunked causal frontend, preserving audited state update equations."""

import numpy as np
from scipy.signal import sosfilt, butter, upfirdn
from .constants import FIR, HF_SOS, ENV_SOS, FS, FS_PLL


class StreamingPLL:
    def __init__(self):
        self.index = 0
        self.initial = []
        self.phase = np.nan
        self.freq = np.nan
        self.last_index = 0

    def push(self, sample):
        n = self.index
        self.index += 1
        if n < 200:
            self.initial.append(float(sample))
            return self.phase, self.freq
        if n == 200:
            tt = np.arange(200) / FS_PLL
            design = np.column_stack(
                [np.ones(200)]
                + [fun(2 * np.pi * 50 * h * tt) for h in (1, 3, 5) for fun in (np.sin, np.cos)]
            )
            c = np.linalg.lstsq(design, self.initial, rcond=None)[0]
            self.dc = c[0]
            self.amp = max(np.hypot(c[1], c[2]), 1e-8)
            self.phase = 2 * np.pi * 50 * 200 / FS_PLL + np.arctan2(c[2], c[1])
            self.a = self.amp * np.sin(self.phase)
            self.b = -self.amp * np.cos(self.phase)
            self.prev = self.initial[-1] - self.dc
            self.integ = 0.0
            self.freq = 50.0
            self.initial = []
            return self.phase, self.freq
        u = float(sample - self.dc)
        bound = 4 * max(np.hypot(self.a, self.b), 0.1 * self.amp)
        u = max(-bound, min(bound, u))
        omega = 2 * np.pi * 50 + self.integ
        w = 0.5 / FS_PLL * omega
        k = np.sqrt(2.0)
        r1 = (1 - k * w) * self.a - w * self.b + k * w * (u + self.prev)
        r2 = w * self.a + self.b
        self.a = (r1 - w * r2) / (1 + k * w + w * w)
        self.b = r2 + w * self.a
        pred = self.phase + 2 * np.pi * self.freq / FS_PLL
        err = np.arctan2(
            self.a * np.cos(pred) + self.b * np.sin(pred),
            self.a * np.sin(pred) - self.b * np.cos(pred),
        )
        err = max(-0.35, min(0.35, err))
        wn = 2 * np.pi * 3
        self.integ = max(-2 * np.pi * 5, min(2 * np.pi * 5, self.integ + wn**2 / FS_PLL * err))
        rate = max(
            2 * np.pi * 40, min(2 * np.pi * 60, 2 * np.pi * 50 + self.integ + 2 * 0.707 * wn * err)
        )
        self.phase += rate / FS_PLL
        self.freq = rate / (2 * np.pi)
        self.prev = u
        return self.phase, self.freq


class StreamingCurrentFrontend:
    def __init__(self):
        self.raw_index = 0
        self.low_index = 0
        self.history = np.zeros((3, 0), dtype=float)
        self.hf_zi = [np.zeros((len(s), 2)) for s in HF_SOS]
        self.env_zi = [np.zeros((len(ENV_SOS), 2)) for _ in HF_SOS]
        self.lp = butter(2, 800, fs=FS, output="sos")
        self.lp_zi = np.zeros((len(self.lp), 2))
        self.pll = StreamingPLL()

    def push(self, raw):
        raw = np.asarray(raw, dtype=float)
        if not len(raw):
            return np.empty(0), np.empty((2, 0)), np.empty(0)
        powers = []
        for j, sos in enumerate(HF_SOS):
            hf, self.hf_zi[j] = sosfilt(sos, raw, zi=self.hf_zi[j])
            power, self.env_zi[j] = sosfilt(ENV_SOS, hf * hf, zi=self.env_zi[j])
            powers.append(power)
        channels = np.stack([raw, *powers])
        n_history = self.history.shape[1]
        start = self.raw_index - n_history
        combined = np.concatenate([self.history, channels], axis=1)
        # Prefix aligns local polyphase output to global sample indices0,40,80...
        prefix = start % 40
        padded = np.pad(combined, ((0, 0), (prefix, 0)))
        filtered = upfirdn(FIR, padded, down=40, axis=1)
        local = np.arange(filtered.shape[1]) * 40 - prefix
        use = (local >= n_history) & (local < combined.shape[1])
        decimated = filtered[:, use]
        self.history = combined[:, -800:].copy()
        self.raw_index += len(raw)
        low = decimated[0]
        env = np.maximum(decimated[1:], 0)
        if not len(low):
            return low, env, np.empty(0)
        smooth, self.lp_zi = sosfilt(self.lp, low, zi=self.lp_zi)
        phase = np.empty(len(low))
        for j, u in enumerate(smooth):
            idx = self.low_index + j
            if idx % 10 == 0:
                self.pll.push(u)
            phase[j] = self.pll.phase + (idx % 10) / FS * 2 * np.pi * self.pll.freq
        self.low_index += len(low)
        return low, env, phase

    def state_bytes(self):
        return int(
            self.history.nbytes
            + sum(a.nbytes for a in self.hf_zi + self.env_zi)
            + self.lp_zi.nbytes
            + 8 * len(self.pll.initial)
        )
