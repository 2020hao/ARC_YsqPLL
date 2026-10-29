"""Public reference facade; original SOGI/RCR/CFT numerical kernels stay unchanged."""

from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from scipy.signal import firwin, upfirdn, butter, sosfilt
from .constants import FIR
from .pll import robust_pll
from .rcr_reference import _implementation


@dataclass(frozen=True)
class CurrentReference:
    low_current: np.ndarray
    clock: pd.DataFrame
    method: str
    events: object


def current_reference(raw, sample_rate_hz, method="rcr"):
    if sample_rate_hz not in (1_000_000, 2_000_000) or method not in (
        "fixed20",
        "sogi",
        "rcr",
        "cft",
    ):
        raise ValueError("Unknown reference or native sampling rate")
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("One finite current channel required")
    down = sample_rate_hz // 50_000
    taps = (
        FIR
        if sample_rate_hz == 2_000_000
        else firwin(401, 20_000, fs=sample_rate_hz, window=("kaiser", 7.5))
    )
    low = upfirdn(taps, raw, down=down)[: (len(raw) + down - 1) // down]
    if method == "fixed20":
        return CurrentReference(low, pd.DataFrame(), method, {})
    ref = sosfilt(butter(2, 800, fs=50_000, output="sos"), low)[::10]
    times = np.arange(len(ref)) / 5000
    if method == "sogi":
        if len(ref) < 3:
            raise ValueError("Insufficient reference samples")
        phase, freq = robust_pll(ref, fs=5000)
        valid = np.isfinite(phase)
        clock = pd.DataFrame(
            dict(
                estimate_time_s=times,
                phase_rad=phase,
                frequency_hz=freq,
                phase_valid=valid,
                reference_segment=np.where(valid, 0, -1),
                reference_epoch=0,
                origin="fundamental",
            )
        )
        return CurrentReference(low, clock, method, {})
    original, path = _implementation()
    spec = json.loads(path.read_text())
    spec["reference_fs_hz"] = 5000
    if method == "rcr":
        clock, events = original(times, ref, spec)
    else:
        from .rcr_fundamental_consensus import estimate

        clock, events = estimate(times, ref, spec, "full")
    return CurrentReference(low, clock, method, events)
