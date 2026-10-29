"""Causal completed windows; native HF integration stays separate from PLL current."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.signal import firwin, upfirdn, butter, sosfilt
from .phase_reference import current_reference
from .causal_window_selection import parallel_candidates, select_available
from ..methods.phase_representation import BANDS_HZ


@dataclass(frozen=True)
class PhaseWindows:
    wave: np.ndarray
    native: np.ndarray
    energy: np.ndarray
    metadata: pd.DataFrame
    clock: pd.DataFrame


def prepare_phase_windows(raw, sample_rate_hz, *, reference="rcr", anchors=None, phase_bins=20):
    """Anchors are seconds from record start. No labels are accepted or generated.

    Native FIR112.5kHz preserves the existing local ArcFENet input policy; its
    attenuation near112kHz is documented, not claimed author-equivalent. Fixed20
    is a true nominal-coordinate control, not a successful PLL observation.
    """
    if phase_bins < 4 or phase_bins % 2 or 1000 % phase_bins:
        raise ValueError("Phase bins must be even and divide1000")
    raw = np.asarray(raw, float)
    result = current_reference(raw, sample_rate_hz, reference)
    clock = result.clock
    if anchors is None:
        anchors = np.arange(1, len(raw) // (sample_rate_hz // 50) + 1) / 50
    anchors = np.asarray(anchors, float)
    if (
        anchors.ndim != 1
        or not len(anchors)
        or not np.isfinite(anchors).all()
        or (np.diff(anchors) <= 0).any()
    ):
        raise ValueError("Nonempty increasing finite anchors required")
    if anchors[0] < 0.02 - 1e-10 or anchors[-1] > len(raw) / sample_rate_hz + 1e-10:
        raise ValueError("Anchor outside completed recording support")
    segments = {}
    if reference == "fixed20":
        good = pd.DataFrame()
    else:
        from .rcr_reference import _implementation

        _implementation()
        from reference_recovery_v2 import complete_cycles

        good = complete_cycles(
            clock, sample_rate_hz, dict(minimum_period_s=0.018, maximum_period_s=0.022)
        )
        segments = {int(k): v for k, v in clock[clock.phase_valid].groupby("reference_segment")}
    ledger = parallel_candidates(good, len(raw), sample_rate_hz)
    entries, grids = [], []
    for e in ledger:
        a, b = e["start_sample"], e["stop_sample_exclusive"]
        start, end = a / sample_rate_hz, b / sample_rate_hz
        if e["kind"] == "pll":
            c = segments[e["segment"]]
            tx, px = c.estimate_time_s.to_numpy(), c.phase_rad.to_numpy()
            grid = np.interp(
                np.linspace(np.interp(start, tx, px), np.interp(end, tx, px), 1001), px, tx
            )
            ci = min(np.searchsorted(tx, end), len(tx) - 1)
            available = float(tx[ci])
            origin = str(c.origin.iloc[0])
        else:
            grid = np.linspace(start, end, 1001)
            available, origin = end, "nominal"
        grids.append(grid)
        entries.append(
            dict(
                start_seconds=start,
                end_seconds=end,
                available_seconds=available,
                kind=e["kind"],
                segment=e["segment"],
                epoch=e["reference_epoch"],
                origin=origin,
            )
        )
    if reference == "fixed20":
        valid = np.zeros(len(anchors), bool)
        segment = epoch = np.full(len(anchors), -1)
    else:
        ix = np.maximum(
            np.searchsorted(clock.estimate_time_s.to_numpy(), anchors + 1e-10, side="right") - 1, 0
        )
        valid = clock.phase_valid.to_numpy()[ix]
        segment, epoch = (
            clock.reference_segment.to_numpy()[ix],
            clock.reference_epoch.to_numpy()[ix],
        )
    selected = select_available(entries, anchors, valid, segment, epoch)
    down = sample_rate_hz // 250_000
    taps = firwin(
        801 if sample_rate_hz == 2_000_000 else 401,
        112_500,
        fs=sample_rate_hz,
        window=("kaiser", 7.5),
    )
    native = upfirdn(taps, raw, down=down)[: (len(raw) + down - 1) // down]
    prefixes = []
    for band in BANDS_HZ:
        filtered = sosfilt(butter(3, band, fs=sample_rate_hz, btype="bandpass", output="sos"), raw)
        prefixes.append(np.r_[0.0, np.cumsum(filtered**2)] / sample_rate_hz)
    waves, high, energies, rows = [], [], [], []
    for anchor, i in zip(anchors, selected):
        e, grid = entries[i], grids[i]
        nt = np.interp(np.arange(5000) / 5000, np.arange(1001) / 1000, grid)
        support = max(
            np.ceil(grid[-2] * 50_000) / 50_000,
            np.ceil(nt[-1] * 250_000) / 250_000,
            e["end_seconds"],
            e["available_seconds"],
        )
        if support > anchor + 1e-9:
            raise ValueError("Interpolation or reference observation needs a future sample")
        edges = np.ceil(grid[:: 1000 // phase_bins] * sample_rate_hz).astype(int)
        edges = np.clip(
            edges,
            round(e["start_seconds"] * sample_rate_hz),
            round(e["end_seconds"] * sample_rate_hz),
        )
        waves.append(
            np.interp(grid[:-1] * 50_000, np.arange(len(result.low_current)), result.low_current)
        )
        high.append(np.interp(nt * 250_000, np.arange(len(native)), native))
        energies.append([np.maximum(np.diff(p[edges]), 0.0) for p in prefixes])
        rows.append(dict(**e, anchor_seconds=float(anchor), phase_valid=e["kind"] == "pll"))
    return PhaseWindows(
        np.asarray(waves, np.float32),
        np.asarray(high, np.float32),
        np.asarray(energies, np.float64),
        pd.DataFrame(rows),
        clock,
    )
