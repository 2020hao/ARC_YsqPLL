"""Read one current channel, validate native rate and preserve source timing."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import struct
import numpy as np


@dataclass(frozen=True)
class CurrentRecording:
    current: np.ndarray
    sample_rate_hz: int
    time_origin_s: float
    source: Path


def read_current(path, sample_rate_hz):
    path = Path(path).resolve()
    if sample_rate_hz not in (1_000_000, 2_000_000):
        raise ValueError("Native 1MHz or 2MHz recording required")
    t0 = 0.0
    if path.suffix.lower() == ".npy":
        x = np.load(path, mmap_mode="r", allow_pickle=False)
        if x.ndim != 1:
            raise ValueError("NPY must contain one native-rate current channel")
    elif path.suffix.lower() == ".dat":
        if not path.name.startswith("C3_"):
            raise ValueError("Only C3 current is a detector input; C2 is annotation-only")
        a = np.loadtxt(path, ndmin=2)
        if a.shape[1] != 2 or len(a) < 2:
            raise ValueError("DAT requires time,current columns")
        if not np.allclose(np.diff(a[:, 0]), 1 / sample_rate_hz, atol=2e-9, rtol=0):
            raise ValueError("DAT timestamp spacing differs from declared native rate")
        t0 = float(a[0, 0])
        x = a[:, 1]
    elif path.suffix.lower() == ".bin":
        with path.open("rb") as f:
            header = f.read(56)
        if len(header) != 56:
            raise ValueError("Truncated NET9770 header")
        device, _, channels, actual_fs, samples = struct.unpack("<32sQIIQ", header)
        if device.rstrip(b"\0") != b"NET9770_114" or channels != 8:
            raise ValueError("Only NET9770_114 CH0 total-current records are supported")
        if actual_fs != sample_rate_hz or path.stat().st_size != 56 + samples * channels * 8:
            raise ValueError("Native sampling rate or BIN payload size mismatch")
        x = np.memmap(path, dtype="<f8", mode="r", offset=56, shape=(samples, channels))[:, 0]
    else:
        raise ValueError("Supported formats: native 1D NPY, C3 DAT, NET9770_114 BIN")
    if len(x) < sample_rate_hz // 50 or not np.isfinite(x).all():
        raise ValueError("Finite current with at least one nominal cycle required")
    return CurrentRecording(x, sample_rate_hz, t0, path)
