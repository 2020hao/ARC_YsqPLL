"""Causal native-rate HF burst concentration and within-band power allocation.

Four extra phase profiles, no labels, learned thresholds or future cycles.
The original five channels are an immutable input to ``assemble``.
"""
import numpy as np
from scipy.signal import butter, sosfilt


FEATURE_NAMES = (
    "log10_burst_concentration_10_100k",
    "log10_burst_concentration_100_400k",
    "power_fraction_10_40_of_10_100k",
    "power_fraction_100_200_of_100_400k",
)
SUBBANDS_HZ = ((10000, 40000), (40000, 100000),
               (100000, 200000), (200000, 400000))


class DetailFilter:
    """Four persistent SOS bandpasses; output six moment-density sequences.

For a subband pair y1,y2, q=y1**2+y2**2. Export q,q**2,y1**2.
This common denominator makes the subband fraction bounded and consistent.
It is a filter-bank allocation, not an exact rectangular FFT-band integral.
"""
    def __init__(self, sample_rate_hz):
        if sample_rate_hz not in (1000000, 2000000):
            raise ValueError("Native 1/2 MHz current required")
        self.sample_rate_hz = sample_rate_hz
        self.sos = [butter(3, b, fs=sample_rate_hz, btype="bandpass", output="sos")
                    for b in SUBBANDS_HZ]
        self.state = [np.zeros((len(s), 2)) for s in self.sos]

    def update(self, current):
        x = np.asarray(current, dtype=np.float64)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError("Finite 1D native current required")
        if not len(x):
            return np.empty((6, 0), dtype=np.float64)
        bands = []
        for i, sos in enumerate(self.sos):
            y, self.state[i] = sosfilt(sos, x, zi=self.state[i])
            bands.append(y * y)
        lo, hi = bands[0] + bands[1], bands[2] + bands[3]
        return np.stack((lo, lo * lo, hi, hi * hi, bands[0], bands[2]))


def phase_profile(prefix, edges, *, sample_offset=0):
    """Native integer bin edges -> four dimensionless profiles.

prefix is an unscaled exclusive cumulative sum with a leading zero column.
log10(E[q**2]/E[q]**2) measures within-bin concentration of HF activity.
Silent bins map to zero. This is not the cross-cycle MAD or an arc label.
"""
    prefix = np.asarray(prefix)
    edges = np.asarray(edges)
    if prefix.ndim != 2 or prefix.shape[0] != 6:
        raise ValueError("Six moment prefix rows required")
    if edges.shape != (65,) or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("65 native integer phase edges required")
    ix = edges - sample_offset
    if ix[0] < 0 or ix[-1] >= prefix.shape[1] or (np.diff(ix) <= 0).any():
        raise ValueError("Unavailable or nonpositive phase support")
    moment = np.maximum(np.diff(prefix[:, ix], axis=1) / np.diff(ix), 0.)
    return _moment_profile(moment)


def profile_from_values(values, edges, *, sample_offset=0):
    """Sum within each bin, avoiding strong-earlier/weak-later prefix cancellation.

    values contains the six nonnegative instantaneous moments from DetailFilter.
    Only samples in [edges[0],edges[-1]) are read; no future support is used.
    """
    values = np.asarray(values)
    edges = np.asarray(edges)
    if values.ndim != 2 or values.shape[0] != 6 or edges.shape != (65,):
        raise ValueError("Six moment rows and 65 edges required")
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("Native integer edges required")
    ix = edges - sample_offset
    if ix[0] < 0 or ix[-1] > values.shape[1] or (np.diff(ix) <= 0).any():
        raise ValueError("Unavailable or nonpositive phase support")
    within = values[:, ix[0]:ix[-1]]
    moment = np.add.reduceat(within, ix[:-1]-ix[0], axis=1) / np.diff(ix)
    return _moment_profile(moment)


def _moment_profile(moment):
    result = np.zeros((4, 64), dtype=np.float64)
    for band, (mean_row, fourth_row, part_row) in enumerate(((0, 1, 4), (2, 3, 5))):
        active = moment[mean_row] > 1e-30
        concentration = np.ones(64)
        concentration[active] = moment[fourth_row, active] / moment[mean_row, active] ** 2
        result[band] = np.log10(np.maximum(concentration, 1.))
        result[band + 2, active] = np.clip(
            moment[part_row, active] / moment[mean_row, active], 0., 1.)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite detail profile")
    return result.astype("float32")


def assemble(base, extra, mask):
    """Append four channels; preserve the original five bit for bit."""
    base = np.asarray(base, dtype="float32")
    extra = np.asarray(extra, dtype="float32")
    mask = np.asarray(mask)
    if base.shape[:2] != extra.shape[:2] or extra.shape[2:] != (4, 64):
        raise ValueError("Aligned [N,9,4,64] detail histories required")
    if base.shape[1] != 9 or base.shape[-1] != 64 or base.shape[2] < 5:
        raise ValueError("Original nine-cycle phase maps required")
    if mask.shape != base.shape[:2] or not np.isin(mask, (0, 1)).all():
        raise ValueError("Binary physical-history mask required")
    return np.concatenate((base[:, :, :5], extra * mask[:, :, None, None]), axis=2)
