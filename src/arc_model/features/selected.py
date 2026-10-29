"""Compute selected energy, phase-bin and harmonic descriptors without a 42D vector.

Indices follow schema.NAMES. Only the feature families selected by the frozen
top-two study are implemented here; unsupported selections fail explicitly.
"""
import numpy as np


def required_bands(indices):
    bands = set()
    for j in indices:
        if j in (4, 5):
            bands.add(j - 4)
        elif 6 <= j < 22:
            bands.add((j - 6) // 8)
        elif not 26 <= j < 34:
            raise ValueError(f'Feature {j} is not implemented in the selected extractor')
    return tuple(sorted(bands))


def extract(waves, envelopes, indices, band_ids=(0, 1)):
    """Return exactly len(indices) columns, supporting batched phase cycles.

    envelopes has axes (..., supplied_band, sample). Its values are smoothed
    band power, not signal amplitude. No training/reference statistics are fit.
    """
    indices = tuple(indices)
    if not indices or len(indices) != len(set(indices)):
        raise ValueError('A nonempty unique feature index list is required')
    needed = required_bands(indices)
    if any(b not in band_ids for b in needed):
        raise ValueError('A required band envelope is missing')
    x = np.asarray(waves, dtype=np.float64)
    e = np.asarray(envelopes, dtype=np.float64)
    if e.shape != x.shape[:-1] + (len(band_ids), x.shape[-1]):
        raise ValueError('Envelope and waveform dimensions do not match')
    if not np.isfinite(x).all() or not np.isfinite(e).all() or (e < 0).any():
        raise ValueError('Inputs must be finite, with nonnegative band power')
    rms2 = np.mean(x*x, axis=-1) + 1e-12
    means = {b: e[..., band_ids.index(b), :].mean(-1) for b in needed}
    harmonic = [j-24 for j in indices if 26 <= j < 34]
    spectral = None
    if harmonic:
        z = x / np.sqrt(rms2)[..., None]
        spectral = np.abs(np.fft.rfft(z, axis=-1))**2
    result = []
    for j in indices:
        if j in (4, 5):
            value = np.log10(means[j-4]/rms2 + 1e-20)
        elif 6 <= j < 22:
            band, k = divmod(j-6, 8)
            part = np.array_split(e[..., band_ids.index(band), :], 8, axis=-1)[k]
            value = np.log1p(part.mean(-1)/(means[band]+1e-20))
        else:
            value = np.log1p(np.sqrt(spectral[..., j-24]/(spectral[..., 1]+1e-12)))
        result.append(value)
    return np.stack(result, axis=-1).astype(np.float32)
