"""Independent fixed-duration phase-start crop; no waveform interpolation.

The caller supplies the model's original causal uniformly filtered stream.
For HYDRA that stream is formed by rectification BEFORE filtering/downsampling.
Only the reference boundary is estimated; current samples are direct slices.
"""
import numpy as np

SAMPLE_RATE = 250000
POINTS = 5000


def crop(signal, start_seconds, anchor_seconds, *, origin_seconds=0.0,
         interval_start=-np.inf, interval_end=np.inf, guard_seconds=0.04):
    """Closest causal start sample; return exact samples and support receipt."""
    start = int(np.rint((start_seconds-origin_seconds)*SAMPLE_RATE))
    latest = int(np.floor((anchor_seconds-origin_seconds-POINTS/SAMPLE_RATE)*SAMPLE_RATE+1e-7))
    start = min(start, latest)
    end = start+POINTS
    a = origin_seconds+start/SAMPLE_RATE
    b = origin_seconds+end/SAMPLE_RATE
    if start < 0 or end > len(signal) or b > anchor_seconds+1e-10:
        raise ValueError("incomplete or future support")
    if a < interval_start+guard_seconds-1e-10 or b > interval_end-guard_seconds+1e-10:
        raise ValueError("support outside frozen label guard")
    if abs(a-start_seconds) > 1/SAMPLE_RATE+1e-9:
        raise ValueError("more than one sample away from selected phase boundary")
    wave = np.asarray(signal[start:end],dtype=np.float32)
    if wave.shape != (POINTS,) or not np.isfinite(wave).all():
        raise ValueError("nonfinite or incorrect-length crop")
    return wave, dict(start_sample=start,end_sample_exclusive=end,
                     start_seconds=a,end_seconds=b,reference_boundary_seconds=start_seconds,
                     quantization_error_seconds=a-start_seconds,
                     sample_spacing_seconds=1/SAMPLE_RATE,waveform_interpolation=False)

