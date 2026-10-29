"""Experimental reference-assisted correction contracts.

The stable detector score is the fallback. These utilities neither run a PLL nor
fit a gate, use labels, certify physical phase, or change existing study paths.
A training-only calibrated head/gate is required before performance evaluation.
"""
from __future__ import annotations
import numpy as np

def comparable_reference_pairs(valid, segments, epochs, origins, current, previous):
    """Check reference-local coordinate continuity for a set of past pairs.

    Negative reference IDs, invalid samples, different epochs/segments/origins,
    and non-past pairs cannot support a correction. Validity is an estimator
    check, not a physical phase-error certificate.
    """
    valid=np.asarray(valid,dtype=bool);segments=np.asarray(segments);epochs=np.asarray(epochs);origins=np.asarray(origins)
    current=np.asarray(current,dtype=int);previous=np.asarray(previous,dtype=int)
    if any(x.shape!=valid.shape for x in [segments,epochs,origins]) or valid.ndim!=1:
        raise ValueError('Reference metadata must be aligned one-dimensional arrays')
    if current.shape!=previous.shape or current.ndim!=1:
        raise ValueError('Pair index arrays must align')
    if np.any(current<0) or np.any(previous<0) or np.any(current>=len(valid)) or np.any(previous>=len(valid)):
        raise ValueError('Pair indices out of range; missing history must be explicitly masked')
    return (valid[current]&valid[previous]&(previous<current)&
            (segments[current]>=0)&(epochs[current]>=0)&
            (segments[current]==segments[previous])&
            (epochs[current]==epochs[previous])&(origins[current]==origins[previous]))

def relative_clock_innovation(primary_phase, auxiliary_phase, current, previous, comparable):
    """Wrapped change in auxiliary-minus-primary phase, in radians.

    A constant offset between two reference origins cancels. Clock disagreement
    is a diagnostic; it does not identify which estimator is wrong.
    """
    primary=np.asarray(primary_phase,dtype=float);aux=np.asarray(auxiliary_phase,dtype=float)
    current=np.asarray(current,dtype=int);previous=np.asarray(previous,dtype=int);ok=np.asarray(comparable,dtype=bool)
    if primary.ndim!=1 or primary.shape!=aux.shape or ok.shape!=current.shape or previous.shape!=current.shape:
        raise ValueError('Aligned clocks and pair masks required')
    if np.any(current<0) or np.any(previous<0) or np.any(current>=len(primary)) or np.any(previous>=len(primary)):
        raise ValueError('Clock pair index out of range')
    out=np.zeros(current.shape,dtype=float);ok=ok&np.isfinite(primary[current])&np.isfinite(primary[previous])&np.isfinite(aux[current])&np.isfinite(aux[previous])
    error=(aux[current[ok]]-primary[current[ok]])-(aux[previous[ok]]-primary[previous[ok]])
    out[ok]=np.angle(np.exp(1j*error))
    return out,ok

def fuse_reference_increment(stable_logit, correction_logit, eligible, gate, strength=0.):
    """Inject a bounded-gate auxiliary increment; retain exact stable fallback.

    The correction and gate must be fit without outer-test information. No
    classification improvement is guaranteed on eligible samples. Default
    strength=0 keeps this experimental feature disabled.
    """
    stable=np.asarray(stable_logit);corr=np.asarray(correction_logit,dtype=float);ok=np.asarray(eligible,dtype=bool);w=np.asarray(gate,dtype=float)
    if stable.shape!=corr.shape or stable.shape!=ok.shape or stable.shape!=w.shape:
        raise ValueError('Aligned score, eligibility and gate shapes required')
    if not np.isfinite(stable).all() or not np.isfinite(strength) or strength<0:
        raise ValueError('Stable scores and nonnegative strength must be finite')
    if np.any(~np.isfinite(w)) or np.any((w<0)|(w>1)):
        raise ValueError('Gate values must be finite and between zero and one')
    active=ok&(w>0)&(strength>0)
    if not np.isfinite(corr[active]).all():raise ValueError('Active corrections must be finite')
    out=stable.copy()
    if not np.issubdtype(out.dtype,np.floating):out=out.astype(float)
    out[active]=stable[active]+strength*w[active]*corr[active]
    return out
