"""Optional two-channel native-HF burst input, separate from the frozen default.

The primary model remains one CNN--LIF classifier. These channels encode
within-phase-bin fourth/second moments; they are not a score-fusion branch.
"""
import numpy as np
from .midband_network import MidbandPhaseMap

INPUT_NAMES=('low_frequency_shape','log_power_10_100k','log_power_100_400k',
             'signed_distribution_change_10_100k','signed_distribution_change_100_400k',
             'log_burst_10_100k','log_burst_100_400k')


def burst_profiles(mean_squared_current,mean_fourth_current):
    """Both arrays end in[2bands,64bins], measured from native filtered current."""
    power=np.asarray(mean_squared_current,dtype=np.float64)
    fourth=np.asarray(mean_fourth_current,dtype=np.float64)
    if power.shape!=fourth.shape or power.shape[-2:]!=(2,64):raise ValueError('Two bands and64bins required')
    if not np.isfinite(power).all() or not np.isfinite(fourth).all() or (power<0).any() or (fourth<0).any():raise ValueError('Finite nonnegative moments required')
    return np.log10(np.maximum(fourth/np.maximum(power,1e-20)**2,1)).astype('float32')


def append_burst(original5,burst2):
    base=np.asarray(original5,dtype='float32');burst=np.asarray(burst2,dtype='float32')
    if base.shape[-2:]!=(5,64) or burst.shape!=base.shape[:-2]+(2,64):raise ValueError('Input shapes do not match')
    return np.concatenate([base,burst],axis=-2)


class Self462BurstMap(MidbandPhaseMap):
    """7257parameters; exact architecture/state-dict compatibility with the study."""
    def __init__(self):super().__init__('mid')
