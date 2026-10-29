"""Standalone RCR front end on native current; no SOGI computation required."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import importlib,json,sys
import numpy as np
import pandas as pd
from scipy.signal import upfirdn,firwin,sosfilt,butter
from .constants import FIR

@dataclass
class RCRReference:
    low_current: np.ndarray
    reference_current: np.ndarray
    clock: pd.DataFrame
    events: list
    low_sample_rate_hz: int = 50000
    reference_sample_rate_hz: int = 5000

@lru_cache(maxsize=1)
def _implementation():
    vendor=Path(__file__).resolve().parents[3]/'third_party/current_reference_recovery_v2'
    sys.path.insert(0,str(vendor))
    try:module=importlib.import_module('reference_recovery_recommended')
    finally:sys.path.remove(str(vendor))
    if Path(module.__file__).resolve().parent!=vendor.resolve():
        raise ImportError('RCR implementation path differs from registered vendor source')
    return module.estimate,vendor/'pll_spec.json'

def estimate_raw_current_reference(raw,sample_rate_hz=2000000):
    """Causal FIR20kHz→50kHz, LP800Hz→5kHz, then registered RCR.

    Matches the CPV2 recovery branch. The clock explicitly records startup,
    validity, reference segments and epochs; callers retain native-rate current
    for their independent high-frequency feature path.
    """
    values=np.asarray(raw,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all():
        raise ValueError('RCR requires one finite current channel')
    if sample_rate_hz not in [1000000,2000000]:
        raise ValueError('Registered raw-current rates are 1MHz and 2MHz')
    if len(values)<sample_rate_hz//50:
        raise ValueError('At least one nominal cycle is required; RCR startup remains explicit')
    down=sample_rate_hz//50000
    taps=FIR if sample_rate_hz==2000000 else firwin(401,20000,fs=sample_rate_hz,window=('kaiser',7.5))
    low=upfirdn(taps,values,down=down)[:(len(values)+down-1)//down]
    reference=sosfilt(butter(2,800,fs=50000,output='sos'),low)[::10]
    estimate,path=_implementation();spec=json.loads(path.read_text());spec['reference_fs_hz']=5000
    clock,events=estimate(np.arange(len(reference))/5000,reference,spec)
    return RCRReference(low,reference,clock,events)
