"""Actual causal rational decimation and native-band limit checks."""
import sys
from pathlib import Path
import numpy as np
from scipy.signal import upfirdn,freqz
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from prepare_self462_12800_signals_20261105 import filters

def decimate(x):
 a,b=filters();low50=upfirdn(a,x,down=40)[:(len(x)+39)//40]
 return upfirdn(b,low50,up=32,down=125)[:(len(low50)*32+124)//125]

def test_antialias_response_and_tone_suppression():
 _,b=filters();f,h=freqz(b/32,worN=65536,fs=1600000);db=20*np.log10(np.maximum(abs(h),1e-30))
 assert np.ptp(db[f<=5000])<.01 and db[f>=6400].max()<-75
 t=np.arange(400000)/2000000;passed=decimate(np.sin(2*np.pi*1000*t));rejected=decimate(np.sin(2*np.pi*8000*t))
 rms=lambda x:np.sqrt(np.mean(x[500:]**2))
 assert abs(rms(passed)-1/np.sqrt(2))<.005
 assert rms(rejected)/rms(passed)<1e-4

def test_future_suffix_cannot_change_past_decimated_samples():
 rng=np.random.default_rng(42);x=rng.normal(size=200000);cut=100000;y=x.copy();y[cut:]*=10
 a,b=decimate(x),decimate(y);n=int(cut/2000000*12800)
 np.testing.assert_array_equal(a[:n],b[:n])

def test_one_nominal_cycle_has256_input_samples():
 x=np.ones(200000);y=decimate(x)
 assert len(y)==1280 and 12800//50==256 and (12800//50)//64==4
 np.testing.assert_allclose(y[500:],1,atol=3e-4)
