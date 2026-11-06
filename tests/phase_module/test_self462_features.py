"""Physical invariances and causal phase-bin support for the new dataset study."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/arc_model/preprocessing'))
spec=importlib.util.spec_from_file_location('pll_standalone',ROOT/'src/arc_model/preprocessing/pll.py')
# The data-host standalone module differs only in the constant import.
import types
module=types.ModuleType('pll_standalone');exec((ROOT/'src/arc_model/preprocessing/pll.py').read_text().replace('from .constants import FS_PLL','FS_PLL=5000'),module.__dict__);sys.modules['pll_standalone']=module
spec=importlib.util.spec_from_file_location('self462_extract',ROOT/'scripts/self462_extract_20261105.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_features_positive_gain_invariance():
 t=np.arange(2000)/50000;low=np.sin(2*np.pi*50*t)+.1*np.sin(2*np.pi*150*t);sh=np.sin((np.arange(64)+.5)*2*np.pi/64)
 power=np.vstack([np.ones(64),1+.5*sh,2+.3*sh]);fourth=power[1:]**2*3
 x,b,p,prev=m.describe(sh,power,fourth,low,0,.02)
 xx,bb,pp,_=m.describe(2*sh,4*power,16*fourth,2*low,0,.02)
 invariant=[i for i,name in enumerate(m.FEATURE_NAMES) if name not in ['log_rms','log_mid_power','log_lowHF_power','log_highHF_power']]
 np.testing.assert_allclose(x[invariant],xx[invariant],atol=1e-6);np.testing.assert_allclose(p,pp,atol=1e-6)
 assert len(x)==40 and b.shape==(5,64) and p.shape==(4,2,64)

def test_absolute_delta_avoids_signed_cancellation():
 t=np.arange(1000)/50000;low=np.sin(2*np.pi*50*t);sh=np.sin((np.arange(64)+.5)*2*np.pi/64);power=np.vstack([np.ones(64),1+.5*sh,2+.3*sh]);fourth=power[1:]**2*3
 _,_,_,prev=m.describe(sh,power,fourth,low,0,.02)
 _,b,p,_=m.describe(sh,power[:,::-1],fourth[:,::-1],low,0,.02,prev)
 assert np.max(abs(b[3:].sum(1)))<1e-6
 assert (p[1].mean(1)>0).all()

def test_completed_native_grids_are_immutable_after_future_append():
 t=np.arange(100)/500;ph=2*np.pi*(50*t+.02*t*t);valid=np.ones(len(t),bool);seg=np.zeros(len(t),int)
 old=m.candidates(t[:50],ph[:50],valid[:50],seg[:50],seg[:50],400000,2000000)
 new=m.candidates(t,ph,valid,seg,seg,400000,2000000)
 old=[e for e in old if e['native']];new=[e for e in new if e['native'] and e['av']<=t[49]+1e-12]
 assert len(old)==len(new)>0
 for a,b in zip(old,new):np.testing.assert_array_equal(a['edges'],b['edges']);assert a['b']/2000000<=a['av']+1e-9
