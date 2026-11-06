import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from real_arc_reference_evidence_20261031 import independent_proxy,probabilities,dynamics,clock_at


def test_positive_sine_proxy_and_static_rotation():
    t=np.arange(5000)/5000;phase=2*np.pi*50*t+.7;x=np.sin(phase)+.2*np.sin(3*phase+.2)
    ref=independent_proxy(x);mask=ref['valid']
    assert mask.sum()>4300
    error=np.rad2deg(np.angle(np.exp(1j*(ref['phase']-phase))))
    assert abs(error[mask]).max()<1e-9
    e=np.ones((5000,2));start=np.arange(500,4400,100)
    p,ok,cover,epoch=probabilities(e,phase,np.ones(5000,bool),np.zeros(5000,int),np.zeros(5000,int),start)
    prev=np.r_[-1,np.arange(len(start)-1)]
    m,j=dynamics(p,ok,prev,epoch);mr,jr=dynamics(np.roll(p,1,-1),ok,prev,epoch)
    np.testing.assert_allclose(m,mr,atol=1e-14,equal_nan=True);np.testing.assert_allclose(j,jr,atol=1e-14,equal_nan=True)
    np.testing.assert_allclose(cover,1.)


def test_invalid_and_segment_crossing_not_comparable():
    t=np.arange(500)/5000;phase=2*np.pi*50*t
    valid=np.ones(500,bool);valid[230]=False
    seg=np.zeros(500,int);seg[350:]=1
    p,ok,_,epoch=probabilities(np.ones((500,2)),phase,valid,np.zeros(500,int),seg,np.array([100,200,300]))
    np.testing.assert_array_equal(ok,[True,False,False])
    m,_=dynamics(p,ok,np.array([-1,0,1]),epoch)
    assert np.isnan(m).all()
