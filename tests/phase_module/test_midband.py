import numpy as np
import torch
from arc_model.implementations.map_input_calibration.midband import MidbandFilter,kernels,profile,assemble
from arc_model.implementations.map_input_calibration.midband_network import MidbandPhaseMap
from arc_model.implementations.map_input_calibration.deep_network import DeepCodedPhaseMap


def test_competition_patterns_preserved_under_uniform_positive_gain():
    x=np.random.RandomState(42).normal(size=4096)
    first=MidbandFilter().update(x);scaled=MidbandFilter().update(7*x)
    np.testing.assert_array_equal(first[1:],scaled[1:])
    np.testing.assert_allclose(scaled[0],49*first[0],rtol=1e-11,atol=1e-11)


def test_causal_filter_chunk_parity_and_no_future_influence():
    x=np.random.RandomState(3).normal(size=2400)
    full=MidbandFilter().update(x);a=MidbandFilter();parts=np.concatenate([a.update(v) for v in np.array_split(x,24)],axis=1)
    np.testing.assert_allclose(full[0],parts[0],rtol=0,atol=1e-12);np.testing.assert_array_equal(full[1:],parts[1:])
    changed=x.copy();changed[1200:]*=100
    np.testing.assert_array_equal(MidbandFilter().update(changed)[:,:1200],full[:,:1200])
    w=kernels();np.testing.assert_allclose(w.sum(-1),0,atol=1e-15);np.testing.assert_allclose(abs(w).sum(-1),1)


def test_profile_integration_and_raw_channels_and_history_reset():
    v=np.stack([np.ones(2000)*4,np.ones(2000),np.zeros(2000)])
    prefix=np.c_[np.zeros(3),np.cumsum(v,axis=1)]/50000
    edges=np.linspace(100,40100,65,dtype='int64');result=profile(prefix,0,edges,2000000)
    np.testing.assert_allclose(result[0],np.log10(4),atol=1e-6);np.testing.assert_allclose(result[1],1,atol=1e-6)
    x=np.random.RandomState(4).normal(size=(2,9,5,64)).astype('float32');e=np.random.RandomState(5).normal(size=(2,9,3,64)).astype('float32');m=np.ones((2,9));m[0,:4]=0
    out=assemble(x,e,m);np.testing.assert_array_equal(out[:,:,:5],x)
    assert not out[0,:5,6].any();assert out.shape==(2,9,9,64)


def test_added_channels_zero_initialized_reference_equivalence_and_budgets():
    x=torch.randn(3,9,9,64);m=torch.ones(3,9)
    torch.manual_seed(27);old=DeepCodedPhaseMap('raw').eval()
    with torch.no_grad():expected=old(x,m)
    for arm,n in [('base',7097),('mid',7257),('competition',7257),('both',7417)]:
        torch.manual_seed(27);net=MidbandPhaseMap(arm).eval()
        with torch.no_grad():torch.testing.assert_close(net(x,m),expected,atol=3e-7,rtol=2e-6)
        assert sum(v.numel() for v in net.parameters())==n
        net.train();y,activity=net(x,m,return_activity=True);(y.square().mean()+.001*activity).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
