import numpy as np
import pytest
import torch
from arc_model.implementations.map_input_calibration.detail_features import DetailFilter, phase_profile, profile_from_values, assemble
from arc_model.implementations.map_input_calibration.detail_network import DetailPhaseMap
from arc_model.implementations.map_input_calibration.deep_network import DeepCodedPhaseMap


def prefix(x):
    return np.c_[np.zeros((6, 1)), np.cumsum(x, axis=1)]


@pytest.mark.parametrize("fs", [1000000, 2000000])
def test_causal_chunk_parity_and_gain_invariance(fs):
    x = np.random.RandomState(7).normal(size=8192)
    whole = DetailFilter(fs).update(x)
    f = DetailFilter(fs)
    chunked = np.concatenate([f.update(a) for a in np.array_split(x, 17)], axis=1)
    np.testing.assert_array_equal(whole, chunked)
    changed = x.copy(); changed[4096:] *= 3
    np.testing.assert_array_equal(whole[:, :4096], DetailFilter(fs).update(changed)[:, :4096])
    edges = np.linspace(1024, 8192, 65, dtype="int64")
    a = phase_profile(prefix(whole), edges)
    np.testing.assert_allclose(a, profile_from_values(whole,edges), atol=1e-6, rtol=1e-6)
    b = phase_profile(prefix(DetailFilter(fs).update(7*x)), edges)
    np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
    assert (a[:2] >= 0).all() and ((a[2:] >= 0) & (a[2:] <= 1)).all()


def test_concentration_allocation_silence_and_support():
    q = np.ones(6400)
    uniform = np.stack((q, q*q, q, q*q, .25*q, .75*q))
    edges = np.arange(0, 6401, 100)
    a = phase_profile(prefix(uniform), edges)
    np.testing.assert_allclose(a, np.array([0, 0, .25, .75])[:, None] + np.zeros((4,64)))
    impulse = np.zeros(6400); impulse[::100] = 100
    v = np.stack((impulse, impulse**2, impulse, impulse**2, .25*impulse, .75*impulse))
    b = phase_profile(prefix(v), edges)
    np.testing.assert_allclose(b[:2], 2., atol=1e-6)
    assert not phase_profile(prefix(np.zeros_like(v)), edges).any()
    with pytest.raises(ValueError): phase_profile(prefix(v), edges+1)


def test_strong_previous_activity_does_not_erase_weak_current_moments():
    q = np.r_[np.full(6400,1e12),np.tile(np.r_[np.ones(50)*1e-12,np.ones(50)*3e-12],64)]
    v = np.stack((q,q*q,q,q*q,.25*q,.75*q))
    edges = np.arange(6400,12801,100)
    got = profile_from_values(v,edges)
    np.testing.assert_allclose(got[:2],np.log10(1.25),atol=1e-6)
    np.testing.assert_allclose(got[2:],np.array([.25,.75])[:,None]+np.zeros((2,64)),atol=1e-6)


def test_mask_and_original_five_unchanged():
    rng = np.random.RandomState(2)
    x = rng.normal(size=(3,9,8,64)).astype("float32")
    e = rng.normal(size=(3,9,4,64)).astype("float32")
    m = np.ones((3,9)); m[0,:4] = 0
    out = assemble(x,e,m)
    np.testing.assert_array_equal(out[:,:,:5],x[:,:,:5])
    assert not out[0,:4,5:].any()
    assert out.shape == (3,9,9,64)


def test_initial_equivalence_budget_and_extra_channel_gradient():
    torch.manual_seed(37); old = DeepCodedPhaseMap("raw").eval()
    x = torch.randn(3,9,9,64); mask = torch.ones(3,9)
    with torch.no_grad(): expected = old(x,mask)
    for arm,n in [("base",7097),("burst",7257),("allocation",7257),("both",7417)]:
        torch.manual_seed(37); model = DetailPhaseMap(arm).eval()
        with torch.no_grad(): torch.testing.assert_close(model(x,mask),expected,atol=3e-7,rtol=2e-6)
        assert sum(p.numel() for p in model.parameters()) == n
        # The zero-initialized multiscale residual branch delays some gradients;
        # verify the newly added input weights themselves can learn immediately.
        model.train(); z,activity = model(x,mask,return_activity=True)
        (z.square().mean()+.001*activity).backward()
        assert torch.isfinite(model.c1.weight.grad).all()
        if arm != "base": assert model.c1.weight.grad[:,5:].abs().sum() > 0
