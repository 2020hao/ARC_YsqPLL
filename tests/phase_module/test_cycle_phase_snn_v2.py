import torch
from arc_model.implementations.rcr_snn.network import CurrentPhaseSNN
from arc_model.implementations.cycle_phase_snn.network import CyclePhaseSNN,ChannelAffine
from arc_model.preprocessing.rcr_cycle_v2 import CycleRCRConfig,CycleRCR,estimate
import numpy as np


def test_original_structure_exact_and_multistate_initial_equivalence():
    torch.set_num_threads(1);torch.manual_seed(13)
    old=CurrentPhaseSNN().eval();same=CyclePhaseSNN().eval();same.load_state_dict(old.state_dict())
    x=torch.randn(2,9,5,64);mask=torch.ones(2,9);mask[:,:2]=0
    torch.testing.assert_close(old(x,mask),same(x,mask),rtol=0,atol=0)
    level=CyclePhaseSNN(preserve_levels=True).eval();full=CyclePhaseSNN(preserve_levels=True,multistate=True).eval()
    full.load_state_dict(level.state_dict(),strict=False)
    torch.testing.assert_close(level(x,mask),full(x,mask),rtol=0,atol=0)
    a=full(x,mask);other=x.clone();other[:,:2]+=200
    torch.testing.assert_close(a,full(other,mask),rtol=0,atol=0)
    a.sum().backward();assert torch.isfinite(full.temporal_weights.grad).all()
    assert sum(p.numel() for p in full.parameters())==21825


def test_affine_preserves_global_level_and_reference_is_causal():
    norm=ChannelAffine(4);x=torch.randn(2,4,64)
    torch.testing.assert_close(norm(x+2)-norm(x),torch.full_like(x,2))
    t=np.arange(2000)/5000;x=np.sin(2*np.pi*50*t+.4)
    a=estimate(x,chunk_samples=10);x[1200:]+=2*np.sin(2*np.pi*71*t[1200:]);b=estimate(x,chunk_samples=23)
    np.testing.assert_allclose(a.phase_rad[:120],b.phase_rad[:120],rtol=0,atol=0,equal_nan=True)
    assert len(a)==200 and a.memory_samples.max()<=426
    assert a[a.observed_valid].estimate_time_s.iloc[0]<.06
    assert CycleRCRConfig().bootstrap_retry_seconds==.02
    assert CycleRCR().rows is None
