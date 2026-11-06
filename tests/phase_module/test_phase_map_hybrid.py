import torch
from arc_model.implementations.direct_efficiency.compact_hybrid import CompactHybrid
from arc_model.implementations.direct_efficiency.phase_map_hybrid import PhaseMapHybrid


def test_parameter_count_and_shared_stem_initialization():
    torch.manual_seed(4);base=CompactHybrid()
    torch.manual_seed(4);net=PhaseMapHybrid()
    assert sum(p.numel() for p in net.parameters())==3945
    for prefix in ('c1','n1','dw1','pw1','n2','dw2','pw2','n3'):
        for k,v in getattr(base,prefix).state_dict().items():torch.testing.assert_close(v,getattr(net,prefix).state_dict()[k],rtol=0,atol=0)


def test_phase_state_shape_mask_and_reset():
    torch.manual_seed(4);net=PhaseMapHybrid().eval();x=torch.randn(2,9,8,64);mask=torch.ones(2,9);mask[:,:3]=0
    altered=x.clone();altered[:,:3]=1e4
    with torch.no_grad():
        z,d=net(x,mask,return_details=True);zz,dd=net(altered,mask,return_details=True)
    torch.testing.assert_close(z,zz,rtol=0,atol=0)
    assert d['membrane'].shape==(2,16,8) and d['representation'].shape==(2,128)
    assert not d['spikes'][:,:3].any()
    # Invalid middle history must sever earlier neural state and CNN inputs.
    mask[:,4]=0;changed=x.clone();changed[:,:4]=123.
    with torch.no_grad():torch.testing.assert_close(net(x,mask),net(changed,mask),rtol=0,atol=0)


def test_full_training_gradient():
    net=PhaseMapHybrid();z,a=net(torch.randn(3,9,8,64),torch.ones(3,9),True)
    (z.square().mean()+.001*a).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
