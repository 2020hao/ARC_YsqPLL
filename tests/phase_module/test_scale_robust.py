import torch
from torch import nn
from arc_model.implementations.map_input_calibration.deep_network import DeepCodedPhaseMap
from arc_model.implementations.map_input_calibration.scale_robust import ScaleRobustPhaseMap,CosineLinear


def test_factorial_reference_is_exact_and_budgets_are_small():
    torch.manual_seed(21);old=DeepCodedPhaseMap('raw').eval()
    torch.manual_seed(21);new=ScaleRobustPhaseMap('batch','linear').eval()
    x=torch.randn(3,9,8,64);mask=torch.ones(3,9)
    with torch.no_grad():torch.testing.assert_close(old(x,mask),new(x,mask),atol=0,rtol=0)
    for norm in ('batch','sample'):
        for head in ('linear','cosine'):
            net=ScaleRobustPhaseMap(norm,head)
            assert sum(p.numel() for p in net.parameters())==7097+(head=='cosine')
            z,activity=net(x,mask,return_activity=True)
            (z.square().mean()+.001*activity).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_cosine_readout_is_invariant_to_positive_feature_scale():
    torch.manual_seed(31);layer=CosineLinear(nn.Linear(32,1)).eval()
    x=torch.randn(7,32)
    torch.testing.assert_close(layer(x),layer(x*50),atol=2e-6,rtol=2e-6)
    assert torch.isfinite(layer(torch.zeros_like(x))).all()


def test_sample_normalization_has_no_batch_dependency_or_running_buffers():
    torch.manual_seed(15);net=ScaleRobustPhaseMap('sample','cosine').eval()
    x=torch.randn(3,9,8,64);m=torch.ones(3,9)
    with torch.no_grad():
        together=net(x,m);alone=net(x[:1],m[:1]);duplicate=net(torch.cat([x[:1],x[:1]*100]),m[:2])
    torch.testing.assert_close(together[:1],alone,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(alone,duplicate[:1],atol=2e-5,rtol=2e-5)
    assert not any('running_' in n or 'num_batches_tracked' in n for n,_ in net.named_buffers())
