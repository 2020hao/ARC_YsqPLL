import torch
from arc_model.implementations.rcr_snn.network import CurrentPhaseSNN,Spike


def test_snn_shapes_masked_history_and_gradients():
    torch.set_num_threads(1);torch.manual_seed(3)
    net=CurrentPhaseSNN().eval();x=torch.randn(3,9,5,64);mask=torch.ones(3,9);mask[:,:3]=0
    other=x.clone();other[:,:3]+=100
    a=net(x,mask);b=net(other,mask)
    assert a.shape==(3,)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    a.sum().backward()
    assert net.input_map.weight.grad is not None and torch.isfinite(net.input_map.weight.grad).all()
    assert sum(p.numel() for p in net.parameters())==21537
    analog=CurrentPhaseSNN(False)
    assert sum(p.numel() for p in analog.parameters())==21537


def test_spikes_are_binary_and_surrogate_gradient_is_finite():
    x=torch.tensor([-1.,-.1,.1,1.],requires_grad=True);y=Spike.apply(x)
    assert set(y.tolist())=={0.,1.}
    y.sum().backward();assert torch.isfinite(x.grad).all() and (x.grad>0).all()
