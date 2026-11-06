import numpy as np
import torch
from arc_model.implementations.phase_event_snn.network import PhaseEventSNN,EventEncoder
from arc_model.implementations.phase_event_snn.data import decompose


def test_deterministic_signed_single_spike_encoding():
    x=torch.randn(2,9,8,64);mask=torch.ones(2,9);mask[:,:3]=0
    encoder=EventEncoder();a=encoder(x,mask);b=encoder(x,mask)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert set(a.unique().tolist())<={0.,1.} and a.sum(2).max()<=1
    assert not a[:,:3].any()
    zero=encoder(torch.zeros_like(x),mask);assert not zero.any()


def test_decomposition_preserves_original_information():
    rng=np.random.default_rng(7);x=rng.normal(size=(3,9,5,64)).astype('float32');x[:,:,1:3]-=4
    y=decompose(x)
    np.testing.assert_allclose(y[:,:,0]*10.**y[:,:,7],x[:,:,0],atol=1e-6)
    np.testing.assert_allclose(y[:,:,1:3]+y[:,:,5:7],x[:,:,1:3],atol=1e-6)
    np.testing.assert_array_equal(y[:,:,3:5],x[:,:,3:5])


def test_binary_communication_mask_reset_and_gradients():
    torch.set_num_threads(1);torch.manual_seed(4);net=PhaseEventSNN().eval()
    x=torch.randn(3,9,8,64);mask=torch.ones(3,9);mask[:,:2]=0
    a,details=net(x,mask,return_details=True);other=x.clone();other[:,:2]+=100
    torch.testing.assert_close(a,net(other,mask),rtol=0,atol=0)
    for name in ('encoded','layer1','pool1','layer2','pool2','layer3'):
        assert set(details[name].unique().tolist())<={0.,1.}
    a.sum().backward();assert torch.isfinite(net.conv1.weight.grad).all()
    assert sum(p.numel() for p in net.parameters())==8817
    assert sum(p.numel() for p in PhaseEventSNN(spiking=False).parameters())==8817
