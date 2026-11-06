"""Mask independence and differentiable standalone direct candidates."""
import torch
from arc_model.implementations.direct_efficiency.compact_cnn import CompactCNN
from arc_model.implementations.direct_efficiency.compact_hybrid import CompactHybrid


def test_invalid_history_values_do_not_enter_prediction():
    torch.manual_seed(11)
    x=torch.randn(2,9,8,64);mask=torch.ones(2,9);mask[:,:3]=0
    altered=x.clone();altered[:,:3]=12345
    for cls in (CompactCNN,CompactHybrid):
        net=cls().eval()
        with torch.no_grad():torch.testing.assert_close(net(x,mask),net(altered,mask),rtol=0,atol=0)


def test_full_training_path_and_declared_parameters():
    for cls,n in ((CompactCNN,4258),(CompactHybrid,3641)):
        torch.manual_seed(11);net=cls();x=torch.randn(2,9,8,64);mask=torch.ones(2,9)
        out,a=net(x,mask,True);(out.square().mean()+.001*a).backward()
        assert sum(p.numel() for p in net.parameters())==n
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
