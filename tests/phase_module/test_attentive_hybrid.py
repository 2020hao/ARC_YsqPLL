import torch
from torch.nn import functional as F
from arc_model.implementations.direct_efficiency.compact_hybrid import CompactHybrid
from arc_model.implementations.direct_efficiency.attentive_hybrid import AttentiveHybrid


def test_initial_uniform_pool_and_common_initialization():
    for mode in ('content','static'):
        torch.manual_seed(40);base=CompactHybrid().eval()
        torch.manual_seed(40);net=AttentiveHybrid(mode).eval()
        for k,v in base.state_dict().items():torch.testing.assert_close(v,net.state_dict()[k],rtol=0,atol=0)
        h=torch.randn(9,24,64);out,w=net.attentive_pool(h,True)
        torch.testing.assert_close(w,torch.full_like(w,.25),rtol=0,atol=0)
        torch.testing.assert_close(out,F.avg_pool1d(h,4),rtol=0,atol=0)
        x=torch.randn(3,9,8,64);m=torch.ones(3,9)
        with torch.no_grad():torch.testing.assert_close(base(x,m),net(x,m),rtol=0,atol=0)
        assert sum(p.numel() for p in net.parameters())==3647


def test_content_dependence_static_independence_and_mask():
    h=torch.randn(9,24,64)
    for mode in ('content','static'):
        net=AttentiveHybrid(mode).eval()
        with torch.no_grad():net.pool_score.weight.fill_(.1)
        _,a=net.attentive_pool(h,True);_,b=net.attentive_pool(h*3,True)
        torch.testing.assert_close(a.sum(-1),torch.ones_like(a.sum(-1)))
        if mode=='static':torch.testing.assert_close(a,b,rtol=0,atol=0)
        else:assert not torch.equal(a,b)
        x=torch.randn(2,9,8,64);mask=torch.ones(2,9);mask[:,:3]=0;altered=x.clone();altered[:,:3]=1e4
        with torch.no_grad():torch.testing.assert_close(net(x,mask),net(altered,mask),rtol=0,atol=0)


def test_attention_training_gradients():
    for mode in ('content','static'):
        net=AttentiveHybrid(mode);x=torch.randn(3,9,8,64);m=torch.ones(3,9)
        z,a=net(x,m,True);(z.square().mean()+.001*a).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        assert net.pool_score.weight.grad.abs().sum()>0
