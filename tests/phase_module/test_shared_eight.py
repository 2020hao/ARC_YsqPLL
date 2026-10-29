import numpy as np
import pytest
import torch

from arc_model.methods.phase_representation.shared_features import NAMES, shared_eight


def test_same_eight_definition_and_gain_invariance():
    rng=np.random.default_rng(9)
    w=rng.normal(size=(5,1000));e=rng.random((5,2,20));t=np.full(5,.02)
    a=shared_eight(w,e,t);b=shared_eight(w*3,e*9,t)
    assert a.shape==(5,8) and len(NAMES)==8
    np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-7)


def test_shift_keeps_strength_but_changes_position():
    w=np.ones((3,1000));e=np.zeros((3,2,20));e[:,:,0]=1
    a=shared_eight(w,e,np.full(3,.02))
    b=shared_eight(w,e,np.full(3,.02),shifts=np.full(3,3,dtype=int))
    np.testing.assert_array_equal(a[:,:2],b[:,:2])
    assert not np.allclose(a[:,2:],b[:,2:])
    assert (a[:,[2,5]]==1).all() and (b[:,[2,5]]==0).all()
    with pytest.raises(ValueError):shared_eight(w,-e,np.full(3,.02))


def test_batched_head_gradients_are_independent():
    torch.manual_seed(3)
    x=torch.randn(3,7,12);y=torch.tensor([0,1,1,0,1,0,1])
    w=torch.randn(3,2,12,requires_grad=True);b=torch.randn(3,2,requires_grad=True)
    logits=torch.einsum('and,aod->ano',x,w)+b[:,None]
    loss=torch.nn.functional.cross_entropy(logits.reshape(-1,2),y[None].expand(3,-1).reshape(-1),reduction='none').reshape(3,-1).mean(1).sum()
    loss.backward()
    for k in range(3):
        wk=w[k].detach().clone().requires_grad_(True);bk=b[k].detach().clone().requires_grad_(True)
        torch.nn.functional.cross_entropy(x[k]@wk.T+bk,y).backward()
        torch.testing.assert_close(w.grad[k],wk.grad)
        torch.testing.assert_close(b.grad[k],bk.grad)
