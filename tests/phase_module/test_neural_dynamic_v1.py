import importlib
import numpy as np
import pytest
import torch


@pytest.mark.parametrize('package', ['dual_arc_fenet','dual_hydra','dual_arc_fusion'])
def test_capacity_initial_identity_and_replay(package):
    torch.set_num_threads(1)
    m=importlib.import_module('arc_model.implementations.'+package+'.neural_dynamic_v1')
    x=torch.randn(7,36);valid=torch.ones(7,4)
    for name, cls in m.ARCHITECTURES.items():
        net=cls().eval()
        assert sum(p.numel() for p in net.parameters())==m.PARAMETERS[name]
        with torch.no_grad():np.testing.assert_array_equal(net(x,valid).numpy(),np.zeros(7))
        state=m.head_state(net,dict(mu=np.zeros(36),sd=np.ones(36)),np.arange(7),name,0,[])
        np.testing.assert_array_equal(m.predict(state,x.numpy()),np.zeros(7))
    raw=np.arange(7,dtype='float32')
    np.testing.assert_array_equal(m.corrected(raw,np.ones(7),np.full(7,np.nan),0),raw)


def test_shared_tokens_use_lag_major_order_and_validity():
    m=importlib.import_module('arc_model.implementations.dual_arc_fusion.neural_dynamic_v1')
    torch.manual_seed(7);net=m.LagSharedResidual().eval()
    with torch.no_grad():net.output.weight.fill_(.1)
    x=torch.randn(2,36);altered=x.clone();altered[:,:8]+=100
    valid=torch.ones(2,4);valid[:,0]=0
    with torch.no_grad():
        a=net(x,valid);b=net(altered,valid)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
