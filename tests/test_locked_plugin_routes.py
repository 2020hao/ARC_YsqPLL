"""Input-only candidates preserve the network and implement exact fallback/off."""
import importlib
import numpy as np
import pytest
import torch


@pytest.mark.parametrize('name', ['arc_fenet', 'arc_fusion'])
def test_network_has_no_new_parameters_and_switch_is_input_only(name):
    package='arc_model.implementations.plugin_'+name
    native=importlib.import_module(package+'.network').Model
    plugin=importlib.import_module(package+'.model').Model
    torch.manual_seed(31)
    source=native(extra=0).eval()
    candidate=plugin(extra=0).eval()
    candidate.load_state_dict(source.state_dict(),strict=True)
    assert sum(p.numel() for p in source.parameters())==sum(p.numel() for p in candidate.parameters())
    available=torch.tensor([True,False])
    if name=='arc_fenet':
        x=torch.randn(2,3,100,100);a=torch.randn_like(x);s=b=None
    else:
        x=torch.randn(2,1,3,250);a=torch.randn_like(x);s=torch.randn(2,1,32);b=torch.randn_like(s)
    with torch.no_grad():
        off=candidate(x,s,aligned={'temporal':a,'spectrum':b},enabled=False,available=available)
        expected_off=source(x,s)
        chosen=torch.where(available.reshape((-1,)+(1,)*(x.ndim-1)),a,x)
        chosen_s=None if s is None else torch.where(available[:,None,None],b,s)
        expected_on=source(chosen,chosen_s)
        on=candidate(x,s,aligned={'temporal':a,'spectrum':b},enabled=True,available=available)
        off_again=candidate(x,s,enabled=False)
    torch.testing.assert_close(off,expected_off,rtol=0,atol=0)
    torch.testing.assert_close(on,expected_on,rtol=0,atol=0)
    torch.testing.assert_close(off_again,expected_off,rtol=0,atol=0)


def test_hydra_keeps_native_1024_dimension_and_fallback():
    from arc_model.implementations.plugin_hydra.model import Model
    state=dict(mu=np.zeros(1024,np.float32),sigma=np.ones(1024,np.float32),epsilon=np.ones(1024,np.float32),weight=np.arange(1024,dtype=np.float64)/1024,intercept=np.array(.1))
    model=Model(state)
    x=np.ones((3,1024),np.float32)
    aligned=np.full_like(x,4.)
    available=np.array([True,False,True])
    off=model.decision_function(x)
    on=model.decision_function(x,aligned=aligned,enabled=True,available=available)
    expected=model.decision_function(np.where(available[:,None],aligned,x))
    np.testing.assert_array_equal(on,expected)
    np.testing.assert_array_equal(model.decision_function(x,aligned=aligned,enabled=False),off)
    with pytest.raises(ValueError,match='1024D'):
        model.decision_function(x,aligned=np.ones((3,448)),enabled=True)
