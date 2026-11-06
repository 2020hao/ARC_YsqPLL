import importlib
import numpy as np
import pytest

@pytest.mark.parametrize('pkg',['dual_arc_fenet','dual_hydra','dual_arc_fusion'])
def test_coordinate_blend_preserves_fallback_and_valid_source(pkg):
    m=importlib.import_module('arc_model.implementations.'+pkg+'.coordinate_utility_v5')
    f=np.column_stack([np.ones((4,32)),np.ones((4,4))]).astype(np.float32);r=f.copy();r[:,:32]=3
    q=np.array([0.,1.,.25,.5],dtype=np.float32);out=m.mixed_coordinate(f,r,q)
    np.testing.assert_array_equal(out[0],f[0]);np.testing.assert_array_equal(out[1],r[1]);np.testing.assert_array_equal(out[2,:32],1.5)
    r[3,32:]=0;np.testing.assert_array_equal(m.mixed_coordinate(f,r,q)[3],f[3])

@pytest.mark.parametrize('pkg',['dual_arc_fenet','dual_hydra','dual_arc_fusion'])
def test_router_honest_indices_and_degenerate_cases(pkg):
    m=importlib.import_module('arc_model.implementations.'+pkg+'.coordinate_utility_v5')
    rng=np.random.default_rng(42);v=rng.normal(size=(20,5));y=np.arange(20)%2;g=np.repeat(np.arange(4),5);ids=np.arange(10)
    raw=np.where(y,1.,-1.);adv=-raw
    state=m.fit_router(v,y,raw,adv,g,ids);assert int(state['router_constant'])==0
    raw=-raw;state=m.fit_router(v,y,raw,adv,g,ids);assert int(state['router_constant'])==0 # ties
    adv=-raw;state=m.fit_router(v,y,raw,adv,g,ids);assert int(state['router_constant'])==1
    adv[::3]=-adv[::3];a=m.fit_router(v,y,raw,adv,g,ids)
    y[10:]=1-y[10:];b=m.fit_router(v,y,raw,adv,g,ids)
    np.testing.assert_array_equal(a['router_coef'],b['router_coef']);assert m.route(a,v).dtype==bool

@pytest.mark.parametrize('pkg',['dual_arc_fenet','dual_hydra','dual_arc_fusion'])
def test_router_learns_relative_benefit_without_outside_labels(pkg):
    m=importlib.import_module('arc_model.implementations.'+pkg+'.coordinate_utility_v5')
    rng=np.random.default_rng(7);v=rng.normal(size=(40,5));y=np.arange(40)%2;ids=np.arange(30);g=np.repeat(np.arange(8),5)
    truth=np.where(y==1,1.,-1.);raw=truth.copy();raw[::2]*=-1;adv=-raw
    a=m.fit_router(v,y,raw,adv,g,ids);assert int(a['router_constant'])==-1
    assert int(a['router_benefit'])==15 and int(a['router_harm'])==15
    y[30:]=1-y[30:];b=m.fit_router(v,y,raw,adv,g,ids)
    np.testing.assert_array_equal(a['router_coef'],b['router_coef'])
    assert np.isfinite(a['router_coef']).all()
