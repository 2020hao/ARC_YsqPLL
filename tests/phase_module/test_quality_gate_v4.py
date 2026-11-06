import importlib
import numpy as np
import pytest

@pytest.mark.parametrize('package',['dual_arc_fenet','dual_hydra','dual_arc_fusion'])
def test_gate_semantics_and_missing_history(package):
    m=importlib.import_module('arc_model.implementations.'+package+'.quality_gate_v4')
    z=np.zeros((4,32),dtype=np.float32);h=np.array([0.,.25,.5,1.],dtype=np.float32)
    state={'gate_kind':np.asarray('no_distance')}
    np.testing.assert_array_equal(m.quality(state,z,h),h)
    state['gate_kind']=np.asarray('reference_quality');q=np.array([1.,1.,.4,0.],dtype=np.float32)
    np.testing.assert_allclose(m.quality(state,z,h,q),h*q)
    with pytest.raises(ValueError):m.quality(state,z,h,np.array([0.,1.,2.,0.]))

@pytest.mark.parametrize('package',['dual_arc_fenet','dual_hydra','dual_arc_fusion'])
def test_nonpca_fit_has_no_pca_and_ignores_other_labels(package):
    m=importlib.import_module('arc_model.implementations.'+package+'.quality_gate_v4')
    rng=np.random.default_rng(4);x=np.column_stack([rng.normal(size=(50,32)),np.ones((50,4))]).astype(np.float32)
    y=np.arange(50)%2;raw=rng.normal(size=50);ids=np.arange(30);q=np.linspace(0,1,50,dtype=np.float32)
    kw=dict(calibration_slope=1.,calibration_intercept=0.,epochs=2,gate_kind='reference_quality',reference_quality=q)
    a=m.fit(x,y,raw,ids,**kw);y[30:]=1-y[30:];b=m.fit(x,y,raw,ids,**kw)
    np.testing.assert_array_equal(a['coef'],b['coef']);assert 'pca_components' not in a
    score=m.score(a,x,raw,reference_quality=q);assert score[0]==raw[0]
    np.testing.assert_array_equal(m.score(a,x,raw,enabled=False),raw)
