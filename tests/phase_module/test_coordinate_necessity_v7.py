import importlib
import numpy as np
import pytest

PACKAGES=['dual_arc_fenet','dual_hydra','dual_arc_fusion']

@pytest.mark.parametrize('package',PACKAGES)
def test_coordinate_controls(package):
    m=importlib.import_module('arc_model.implementations.'+package+'.coordinate_necessity_v7')
    rng=np.random.default_rng(17);p=rng.uniform(.1,1,(14,2,20));p/=p.sum(-1,keepdims=True)
    previous=np.full((14,4),-1,dtype=int)
    for k,lag in enumerate([1,2,4,8]):previous[lag:,k]=np.arange(14-lag)
    ordered=m.paired_dynamics(p,previous)
    np.testing.assert_allclose(ordered,m.paired_dynamics(np.roll(p,7,axis=-1),previous),atol=1e-14)
    shifts=np.arange(14)%20
    q=np.take_along_axis(p,(np.arange(20)[None,None,:]-shifts[:,None,None])%20,axis=-1)
    np.testing.assert_allclose(m.paired_dynamics(p,previous,'registered'),m.paired_dynamics(q,previous,'registered'),atol=1e-14)
    np.testing.assert_allclose(m.paired_dynamics(p,previous,'sorted'),m.paired_dynamics(q,previous,'sorted'),atol=1e-14)
    assert abs(ordered[:,:32]-m.paired_dynamics(q,previous)[:,:32]).max()>.01
    # Current and future changes cannot alter earlier descriptors.
    r=p.copy();r[9:]=r[9:,::-1]
    np.testing.assert_array_equal(m.paired_dynamics(p,previous)[:9],m.paired_dynamics(r,previous)[:9])
    np.testing.assert_array_equal(ordered[:8,35],np.zeros(8))
    previous[5,0]=5
    with pytest.raises(ValueError):m.paired_dynamics(p,previous)


def test_exact_standard_feature_semantics():
    from arc_model.methods.cycle_dynamics import dynamic_features
    m=importlib.import_module('arc_model.implementations.dual_hydra.coordinate_necessity_v7')
    rng=np.random.default_rng(22);p=rng.random((24,2,20));p/=p.sum(-1,keepdims=True)
    records=[dict(record_id=str(i//12),cycle=i%12) for i in range(24)]
    previous=np.full((24,4),-1,dtype=int)
    for i in range(24):
        for k,lag in enumerate([1,2,4,8]):
            if i%12>=lag:previous[i,k]=i-lag
    np.testing.assert_allclose(m.paired_dynamics(p,previous),dynamic_features(p,records),atol=1e-14)
