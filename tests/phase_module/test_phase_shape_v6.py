import importlib
import numpy as np
import pytest

PACKAGES=['dual_arc_fenet','dual_hydra','dual_arc_fusion']

def metadata(n):
    return [dict(record_id=0,anchor_index=i,kind='pll',segment=1,epoch=0,start_seconds=i*.02,end_seconds=(i+1)*.02,available_seconds=(i+1)*.02,anchor_seconds=(i+1)*.02) for i in range(n)]

@pytest.mark.parametrize('package',PACKAGES)
def test_known_shift_and_half_normalization(package):
    m=importlib.import_module('arc_model.implementations.'+package+'.phase_shape_v6')
    e=np.zeros((3,2,20));e[:,:,2]=1;e[1]=np.roll(e[0],3,axis=-1);e[2]=2*e[1]
    x,v,d=m.six_features(e,metadata(3),2000000)
    np.testing.assert_allclose(x[1,[0,3]],.3);np.testing.assert_allclose(x[1,[1,4]],0.)
    assert not v[0,[0,1,3,4]].any();assert not v[:,[2,5]].any()
    e=np.ones((2,2,20));x,v,_=m.six_features(e,metadata(2),1000000)
    assert not v[:,[0,3]].any();assert v[1,[1,4]].all();np.testing.assert_array_equal(x[:,[2,5]],0.)

@pytest.mark.parametrize('package',PACKAGES)
def test_rotation_identities_and_epoch_gap(package):
    m=importlib.import_module('arc_model.implementations.'+package+'.phase_shape_v6');rng=np.random.default_rng(3)
    e=rng.uniform(.01,2,size=(12,2,20));rows=metadata(12)
    x,v,_=m.six_features(e,rows,1000000);r,rv,_=m.six_features(e,rows,1000000,np.full(12,7));q,qv,_=m.six_features(e,rows,1000000,np.arange(12)%20)
    np.testing.assert_allclose(x[:,[0,1,3,4]],r[:,[0,1,3,4]],atol=1e-7);np.testing.assert_allclose(x[:,[1,4]],q[:,[1,4]],atol=1e-7)
    rows[5]['epoch']=1;rows[8]['start_seconds']+=.001
    _,_,d=m.six_features(e,rows,1000000);assert d['previous_index'][5]==-1 and d['previous_index'][6]==-1 and d['previous_index'][8]==-1

@pytest.mark.parametrize('package',PACKAGES)
def test_missing_and_removal_are_neutral_and_fit_is_train_only(package):
    m=importlib.import_module('arc_model.implementations.'+package+'.phase_shape_v6');rng=np.random.default_rng(9)
    x=np.column_stack([rng.normal(size=(30,32)),np.ones((30,4))]).astype(np.float32)
    e=rng.normal(size=(30,6)).astype(np.float32);v=np.ones((30,6),bool);v[::4]=False
    y=np.arange(30)%2;raw=rng.normal(size=30);ids=np.arange(20);active=np.array([1,0,1,1,1,1],bool)
    kw=dict(calibration_slope=1.,calibration_intercept=0.,epochs=2,seed=2)
    a=m.fit(x,e,v,y,raw,ids,active,**kw);e2=e.copy();e2[~v]=10000;e2[:,1]=10000;y[20:]=1-y[20:]
    b=m.fit(x,e2,v,y,raw,ids,active,**kw)
    np.testing.assert_array_equal(a['coef'],b['coef']);np.testing.assert_array_equal(a['extra_coef'],b['extra_coef'])
    np.testing.assert_array_equal(m.score(a,x,e,v,raw),m.score(a,x,e2,v,raw))
    np.testing.assert_array_equal(m.score(a,x,e,v,raw,enabled=False),raw)

@pytest.mark.parametrize('package',PACKAGES)
def test_no_new_features_exact_old_base_training(package):
    m=importlib.import_module('arc_model.implementations.'+package+'.phase_shape_v6');old=importlib.import_module('arc_model.implementations.'+package+'.quality_gate_v4')
    rng=np.random.default_rng(21);x=np.column_stack([rng.normal(size=(40,32)),rng.integers(0,2,size=(40,4))]).astype(np.float32)
    e=np.zeros((40,6),np.float32);v=np.zeros_like(e,bool);y=np.arange(40)%2;raw=rng.normal(size=40);ids=np.arange(30)
    kw=dict(calibration_slope=1.1,calibration_intercept=-.1,epochs=3,seed=1)
    a=m.fit(x,e,v,y,raw,ids,np.zeros(6,bool),**kw);b=old.fit(x,y,raw,ids,gate_kind='no_distance',**kw)
    np.testing.assert_array_equal(a['coef'],b['coef']);np.testing.assert_array_equal(m.score(a,x,e,v,raw),old.score(b,x,raw))
