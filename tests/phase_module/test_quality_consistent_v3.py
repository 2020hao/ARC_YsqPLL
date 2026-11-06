"""Engineering contracts; these synthetic arrays are not detection evidence."""
import importlib
import numpy as np
import pytest
import torch

PACKAGES=['dual_arc_fenet','dual_hydra','dual_arc_fusion']


def example():
    rng=np.random.default_rng(73107)
    x=rng.normal(size=(64,36)).astype('float32')
    x[:,32:]=(rng.random((64,4))>.25).astype('float32')
    y=np.arange(64)%2;raw=rng.normal(size=64)
    return x,y,raw,np.arange(48)


@pytest.fixture(autouse=True)
def cpu_threads():torch.set_num_threads(2)


@pytest.mark.parametrize('package',PACKAGES)
def test_missing_history_never_changes_score_or_fit_statistics(package):
    m=importlib.import_module('arc_model.implementations.'+package+'.quality_consistent_v3')
    x,y,raw,ids=example();other=x.copy()
    unavailable=np.repeat(x[:,32:]==0,8,axis=1)
    other[:,:32][unavailable]=10000.
    a=m.fit(x,y,raw,ids,calibration_slope=1.2,calibration_intercept=.1,epochs=2)
    b=m.fit(other,y,raw,ids,calibration_slope=1.2,calibration_intercept=.1,epochs=2)
    for key in ['mu','sd','coef','pca_components','pca_projected_mean']:
        np.testing.assert_array_equal(a[key],b[key])
    np.testing.assert_array_equal(m.score(a,x,raw),m.score(a,other,raw))


@pytest.mark.parametrize('package',PACKAGES)
def test_no_history_and_off_restore_raw_including_calibration(package):
    m=importlib.import_module('arc_model.implementations.'+package+'.quality_consistent_v3')
    x,y,raw,ids=example();state=m.fit(x,y,raw,ids,calibration_slope=2.,calibration_intercept=1.,epochs=2)
    x[:,32:]=0
    np.testing.assert_array_equal(m.score(state,x,raw),raw)
    np.testing.assert_array_equal(m.score(state,x,raw,enabled=False),raw)
    state['strength']=np.asarray(0.)
    np.testing.assert_array_equal(m.score(state,x,raw),raw)


@pytest.mark.parametrize('package',PACKAGES)
def test_outer_features_and_labels_do_not_enter_training(package):
    m=importlib.import_module('arc_model.implementations.'+package+'.quality_consistent_v3')
    x,y,raw,ids=example();xx=x.copy();yy=y.copy()
    xx[48:,:32]*=100;yy[48:]=1-yy[48:]
    a=m.fit(x,y,raw,ids,calibration_slope=1.,calibration_intercept=0.,epochs=2)
    b=m.fit(xx,yy,raw,ids,calibration_slope=1.,calibration_intercept=0.,epochs=2)
    for key in ['mu','sd','coef','pca_components','pca_projected_mean','gate_threshold']:
        np.testing.assert_array_equal(a[key],b[key])
    assert set(a['fit_indices'])==set(ids)
    assert set(a['PCA_fit_indices'])<=set(ids[y[ids]==0])


@pytest.mark.parametrize('package',PACKAGES)
def test_consistent_objective_and_serialized_state(package,tmp_path):
    m=importlib.import_module('arc_model.implementations.'+package+'.quality_consistent_v3')
    x,y,raw,ids=example()
    a=m.fit(x,y,raw,ids,calibration_slope=1.5,calibration_intercept=.5,epochs=3,consistent_objective=True)
    b=m.fit(x,y,raw,ids,calibration_slope=1.5,calibration_intercept=.5,epochs=3,consistent_objective=False)
    assert a['coef'].shape==(32,)
    assert np.isfinite(a['train_loss']).all() and not np.allclose(a['train_loss'],b['train_loss'])
    path=tmp_path/'state.npz';m.save(a,path)
    np.testing.assert_array_equal(m.score(a,x,raw),m.score(m.load(path),x,raw))
