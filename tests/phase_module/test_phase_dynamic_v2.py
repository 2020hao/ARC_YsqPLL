import importlib
import numpy as np
import pytest
import torch

PACKAGES=['dual_arc_fenet','dual_hydra','dual_arc_fusion']


def fixture(name,dim=40):
    m=importlib.import_module('arc_model.implementations.'+name+'.phase_dynamic_v2')
    bid={'dual_arc_fenet':'B_FENET_fixed20_v1','dual_hydra':'B_HYDRA_FD_fixed20_v1','dual_arc_fusion':'B_FUSION_CURRENT_fixed20_v2'}[name]
    kind='calibrated_svm_logit' if name=='dual_hydra' else 'logit_raw'
    s=dict(dimension=np.asarray(dim),coef=np.ones(dim,dtype='float32')*.1,keep=np.arange(dim),mu=np.zeros(dim,dtype='float32'),sd=np.ones(dim,dtype='float32'),
      pca_components=np.eye(dim,dtype='float32')[:min(dim,6)],pca_mean=np.zeros(dim,dtype='float32'),
      baseline_id=np.asarray(bid),score_kind=np.asarray(kind),calibration_slope=np.asarray(3.),calibration_intercept=np.asarray(4.),
      gate_threshold=np.asarray(1.),strength=np.asarray(0.),fusion=np.asarray('whole_path'))
    return m,bid,kind,s


@pytest.mark.parametrize('name',PACKAGES)
@pytest.mark.parametrize('dim',[4,32,36,40])
def test_whole_path_zero_is_exact_raw(name,dim):
    m,bid,kind,s=fixture(name,dim);c=m.Correction(s,baseline_id=bid,score_kind=kind)
    raw=np.asarray([-2.,.1,1.],dtype='float32');x=np.zeros((3,dim),dtype='float32')
    assert np.array_equal(c.score(raw,x),raw)
    s['strength']=np.asarray(1.);on=m.Correction(s,baseline_id=bid,score_kind=kind)
    assert np.array_equal(on.score(raw,x,enabled=False),raw)


@pytest.mark.parametrize('name',PACKAGES)
def test_whole_path_formula_and_parent_binding(name):
    m,bid,kind,s=fixture(name);s['strength']=np.asarray(.5);c=m.Correction(s,baseline_id=bid,score_kind=kind)
    raw=np.asarray([-1.,.1,1.]);x=np.zeros((3,40),dtype='float32');gate=1/(1+np.exp(-1/(.15+1e-8)))
    expected=raw+.5*gate*((3.-1)*raw+4.)
    np.testing.assert_allclose(c.score(raw,x),expected,rtol=1e-7,atol=1e-7)
    with pytest.raises(ValueError,match='parent'):m.Correction(s,baseline_id='wrong',score_kind=kind)


@pytest.mark.parametrize('name',PACKAGES)
def test_condition_metrics_equal_weighting(name):
    m,*_=fixture(name);y=np.array([0,1,0,0,0,1]);p=np.array([0,0,1,1,1,1],bool);g=np.array([0,0,1,1,1,1]);ids=np.arange(6)
    bacc,recall,fpr=m.condition_metrics(y,p,g,ids)
    assert bacc==.5 and recall==.5 and fpr==.5


@pytest.mark.parametrize('name',['dual_arc_fenet','dual_arc_fusion'])
def test_neural_lambda_zero_returns_identical_logits(name,monkeypatch):
    m,bid,kind,s=fixture(name)
    monkeypatch.setattr(m.Original,'forward',lambda self,x,spectral=None:x)
    net=m.Model(extra=0).eval();net.attach_correction(s)
    x=torch.tensor([[.123,.456],[1e5,1e-4]],dtype=torch.float32)
    out=net(x,dynamic=np.zeros((2,40),dtype='float32'),enabled=True)
    assert torch.equal(out,x)
