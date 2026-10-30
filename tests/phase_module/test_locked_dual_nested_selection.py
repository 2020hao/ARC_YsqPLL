"""Checks for grouped nested selection and switchable candidate calibration."""
import importlib
import numpy as np
import pytest
import torch

PACKAGES=('dual_arc_fenet','dual_hydra','dual_arc_fusion')


def data():
    # Both classes in every group; groups0..2 train inner0 and3..5 held out.
    groups=np.repeat(np.arange(6),6);y=np.tile([0,1,0,1,0,1],6)
    rng=np.random.default_rng(9);x=rng.normal(size=(36,36)).astype('float32')
    a=np.arange(18);b=np.arange(18,30);c=np.arange(30,36);all_ids=np.arange(36)
    receipts=[dict(train=a,validation=np.r_[b,c]),dict(train=np.r_[b,c],validation=a)]
    return x,y,groups,a,b,c,all_ids,receipts


@pytest.mark.parametrize('name',PACKAGES)
def test_nested_rejects_selection_backbone_leakage(name):
    module=importlib.import_module('arc_model.implementations.'+name+'.training')
    x,y,g,a,b,c,ids,receipts=data();config=module.load_config()
    with pytest.raises(ValueError,match='isolated'):
        module.fit_from_nested_oof(x,y,np.zeros(len(y)),g,ids,receipts,
            selection_offset=np.zeros(len(y)),selection_fit=b,selection_validation=c,
            selection_base_train=np.r_[a,c],baseline_id=config['parent_baseline_id'],score_kind='logit_raw')


@pytest.mark.parametrize('name',PACKAGES)
def test_nested_rejects_missing_oof_rows(name):
    module=importlib.import_module('arc_model.implementations.'+name+'.training')
    x,y,g,a,b,c,ids,receipts=data();config=module.load_config()
    with pytest.raises(ValueError,match='OOF coverage'):
        module.fit_from_nested_oof(x,y,np.zeros(len(y)),g,ids,receipts[:1],
            selection_offset=np.zeros(len(y)),selection_fit=b,selection_validation=c,
            selection_base_train=a,baseline_id=config['parent_baseline_id'],score_kind='logit_raw')


@pytest.mark.parametrize('name',['dual_arc_fenet','dual_arc_fusion'])
def test_neural_optional_calibration_switch(name,monkeypatch):
    module=importlib.import_module('arc_model.implementations.'+name+'.model')
    original=module.Original
    # Exercise the actual candidate wrapper with a cheap deterministic base forward.
    def stub(self,x,spectral=None):return x
    monkeypatch.setattr(original,'forward',stub)
    model=module.Model(extra=0).eval();logits=torch.tensor([[.1,.4],[-.2,.5]])
    config=importlib.import_module('arc_model.implementations.'+name+'.configuration').load_config()
    state=dict(keep=np.arange(36),mu=np.zeros(36),sd=np.ones(36),coef=np.zeros(36),
        pca_components=np.eye(36)[:6],pca_mean=np.zeros(36),alpha=np.asarray(0.),
        gate_threshold=np.asarray(1.),baseline_id=np.asarray(config['parent_baseline_id']),score_kind=np.asarray('logit_raw'))
    model.attach_correction(state,calibration=dict(slope=2.,intercept=.3,fit_indices=np.arange(2)))
    assert torch.equal(logits,model(logits,enabled=False))
    enabled=model(logits,dynamic=np.zeros((2,36)),enabled=True)
    torch.testing.assert_close(enabled[:,1]-enabled[:,0],2*(logits[:,1]-logits[:,0])+.3)
    # Strength0 can still calibrate; disabling must be exactly the original output.
    assert not torch.equal(enabled,logits)
