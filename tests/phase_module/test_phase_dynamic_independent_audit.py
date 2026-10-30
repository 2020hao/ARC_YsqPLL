"""The audit must distinguish training-only states from leaked fitted statistics."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from sklearn.decomposition import PCA


@pytest.fixture
def audit_module():
    path=Path(__file__).resolve().parents[2]/'scripts/audit_phase_dynamic_evidence_20261030.py'
    import sys
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('phase_evidence_independent_audit',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training_fixture():
    rng=np.random.default_rng(7)
    x=rng.normal(size=(40,4)).astype('float32')
    ids=np.arange(20)
    y=np.tile([0,1],20)
    keep=np.arange(4)
    mu=x[ids][:,keep].mean(0)
    sd=np.maximum(x[ids][:,keep].std(0),1e-6)
    z=(x[:,keep]-mu)/sd
    pca=PCA(n_components=4,random_state=3).fit(z[ids[y[ids]==0]])
    state=dict(fit_indices=ids,keep=keep,mu=mu,sd=sd,pca_mean=pca.mean_,pca_components=pca.components_)
    return x,y,ids,state


def test_stats_audit_ignores_changes_outside_fit_partition(audit_module):
    x,y,ids,state=training_fixture()
    audit_module.fitted_stats(state,x,y,ids,3)
    x[20:]+=10000
    audit_module.fitted_stats(state,x,y,ids,3)


def test_stats_audit_rejects_all_data_normalization(audit_module):
    x,y,ids,state=training_fixture()
    x[20:]+=10000
    leaked=dict(state,mu=x.mean(0),sd=x.std(0))
    with pytest.raises(AssertionError):
        audit_module.fitted_stats(leaked,x,y,ids,3)
