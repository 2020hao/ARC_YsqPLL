import numpy as np
import pytest
from scipy.special import expit
from arc_model.methods.dynamic_score_module import DynamicScoreModule


def test_toggle_and_inference_without_training_samples(tmp_path):
    state=dict(keep=np.arange(2),mu=np.zeros(2),sd=np.ones(2),coef=np.array([.2,-.1]),
               pca_components=np.eye(2),pca_mean=np.zeros(2),alpha=np.array(1.),gate_threshold=np.array(1.))
    p=tmp_path/'module.npz';np.savez(p,**state);m=DynamicScoreModule.load(p)
    base=np.array([-2.,1.]);x=np.zeros((2,36));x[:,0]=1
    np.testing.assert_array_equal(m.predict(base,enabled=False),expit(base))
    assert np.all(m.predict(base,x)>expit(base))
    with pytest.raises(ValueError):m.predict(base,np.zeros((2,8)))
