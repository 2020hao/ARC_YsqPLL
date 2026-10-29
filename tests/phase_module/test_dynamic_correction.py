import numpy as np
from sklearn.decomposition import PCA
from arc_model.methods.dynamic_correction import feature_names, select_columns, calibration_split, replay, weights


def test_selection_is_training_only_and_shared():
    rng=np.random.default_rng(0);x=rng.normal(size=(40,36)).astype('float32')
    y=np.tile([0,1],20);g=np.repeat(np.arange(10),4);ids=np.arange(32)
    a=select_columns(x,ids,y,g)
    changed=y.copy();changed[32:]=1-changed[32:]
    b=select_columns(x,ids,changed,g)
    np.testing.assert_array_equal(a,b)
    assert len(feature_names())==36 and len(a)<=24


def test_pca_replay_matches_sklearn_transform():
    rng=np.random.default_rng(4);x=rng.normal(size=(30,8)).astype('float32')
    mu=x[:20].mean(0);sd=x[:20].std(0);z=(x-mu)/sd
    p=PCA(6,random_state=0).fit(z[:20]);coef=rng.normal(size=8).astype('float32')
    state=dict(keep=np.arange(8),mu=mu,sd=sd,coef=coef,pca_components=p.components_,pca_mean=p.mean_)
    delta,distance=replay(state,x)
    np.testing.assert_allclose(delta,z@coef,rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(distance,np.sqrt(np.mean(p.transform(z)**2,axis=1)),rtol=1e-5,atol=1e-6)


def test_calibration_group_isolation_and_confidence():
    g=np.repeat(np.arange(12).astype(str),4);tr=np.arange(len(g))
    fit,cal=calibration_split(g,tr,0)
    assert not set(g[fit])&set(g[cal]) and len(fit)+len(cal)==len(tr)
    d=np.array([0.,1.,2.,3.]);w=weights(d,np.arange(4),1.,.5)
    assert np.all(np.diff(w)<0) and np.all((w>=0)&(w<=1))
    np.testing.assert_array_equal(weights(d,np.arange(4),0.,.5),np.zeros(4))
