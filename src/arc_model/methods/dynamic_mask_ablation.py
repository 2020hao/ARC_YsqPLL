"""Twelve-slot input ablations with identical neural parameter budgets."""
import numpy as np
from sklearn.decomposition import PCA


def ablation_inputs(dynamic,masks,offset,ids,keep,content):
    """D/M/DM/Z/C/CM; fit no transform outside the supplied training IDs."""
    state={};n=len(offset)
    if content in ['D','DM']:
        values=dynamic[:,keep];mu=values[ids].mean(0);sd=np.maximum(values[ids].std(0),1e-6)
        z=(values-mu)/sd;k=min(8,z.shape[1],len(ids))
        pca=PCA(n_components=k,random_state=0).fit(z[ids]);d=pca.transform(z)
        if k<8:d=np.pad(d,((0,0),(0,8-k)))
        state.update(dynamic_keep=np.asarray(keep),dynamic_mean=mu,dynamic_std=sd,pca_mean=pca.mean_,pca_components=pca.components_)
        x=np.c_[d,masks if content=='DM' else np.zeros((n,4))]
    elif content=='M':x=np.c_[np.zeros((n,8)),masks]
    elif content=='Z':x=np.zeros((n,12))
    elif content=='C':x=np.repeat(offset[:,None],12,axis=1)
    elif content=='CM':x=np.c_[np.repeat(offset[:,None],8,axis=1),masks]
    else:raise ValueError(content)
    x=x.astype('float32');mu=x[ids].mean(0);sd=np.maximum(x[ids].std(0),1e-6)
    return np.clip((x-mu)/sd,-8,8).astype('float32'),dict(mean=mu,std=sd,**state)
