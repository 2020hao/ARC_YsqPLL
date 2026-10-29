"""Deployment interface: dynamic36 + original logit -> corrected probability.

Only a saved parameter bundle is required; training samples are not needed at
inference. Disabling the module returns the original sigmoid probability.
"""

import numpy as np
from scipy.special import expit
from .dynamic_correction import replay


class DynamicScoreModule:
    def __init__(self,state):
        required={'keep','mu','sd','coef','pca_components','pca_mean','alpha','gate_threshold'}
        if not required.issubset(state):raise ValueError('Incomplete correction state')
        self.state={k:np.asarray(v).copy() for k,v in state.items()}
        if float(self.state['gate_threshold'])<=0:raise ValueError('Positive gate threshold required')

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as state:return cls(dict(state))

    def predict(self,base_logit,dynamic=None,*,enabled=True):
        base=np.asarray(base_logit)
        if base.ndim!=1 or not np.isfinite(base).all():raise ValueError('Finite per-window logits required')
        if not enabled:return expit(base)
        x=np.asarray(dynamic)
        if x.shape!=(len(base),36) or not np.isfinite(x).all():raise ValueError('Finite [N,36] dynamics required')
        delta,distance=replay(self.state,x)
        threshold=float(self.state['gate_threshold']);alpha=float(self.state['alpha'])
        w=alpha*expit(-(distance-threshold)/(.15*threshold+1e-8))
        return expit(base+w*delta)
