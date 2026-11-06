"""Portable inference adapter for this model's protected dynamic encoder."""
import numpy as np
from . import neural_dynamic_v1 as method

BASELINE_ID = 'B_HYDRA_FD_fixed20_v1'


class ProtectedDynamicEncoder:
    def __init__(self, state):
        if str(state['baseline_id']) != BASELINE_ID:
            raise ValueError('Frozen baseline identity mismatch')
        self.beta = float(state['beta'])
        self.old = {k.removeprefix('old::'): np.asarray(v) for k,v in state.items() if k.startswith('old::')}
        self.head = {k.removeprefix('head::'): np.asarray(v) for k,v in state.items() if k.startswith('head::')}

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls({k:z[k].copy() for k in z.files})

    def score(self, raw_margin, dynamic36, *, new_encoder_enabled=True, device='cpu'):
        raw = np.asarray(raw_margin); x = np.asarray(dynamic36)
        if x.shape != (len(raw),36) or not np.isfinite(x).all() or not np.isfinite(raw).all():
            raise ValueError('Finite raw margins and unchanged 36D features required')
        s = self.old
        old_score, scale, _ = method.base_terms(s,x,raw,float(s['calibration_slope']),
              float(s['calibration_intercept']),float(s['strength']),float(s['quantile']),bool(s['raw_off']))
        # Disabling the new encoder retains the complete OLD enhancement.
        if not new_encoder_enabled or self.beta == 0 or not np.any(scale):
            return old_score
        delta = method.predict(self.head,x,device=device)
        return method.corrected(old_score,scale,delta,self.beta)

    def probability(self, raw_margin, dynamic36, **kwargs):
        from scipy.special import expit
        return expit(self.score(raw_margin,dynamic36,**kwargs))

