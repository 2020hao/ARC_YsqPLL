"""Single-input aligned-feature switch; original SVM interface stays1024D."""

import numpy as np
from ._source_plugin_hydra_network import Model as Original


class Model(Original):
    def decision_function(self, features, *, aligned=None, enabled=False, available=None):
        if not enabled:
            return super().decision_function(features)
        if aligned is None or np.asarray(aligned).shape != np.asarray(features).shape:
            raise ValueError("Aligned features with unchanged1024D interface required")
        a = np.asarray(aligned)
        if available is not None:
            v = np.asarray(available)
            if v.shape != (len(a),) or v.dtype != bool:
                raise ValueError("Per-window boolean phase availability required")
            a = np.where(v[:, None], a, features)
        return super().decision_function(a)

    def predict(self, features, **kwargs):
        return self.decision_function(features, **kwargs) >= 0
