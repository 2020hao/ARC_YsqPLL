"""Train-fold SparseScaler and optional physical-feature transformation."""

import numpy as np
import torch
from arc_model.models.standalone.hydra import SparseScaler


class Features:
    def __init__(self, F, P_, y, method):
        self.F = F
        self.P = P_
        self.y = y
        self.method = method

    def fit(self, tr):
        self.sc = SparseScaler()
        self.sc.fit(torch.from_numpy(self.F[tr]))
        if self.method >= 2:
            # Enhancement reference fitted ONLY to healthy training windows.
            normal = self.P[tr[self.y[tr] == 0]]
            self.med = np.median(normal, axis=0)
            self.mad = np.maximum(1.4826 * np.median(np.abs(normal - self.med), axis=0), 0.02)
            e = self.enhance(self.P[tr])
            self.mu = e.mean(axis=0)
            self.sd = np.maximum(e.std(axis=0), 0.05)
        return self

    def enhance(self, x):
        if self.method >= 4:
            return x
        z = (x - self.med) / self.mad
        return np.sign(z) * np.log1p(np.abs(z))

    def transform(self, ids):
        x = self.sc.transform(torch.from_numpy(self.F[ids])).numpy()
        if self.method >= 2:
            x = np.c_[x, (self.enhance(self.P[ids]) - self.mu) / self.sd]
        return np.ascontiguousarray(x, np.float64)
