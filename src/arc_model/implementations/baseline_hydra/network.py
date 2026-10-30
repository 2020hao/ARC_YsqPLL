"""Independent paper-FD classifier. Decision score is a hinge margin."""

import numpy as np
from .transform import SparseScaler
from .features import fit_l1_hinge, paper_transform


class Model:
    def __init__(self, state=None, k=8, g=8, kernel_length=5, stride=1, transform_seed=0):
        self.transform_kwargs = dict(
            k=k, g=g, kernel_length=kernel_length, stride=stride, seed=transform_seed
        )
        self.feature_dim = 2 * (int(np.log2(999 / (kernel_length - 1))) + 1) * g * k
        self.state = None if state is None else {k: np.asarray(v).copy() for k, v in state.items()}

    @classmethod
    def load(cls, path, **kwargs):
        with np.load(path, allow_pickle=False) as s:
            return cls(dict(s), **kwargs)

    def fit(self, features, labels, train_indices, *, lambda_l1=0.001):
        import torch

        ids = np.asarray(train_indices, dtype=int)
        x = torch.as_tensor(np.asarray(features, dtype="float32"))
        scaler = SparseScaler()
        z = scaler.fit_transform(x[ids]).numpy()
        w, b, diagnostics = fit_l1_hinge(z, np.asarray(labels)[ids], lambda_l1=lambda_l1)
        self.state = dict(
            weight=w,
            intercept=np.asarray(b),
            mu=scaler.mu.numpy(),
            sigma=scaler.sigma.numpy(),
            epsilon=scaler.epsilon.numpy(),
        )
        return diagnostics

    def decision_function(self, features):
        if self.state is None:
            raise ValueError("Fit or load the frozen SVM state first")
        x = np.asarray(features, dtype="float32")
        if x.ndim != 2 or x.shape[1] != self.feature_dim or not np.isfinite(x).all():
            raise ValueError("Finite [N,1024] author FD features required")
        import torch

        xx = torch.as_tensor(x).clamp(0).sqrt()
        mu = torch.as_tensor(self.state["mu"])
        sigma = torch.as_tensor(self.state["sigma"])
        z = (((xx - mu) * (xx != 0)) / sigma).numpy()
        return z @ self.state["weight"] + self.state["intercept"]

    def predict(self, features):
        return self.decision_function(features) >= 0

    def extract(self, waves, device="cpu"):
        return paper_transform(waves, device=device, **self.transform_kwargs)
