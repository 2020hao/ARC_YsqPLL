"""Independent HYDRA random transform and scaler.

Copied numerical definitions from the attributed author-derived implementation.
No other project model module is imported. Fixed random weights are CPU tensors
in the original API; call transform_numpy for the device-aware equivalent.
"""
# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb

# HYDRA: Competing Convolutional Kernels for Fast and Accurate Time Series Classification
# https://arxiv.org/abs/2203.13652

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Hydra(nn.Module):
    def __init__(self, input_length, k=8, g=64, kernel_length=9, stride=1, divisor=1, seed=None):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.k = k  # num kernels per group
        self.g = g  # num groups
        self.kernel_length = kernel_length  # kernel length = 9
        self.stride = stride

        self.max_exponent = np.log2((input_length - 1) / (self.kernel_length - 1))

        self.dilations = 2 ** torch.arange(int(self.max_exponent) + 1)
        self.num_dilations = len(self.dilations)

        self.paddings = torch.div(
            (self.kernel_length - 1) * self.dilations, 2, rounding_mode="floor"
        ).int()

        self.divisor = min(divisor, self.g)
        self.h = self.g // self.divisor

        self.W = torch.randn(
            self.num_dilations, self.divisor, self.k * self.h, 1, self.kernel_length
        )
        self.W = self.W - self.W.mean(-1, keepdims=True)
        self.W = self.W / self.W.abs().sum(-1, keepdims=True)

    # transform in batches of *batch_size*
    def batch(self, X, batch_size=256):
        num_examples = X.shape[0]
        if num_examples <= batch_size:
            return self(X)
        else:
            Z = []
            batches = torch.arange(num_examples).split(batch_size)
            for batch in batches:
                Z.append(self(X[batch]))
            return torch.cat(Z)

    def forward(self, X):
        num_examples = X.shape[0]

        if self.divisor > 1:
            diff_X = torch.diff(X)

        stride = self.stride
        Z = []
        for dilation_index in range(self.num_dilations):
            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            for diff_index in range(self.divisor):
                _Z = F.conv1d(
                    X if diff_index == 0 else diff_X,
                    self.W[dilation_index, diff_index],
                    dilation=d,
                    padding=p,
                    stride=stride,
                ).view(num_examples, self.h, self.k, -1)

                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(num_examples, self.h, self.k)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(num_examples, self.h, self.k)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(num_examples, -1)

        return Z


class SparseScaler:
    def __init__(self, mask=True, exponent=4):
        self.mask = mask
        self.exponent = exponent

        self.fitted = False

    def fit(self, X):
        assert not self.fitted, "Already fitted."

        X = X.clamp(0).sqrt()

        self.epsilon = (X == 0).float().mean(0) ** self.exponent + 1e-8

        self.mu = X.mean(0)
        self.sigma = X.std(0) + self.epsilon

        self.fitted = True

    def transform(self, X):
        assert self.fitted, "Not fitted."

        X = X.clamp(0).sqrt()

        if self.mask:
            return ((X - self.mu) * (X != 0)) / self.sigma
        else:
            return (X - self.mu) / self.sigma

    def fit_transform(self, X):
        self.fit(X)

        return self.transform(X)


def transform_numpy(waves, seed=0, device="cpu", batch_size=128):
    """Device-aware transform, identical seeded weights and reductions to Hydra."""
    waves = np.asarray(waves, dtype=np.float32)
    if waves.ndim != 2 or waves.shape[1] != 1000 or not np.isfinite(waves).all():
        raise ValueError("Finite [N,1000] input required")
    if len(waves) == 0:
        raise ValueError("Nonempty waves required")
    h = Hydra(1000, k=8, g=4, kernel_length=9, stride=2, divisor=1, seed=seed)
    weights = h.W.to(device)
    result = []
    with torch.no_grad():
        for start in range(0, len(waves), batch_size):
            x = torch.tensor(waves[start : start + batch_size, None], device=device)
            z = []
            for j in range(h.num_dilations):
                conv = F.conv1d(
                    x,
                    weights[j, 0],
                    dilation=int(h.dilations[j]),
                    padding=int(h.paddings[j]),
                    stride=h.stride,
                )
                conv = conv.reshape(len(x), h.h, h.k, -1)
                vmax, imax = conv.max(2)
                _, imin = conv.min(2)
                a = torch.zeros((len(x), h.h, h.k), device=device)
                b = torch.zeros_like(a)
                a.scatter_add_(-1, imax, vmax)
                b.scatter_add_(-1, imin, torch.ones_like(vmax))
                z.extend((a, b))
            result.append(torch.cat(z, 1).flatten(1).cpu().numpy())
    return np.concatenate(result)


def make_classifier(width):
    """Local neural HYDRA classifier, no auxiliary feature branch."""
    return nn.Linear(width, 2)
