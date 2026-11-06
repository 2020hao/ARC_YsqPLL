"""Paper FD HYDRA configuration and the published L1 hinge objective.

The random transform/scaler are the independently maintained HYDRA definitions.
HiGHS solves the same convex objective; it is not MATLAB SpaRSA bitwise replay.
"""

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linprog
from ._source_dual_hydra_transform import Hydra


def paper_transform(
    waves, device="cpu", batch_size=128, *, k=8, g=8, kernel_length=5, stride=1, seed=0
):
    x = np.asarray(waves, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 1000 or not np.isfinite(x).all():
        raise ValueError("Finite [N,1000] 50 kHz nominal 20 ms windows required")
    h = Hydra(1000, k=k, g=g, kernel_length=kernel_length, stride=stride, divisor=1, seed=seed)
    w = h.W.to(device)
    result = []
    with torch.no_grad():
        for a in range(0, len(x), batch_size):
            b = torch.as_tensor(x[a : a + batch_size, None], device=device)
            parts = []
            for j in range(h.num_dilations):
                c = torch.nn.functional.conv1d(
                    b,
                    w[j, 0],
                    dilation=int(h.dilations[j]),
                    padding=int(h.paddings[j]),
                    stride=stride,
                ).view(len(b), h.h, h.k, -1)
                maximum, imax = c.max(2)
                _, imin = c.min(2)
                high = torch.zeros(len(b), h.h, h.k, device=device)
                low = torch.zeros_like(high)
                high.scatter_add_(-1, imax, maximum)
                low.scatter_add_(-1, imin, torch.ones_like(maximum))
                parts.extend([high, low])
            result.append(torch.cat(parts, 1).reshape(len(b), -1).cpu().numpy())
    return np.concatenate(result)


def fit_l1_hinge(x, y, lambda_l1=0.001, time_limit=900):
    """min mean hinge(y*(Xw+b)) + lambda*|w|; intercept unpenalized."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both binary classes are required")
    sign = 2 * y - 1
    n, d = x.shape
    a = sparse.csr_matrix(-sign[:, None] * x)
    constraints = sparse.hstack(
        [a, -a, sparse.csr_matrix(-sign[:, None]), -sparse.eye(n)], format="csr"
    )
    cost = np.r_[np.full(2 * d, lambda_l1), 0.0, np.full(n, 1 / n)]
    r = linprog(
        cost,
        A_ub=constraints,
        b_ub=-np.ones(n),
        bounds=[(0, None)] * (2 * d) + [(None, None)] + [(0, None)] * n,
        method="highs",
        options={"time_limit": time_limit},
    )
    if not r.success:
        raise RuntimeError(f"L1 hinge solver failed: {r.message}")
    w = r.x[:d] - r.x[d : 2 * d]
    b = r.x[2 * d]
    objective = np.maximum(0.0, 1 - sign * (x @ w + b)).mean() + lambda_l1 * abs(w).sum()
    if abs(objective - r.fun) > 1e-6:
        raise ValueError("Primal objective replay failed")
    return (
        w,
        b,
        dict(
            objective=float(objective),
            solver_objective=float(r.fun),
            solver_status=r.message,
            iterations=int(r.nit),
            max_constraint_violation=float(max(0.0, -r.ineqlin.residual.min())),
            nonzero_weights=int((abs(w) > 1e-8).sum()),
        ),
    )
