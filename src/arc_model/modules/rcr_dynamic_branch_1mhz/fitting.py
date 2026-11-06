"""Independent phase-dynamic correction v2, exploratory candidate.

Keeps the historical CPU80-epoch zero-initialized linear fitting recipe, and
adds an entire-path bounded blend whose zero-strength state is the raw parent.
No imports from another model package and no changes to historical modules.
"""
from __future__ import annotations
import numpy as np
import torch
from sklearn.decomposition import PCA

ALPHAS=(0.,.1,.25,.5,1.,1.5,2.)
LAMBDAS=(0.,.1,.25,.5,.75,1.)
QUANTILES=(.75,.9,.95,.975)

def fit(
    x,
    labels,
    offset,
    indices,
    columns,
    seed,
    *,
    device="cpu",
    epochs=80,
    lr=0.002,
    weight_decay=0.002,
    batch_size=256,
    l1_mean_penalty=0.0002,
):
    """Train a zero-initialized linear correction against fixed OOF logits.

    Preserve historical float32 indexing/reduction order for numerical replay.
    All fitted statistics and PCA use only the supplied training indices.
    """
    mu = x[indices][:, columns].mean(0)
    sd = np.maximum(x[indices][:, columns].std(0), 1e-6)
    z = (x[:, columns] - mu) / sd
    torch.manual_seed(seed + 501)
    model = torch.nn.Linear(len(columns), 1, bias=False).to(device)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator(device=device).manual_seed(seed + 601)
    xx = torch.tensor(z, dtype=torch.float32, device=device)
    base = torch.tensor(offset, dtype=torch.float32, device=device)
    yy = torch.tensor(labels, dtype=torch.float32, device=device)
    ids = torch.tensor(indices, device=device)
    for _ in range(epochs):
        for j in ids[torch.randperm(len(ids), generator=generator, device=device)].split(
            batch_size
        ):
            optimizer.zero_grad(set_to_none=True)
            delta = model(xx[j]).flatten()
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(base[j] + delta, yy[j])
                + l1_mean_penalty * model.weight.abs().mean()
            )
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        delta = model(xx).flatten().cpu().numpy()
        coef = model.weight.flatten().cpu().numpy()
    normal = indices[labels[indices] == 0]
    pca = PCA(n_components=min(6, len(columns), len(normal) - 1), random_state=seed).fit(z[normal])
    scores = pca.transform(z)
    distance = np.sqrt(np.mean(scores * scores, axis=1))
    state = dict(
        keep=np.asarray(columns),
        mu=mu,
        sd=sd,
        coef=coef,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        fit_indices=indices,
    )
    return delta, distance, state


def replay(state, x):
    z = (x[:, state["keep"]] - state["mu"]) / state["sd"]
    delta = z.astype("float32") @ state["coef"]
    # Match sklearn PCA.transform's uncentred-input reduction order.
    score = z @ state["pca_components"].T - state["pca_mean"] @ state["pca_components"].T
    return delta, np.sqrt(np.mean(score * score, axis=1))



