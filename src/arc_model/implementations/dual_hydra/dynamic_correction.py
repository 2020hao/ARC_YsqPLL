"""Portable causal-dynamic residual correction; no dataset/model paths.

This is a disclosed restoration of the existing C0 algorithm, not a new PLL
or a claim that a new 8D static descriptor can inherit C0's results.
"""

import hashlib
import numpy as np
import torch
from scipy.special import expit
from sklearn.decomposition import PCA

ALPHAS = (0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0)
QUANTILES = (0.75, 0.9, 0.95, 0.975)


def feature_names():
    return tuple(
        f"lag{lag}_band{band}_{stat}"
        for lag in (1, 2, 4, 8)
        for band in (0, 1)
        for stat in ("cosine", "MAD", "JS", "peak_change")
    ) + tuple(f"history_lag{lag}_available" for lag in (1, 2, 4, 8))


def select_columns(x, indices, labels, groups):
    effects = []
    for group in np.unique(groups[indices]):
        ix = indices[groups[indices] == group]
        if len(np.unique(labels[ix])) == 2:
            effects.append(x[ix[labels[ix] == 1]].mean(0) - x[ix[labels[ix] == 0]].mean(0))
    effects = np.stack(effects)
    sign = np.sign(np.median(effects, 0))
    agree = np.mean(np.sign(effects) == sign, 0)
    strength = np.median(np.abs(effects), 0)
    candidates = np.where((agree >= 0.70) & (strength >= 1e-6))[0]
    if len(candidates) == 0:
        candidates = np.argsort(strength)[-6:]
    return candidates[np.argsort(strength[candidates])[-24:]]


def calibration_split(groups, train, fold):
    unique = sorted(set(groups[train]))
    calibration = [
        g
        for g in unique
        if int.from_bytes(hashlib.sha256(f"rcpclhc|{fold}|{g}".encode()).digest()[:4], "little") % 4
        == 0
    ]
    if len(calibration) < 2:
        calibration = unique[-2:]
    return train[~np.isin(groups[train], calibration)], train[np.isin(groups[train], calibration)]


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


def weights(distance, normal, alpha, quantile):
    threshold = max(float(np.quantile(distance[normal], quantile)), 1e-6)
    return alpha * expit(-(distance - threshold) / (0.15 * threshold + 1e-8))


def cbacc(labels, prediction, groups, indices):
    values = []
    for group in np.unique(groups[indices]):
        ix = indices[groups[indices] == group]
        y = labels[ix]
        p = prediction[ix]
        if (y == 0).any() and (y == 1).any():
            values.append(0.5 * (p[y == 1].mean() + (1 - p[y == 0]).mean()))
    return float(np.mean(values))


def fpr(labels, prediction, indices):
    normal = indices[labels[indices] == 0]
    return float(prediction[normal].mean())


def calibrate(labels, groups, offset, delta, distance, fit_indices, calibration_indices):
    baseline = (offset >= 0).astype(int)
    normal = fit_indices[labels[fit_indices] == 0]
    baseline_fpr = fpr(labels, baseline, calibration_indices)
    best = (
        (cbacc(labels, baseline, groups, calibration_indices), -baseline_fpr, 0.0, -1.0),
        (0.0, 0.9),
    )
    trace = []
    for alpha in ALPHAS:
        for quantile in QUANTILES:
            p = (offset + weights(distance, normal, alpha, quantile) * delta >= 0).astype(int)
            false_positive = fpr(labels, p, calibration_indices)
            score = cbacc(labels, p, groups, calibration_indices)
            eligible = false_positive <= baseline_fpr + 0.0200001
            rank = (score, -false_positive, -alpha, -quantile)
            trace.append(
                dict(
                    alpha=alpha,
                    quantile=quantile,
                    BAcc=score,
                    FPR=false_positive,
                    eligible=bool(eligible),
                )
            )
            if eligible and rank > best[0]:
                best = (rank, (alpha, quantile))
    return best[1], trace
