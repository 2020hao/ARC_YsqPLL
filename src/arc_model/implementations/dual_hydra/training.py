"""Fit only the local residual branch from honest grouped OOF scores.

The backbone is trained independently by backbone_training.py. Caller supplies
record/condition-disjoint inner-OOF receipts; no in-sample replacement is allowed.
"""

import numpy as np
from . import dynamic_correction as method
from .configuration import load_config


def fit_from_oof(
    x,
    y,
    base_logit,
    groups,
    train_indices,
    receipts,
    *,
    fold,
    seed,
    baseline_id,
    score_kind,
    device="cpu",
):
    full_config = load_config()
    if baseline_id != full_config["parent_baseline_id"]:
        raise ValueError("Wrong frozen parent baseline")
    cfg = full_config["feature_module"]["training"]
    settings = dict(
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        weight_decay=cfg["wd"],
        batch_size=cfg["batch_size"],
        l1_mean_penalty=cfg["l1_mean_penalty"],
    )
    x = np.asarray(x)
    y = np.asarray(y)
    base_logit = np.asarray(base_logit)
    groups = np.asarray(groups)
    tr = np.asarray(train_indices, dtype=int)
    coverage = np.zeros(len(y), int)
    for r in receipts:
        fit = np.asarray(r["train"], dtype=int)
        val = np.asarray(r["validation"], dtype=int)
        if not np.isin(np.r_[fit, val], tr).all() or np.intersect1d(fit, val).size:
            raise ValueError("Invalid inner OOF rows")
        if set(groups[fit]) & set(groups[val]):
            raise ValueError("Inner OOF condition leakage")
        coverage[val] += 1
    if not (coverage[tr] == 1).all():
        raise ValueError("Every outer training row requires exactly one honest OOF score")
    if x.shape != (len(y), 36) or base_logit.shape != (len(y),):
        raise ValueError("Full-row36 dynamics and OOF scores required")
    if score_kind not in ["logit_raw", "calibrated_svm_logit"]:
        raise ValueError(
            "BCE correction needs declared logit convention; raw SVM margins are not probabilities"
        )
    a, b = method.calibration_split(groups, tr, fold)
    cols = np.arange(36)
    delta, distance, _ = method.fit(x, y, base_logit, a, cols, seed, device=device, **settings)
    (alpha, q), trace = method.calibrate(y, groups, base_logit, delta, distance, a, b)
    _, distance, state = method.fit(
        x, y, base_logit, tr, cols, seed + 31, device=device, **settings
    )
    normal = tr[y[tr] == 0]
    threshold = max(float(np.quantile(distance[normal], q)), 1e-6)
    state.update(
        alpha=np.asarray(alpha),
        gate_threshold=np.asarray(threshold),
        baseline_id=np.asarray(baseline_id),
        score_kind=np.asarray(score_kind),
    )
    return state, trace
