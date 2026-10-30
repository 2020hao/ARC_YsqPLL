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


def fit_from_nested_oof(
    x, y, oof_offset, groups, train_indices, receipts, *,
    selection_offset, selection_fit, selection_validation, selection_base_train,
    baseline_id, score_kind, seed=0, device="cpu",
):
    """Choose strength on groups unseen by both selection backbone and head.

    ``selection_offset`` comes from one A-trained backbone; selection B/C
    groups are both absent from A. Final fitting uses complete honest inner OOF.
    Calibration must be fitted on B for selection, on outer-train OOF for refit.
    """
    config = load_config()
    if baseline_id != config["parent_baseline_id"]:
        raise ValueError("Wrong frozen parent")
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    tr = np.asarray(train_indices, dtype=int)
    b, c, a = [np.asarray(v, dtype=int) for v in
               (selection_fit, selection_validation, selection_base_train)]
    if x.shape != (len(y), 36):
        raise ValueError("Shared finite36 features required")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite features")
    cover = np.zeros(len(y), dtype=int)
    for receipt in receipts:
        fit, val = [np.asarray(receipt[k], dtype=int) for k in ("train", "validation")]
        if not np.isin(np.r_[fit, val], tr).all() or np.intersect1d(fit, val).size:
            raise ValueError("Invalid inner OOF rows")
        if set(groups[fit]) & set(groups[val]):
            raise ValueError("Inner OOF group leakage")
        cover[val] += 1
    if not (cover[tr] == 1).all() or cover.sum() != len(tr):
        raise ValueError("Incomplete or repeated honest OOF coverage")
    if not np.isin(np.r_[a, b, c], tr).all():
        raise ValueError("Selection uses outer test rows")
    for left, right in [(a,b), (a,c), (b,c)]:
        if np.intersect1d(left, right).size or set(groups[left]) & set(groups[right]):
            raise ValueError("Backbone/head/selection groups must be isolated")
    if not len(b) or not len(c):
        raise ValueError("Nonempty head fitting/selection groups required")
    if not np.isfinite(np.asarray(oof_offset)[tr]).all() or not np.isfinite(np.asarray(selection_offset)[np.r_[b,c]]).all():
        raise ValueError("Required training logits are missing")
    raw = config["feature_module"]["training"]
    settings = dict(epochs=raw["epochs"], lr=raw["lr"], weight_decay=raw["wd"],
                    batch_size=raw["batch_size"], l1_mean_penalty=raw["l1_mean_penalty"])
    columns = np.arange(36)
    delta, distance, _ = method.fit(x, y, selection_offset, b, columns, seed, device=device, **settings)
    (alpha, quantile), trace = method.calibrate(y, groups, selection_offset, delta, distance, b, c)
    _, distance, state = method.fit(x, y, oof_offset, tr, columns, seed+31, device=device, **settings)
    normal = tr[y[tr] == 0]
    state.update(alpha=np.asarray(alpha),
                 gate_threshold=np.asarray(max(float(np.quantile(distance[normal], quantile)),1e-6)),
                 baseline_id=np.asarray(baseline_id), score_kind=np.asarray(score_kind),
                 selection_fit_indices=b, selection_validation_indices=c,
                 selection_base_fit_indices=a, selected_quantile=np.asarray(quantile))
    return state, trace
