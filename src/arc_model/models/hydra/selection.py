"""Training-only grouped inner validation for HYDRA regularization."""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from .classifier import train
from .preprocessing import Features

ALPHAS = [0.001, 0.0046415888, 0.0215443469, 0.1]


def record_scores(ids, scores, rid):
    rr = np.unique(rid[ids])
    return rr, np.array([np.mean(scores[rid[ids] == r]) for r in rr])


def select_alpha(F, P_, y, rid, ledger, tr_records, method, seed, group="appliance"):
    yr = np.array([ledger[i]["label"] for i in tr_records])
    gr = np.array([ledger[i][group] for i in tr_records])
    splits = list(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=11).split(tr_records, yr, gr)
    )
    scores = np.zeros((len(ALPHAS), len(splits)))
    for j, (a, b) in enumerate(splits):
        tr = np.flatnonzero(np.isin(rid, tr_records[a]))
        va = np.flatnonzero(np.isin(rid, tr_records[b]))
        tx = Features(F, P_, y, method).fit(tr)
        xx = tx.transform(tr)
        vv = tx.transform(va)
        for k, alpha in enumerate(ALPHAS):
            clf, _ = train(xx, y[tr], alpha, seed)
            rr, ss = record_scores(va, clf.decision_function(vv), rid)
            scores[k, j] = balanced_accuracy_score([ledger[i]["label"] for i in rr], ss >= 0)
    mean = scores.mean(axis=1)
    # Deterministic tie break: stronger regularisation.
    best = np.flatnonzero(mean >= mean.max() - 1e-12)[-1]
    return ALPHAS[best], float(mean[best])
