"""Local RF32 and train-only scaling; no external model component imports."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def fit_state(loader, train, seed):
    x = np.asarray(loader.temporal)
    sp = np.asarray(loader.spectrum)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=2,
    )
    rf.fit(sp[train], loader.labels[train])
    bins = np.sort(np.argsort(rf.feature_importances_)[-32:])
    a = sp[:, bins]
    return dict(
        fit_indices=np.asarray(train),
        harmonic_indices=bins,
        rf_importance=rf.feature_importances_,
        temporal_mean=x[train].mean((0, 3), keepdims=True),
        temporal_std=np.maximum(x[train].std((0, 3), keepdims=True), 1e-6),
        spectral_mean=a[train].mean(0),
        spectral_std=np.maximum(a[train].std(0), 1e-6),
    )


def apply_state(loader, state):
    x = (loader.temporal - state["temporal_mean"]) / state["temporal_std"]
    s = loader.spectrum[:, state["harmonic_indices"]]
    return x, ((s - state["spectral_mean"]) / state["spectral_std"])[:, None]
