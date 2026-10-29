"""Fit RF/normalization only on outer training indices; reuse frozen state at inference."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from arc_model.features.enhancement import FeatureTransform


def fit_state(dataset, config, train, runseed):
    state = {}
    if config.model == "ArcFusionNet":
        x, spectrum = dataset.neural_inputs(config.model, config.variant)
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=runseed,
            n_jobs=2,
        )
        rf.fit(spectrum[train], dataset.labels[train])
        selected = np.sort(np.argsort(rf.feature_importances_)[-32:])
        s = spectrum[:, selected]
        state.update(
            harmonic_indices=selected,
            rf_importance=rf.feature_importances_,
            spectral_mean=s[train].mean(0),
            spectral_std=np.maximum(s[train].std(0), 1e-6),
            temporal_mean=x[train].mean(axis=(0, 3), keepdims=True),
            temporal_std=np.maximum(x[train].std(axis=(0, 3), keepdims=True), 1e-6),
        )
    if config.mode != "base":
        physical = dataset.physical(config.variant, config.mode)
        transform = FeatureTransform().fit(physical[train], dataset.labels[train], config.mode)
        state.update({f"feature_{k}": value for k, value in transform.state().items()})
        if hasattr(dataset, 'feature_indices'):
            state['feature_indices'] = np.asarray(dataset.feature_indices, dtype=np.int64)
    return state


def transform_physical(physical, mode, state):
    if mode == "base":
        return None
    a = physical
    if 'feature_indices' in state and a.shape[-1] != len(state['feature_indices']):
        if a.shape[-1] != 42:
            raise ValueError('Feature input does not match saved selection')
        a = a[:, state['feature_indices']]
    if mode != "raw":
        z = (a - state["feature_med"]) / state["feature_mad"]
        a = np.sign(z) * np.log1p(np.abs(z))
    return ((a - state["feature_mean"]) / state["feature_std"]).astype("float32")


def apply_neural_state(dataset, name, variant, mode, state, indices=None):
    x, s = dataset.neural_inputs(name, variant, indices)
    if name == "ArcFusionNet":
        x = (x - state["temporal_mean"]) / state["temporal_std"]
        s = ((s[:, state["harmonic_indices"]] - state["spectral_mean"]) / state["spectral_std"])[
            :, None
        ]
    f = (
        None
        if mode == "base"
        else transform_physical(dataset.physical(variant, mode, indices), mode, state)
    )
    return x, s, f
