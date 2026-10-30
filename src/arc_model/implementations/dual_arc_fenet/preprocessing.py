"""Prepared MBWVD images; normalization is per-window, no dataset-fitted scaler."""

import numpy as np


def fit_state(loader, train, seed):
    return {"fit_indices": np.asarray(train)}


def apply_state(loader, state):
    return loader.rgb, None
