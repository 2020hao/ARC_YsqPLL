"""Prepared MBWVD images; normalization is per-window, no dataset-fitted scaler."""

import numpy as np


def fit_state(loader, train, seed):
    return {"fit_indices": np.asarray(train)}


def apply_state(loader, state):
    return loader.rgb, None


def from_native_cycles(cycles):
    from .mbwvd import mbwvd
    from matplotlib import colormaps

    images = []
    for wave in np.asarray(cycles):
        image, _, _, _ = mbwvd(wave)
        lo, hi = float(image.min()), float(image.max())
        images.append(
            colormaps["jet"]((image - lo) / max(hi - lo, 1e-20))[..., :3].transpose(2, 0, 1)
        )
    return np.asarray(images, dtype="float32")
