"""Checkpoint inference using only saved training state; never refit preprocessing."""

from pathlib import Path
import json
import numpy as np
import torch
from arc_model.models.factory import make_model
from arc_model.training.common import configure_runtime, to_tensors, predict_probabilities
from arc_model.training.preprocessing import apply_neural_state


def hydra_scores(raw, physical, mode, state):
    a = torch.from_numpy(np.array(raw, copy=True)).clamp(0).sqrt()
    a = ((a - torch.from_numpy(state["scaler_mu"])) * (a != 0)) / torch.from_numpy(
        state["scaler_sigma"]
    )
    x = a.numpy()
    if mode != "base":
        # Preserve the float64 path of the original HYDRA training transform.
        f = physical
        if 'feature_indices' in state and f.shape[-1] != len(state['feature_indices']):
            if f.shape[-1] != 42:
                raise ValueError('Feature input does not match saved selection')
            f = f[:, state['feature_indices']]
        if mode != "raw":
            z = (f - state["feature_med"]) / state["feature_mad"]
            f = np.sign(z) * np.log1p(np.abs(z))
        f = (f - state["feature_mean"]) / state["feature_std"]
        x = np.c_[x, f]
    return np.asarray(x, np.float64) @ state["coef"].ravel() + state["intercept"][0]


def predict_checkpoint(metadata, dataset, indices=None, device="cpu", threads=2):
    path = Path(metadata)
    meta = json.loads(path.read_text())
    split = meta.get('split', meta)
    if split.get("protocol") != "condition_grouped_cv5":
        raise ValueError("Only current condition-grouped checkpoints are accepted")
    expected = dataset.splits[meta['fold_index']]
    if set(split['test_records']) != set(expected['test_records']):
        raise ValueError('Checkpoint and dataset test partitions differ')
    if indices is None:
        indices = np.flatnonzero(np.isin(dataset.record_ids, split["test_records"]))
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0 or indices.min() < 0 or indices.max() >= len(dataset.record_ids):
        raise ValueError("Invalid/empty prediction indices")
    stem = path.with_suffix("")
    name, variant, mode = meta["model"], meta["variant"], meta["mode"]
    dev = configure_runtime(device, threads)
    if name == "HYDRA":
        with np.load(stem.with_suffix(".npz")) as state:
            return (
                hydra_scores(
                    dataset.array(f"hydra_features_seed{meta['seed']}")[indices, variant],
                    dataset.physical(variant, mode, indices),
                    mode,
                    state,
                ),
                indices,
                meta,
            )
    with np.load(str(stem) + "_preprocessing.npz") as state:
        tensors = to_tensors(apply_neural_state(dataset, name, variant, mode, state, indices), dev)
    model = make_model(name, 0 if mode == "base" else int(tensors[2].shape[-1])).to(dev)
    model.load_state_dict(torch.load(stem.with_suffix(".pt"), map_location=dev, weights_only=True))
    return (
        predict_probabilities(model, *tensors, torch.arange(len(indices), device=dev)),
        indices,
        meta,
    )
