"""Strict replay of all current folded checkpoints on the same CPU path."""
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
FOLDED = ROOT / "outputs/folded_calibration_20261105/runs"
CHECKPOINTS = sorted(FOLDED.glob("*/*/folded_affine_model.pt"))


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.integration
@pytest.mark.skipif(not CHECKPOINTS, reason="Private current-cohort checkpoints are local only")
@pytest.mark.parametrize("checkpoint", CHECKPOINTS, ids=lambda p: "/".join(p.parts[-3:-1]))
def test_saved_affine_checkpoint_replays_without_key_or_output_change(checkpoint):
    from arc_model.implementations.dynamic_state_net import DynamicStateNet
    from arc_model.model_instances.rcr_cnn_snn_peak_dynamic_affine_calibrated import create_model

    dataset = checkpoint.parent.parent.name
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    old = DynamicStateNet("maxpool_dynamic").eval()
    new = create_model().eval()
    old.load_state_dict(state, strict=True)
    new.load_state_dict(state, strict=True)
    prediction = np.load(checkpoint.parent / "folded_affine_predictions.npz", allow_pickle=False)
    indices = prediction["indices"][:8]
    if dataset == "self462":
        folder = ROOT / "outputs/self462_1m_baselines_20261105"
        raw = np.load(folder / "cnn_snn_2m_control.npy", mmap_mode="r")[indices]
        mask = torch.ones(len(indices), 9)
    else:
        folder = ROOT / "outputs/map_input_calibration_20261105/data/zenodo5501527"
        raw = np.load(folder / "realtime_raw.npy", mmap_mode="r")[indices, :, :5]
        mask = torch.from_numpy(np.array(np.load(folder / "realtime_mask.npy", mmap_mode="r")[indices], copy=True))
    x = torch.from_numpy(np.array(raw, dtype=np.float32, copy=True))
    x = (x - new.input_mean[None, None, :, None]) / new.input_std[None, None, :, None]
    with torch.inference_mode():
        old_logits = old(x, mask)
        new_logits = new(x, mask)
    torch.testing.assert_close(old_logits, new_logits, atol=0, rtol=0)
    np.testing.assert_array_equal(old_logits.numpy() >= 0, new_logits.numpy() >= 0)
