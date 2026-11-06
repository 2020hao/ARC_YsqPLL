"""Each independent local trainer must perform a finite training update."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = json.loads((ROOT / "src/arc_model/model_instances/inventory.json").read_text())
DIRECT = [row for row in INVENTORY if row["role"] == "direct_classifier"]


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("spec", DIRECT, ids=lambda row: row["id"])
def test_local_train_batch_updates_parameters(spec):
    package = "arc_model.model_instances." + spec["id"]
    model = importlib.import_module(package).create_model()
    training = importlib.import_module(package + ".training")
    optimizer = training.make_optimizer(model)
    generator = torch.Generator().manual_seed(723)
    x = torch.randn(2, 9, spec["channels"], 64, generator=generator)
    mask = torch.ones(2, 9)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    value = training.train_batch(model, optimizer, x, mask, torch.tensor([0., 1.]), torch.ones(2),
                                 augmentation_generator=generator)
    assert np.isfinite(value)
    assert any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters())


@pytest.mark.parametrize("model_id", ["rcr_cnn_snn_peak_dynamic_augmented", "rcr_cnn_snn_peak_dynamic_bias_calibrated", "rcr_cnn_snn_peak_dynamic_affine_calibrated"])
def test_augmented_training_batch_preserves_original_cached_summary_numerics(model_id):
    from arc_model.implementations.dynamic_state_net import DynamicStateNet
    from arc_model.implementations.dynamic_state_net.summary import native_dynamic_summary
    from arc_model.implementations.phase_hierarchy_net.augmentation import augment_standardized

    package = "arc_model.model_instances." + model_id
    training = importlib.import_module(package + ".training")
    torch.manual_seed(138)
    original = DynamicStateNet("maxpool_dynamic")
    torch.manual_seed(138)
    migrated = importlib.import_module(package).create_model()
    rng = torch.Generator().manual_seed(78)
    x = torch.randn(2, 9, 5, 64, generator=rng)
    mask = torch.ones(2, 9)
    cached = native_dynamic_summary(x, mask)
    weights = torch.ones(2)
    labels = torch.tensor([0., 1.])
    left_optimizer = torch.optim.AdamW(original.parameters(), lr=.001, weight_decay=.001)
    right_optimizer = training.make_optimizer(migrated)
    left_rng = torch.Generator().manual_seed(228)
    right_rng = torch.Generator().manual_seed(228)
    original.train()
    changed = augment_standardized(x, original.input_mean, original.input_std, left_rng)
    logits, activity = original(changed, mask, return_activity=True, summary=cached)
    loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights).mean() + .001 * activity
    loss.backward()
    torch.nn.utils.clip_grad_norm_(original.parameters(), 1.)
    left_optimizer.step()
    migrated_loss = training.train_batch(migrated, right_optimizer, x, mask, labels, weights,
                                         summary=cached, augmentation_generator=right_rng)
    assert float(loss.detach()) == migrated_loss
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, migrated.state_dict()[key], atol=0, rtol=0, msg=key)
