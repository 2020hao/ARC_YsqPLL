"""Fixed-instance migration must preserve source numerics and independence."""
import ast
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ROOT / "src/arc_model/model_instances"
INVENTORY = json.loads((PACKAGES / "inventory.json").read_text())


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def source_and_migrated(spec):
    original_class = getattr(importlib.import_module(spec["module"]), spec["cls"])
    local_class = importlib.import_module("arc_model.model_instances." + spec["id"]).create_model
    torch.manual_seed(441)
    original = original_class(**spec["kwargs"])
    source_rng = torch.get_rng_state().clone()
    torch.manual_seed(441)
    migrated = local_class()
    assert torch.equal(source_rng, torch.get_rng_state()), spec["id"]
    return original, migrated


def input_arguments(spec):
    generator = torch.Generator().manual_seed(218)
    if spec["role"] != "direct_classifier":
        if spec["source_package"].endswith("arc_fenet"):
            return (torch.randn(2, 3, 100, 100, generator=generator),)
        if spec["source_package"].endswith("arc_fusion"):
            return (torch.randn(2, 1, 3, 250, generator=generator), torch.randn(2, 1, 32, generator=generator))
        return (np.random.default_rng(19).normal(size=(2, 1024)).astype("float32"),)
    x = torch.randn(2, 9, spec["channels"], 64, generator=generator)
    mask = torch.ones(2, 9)
    mask[0, :3] = 0
    return x, mask


@pytest.mark.parametrize("spec", INVENTORY, ids=lambda row: row["id"])
def test_same_initialization_keys_buffers_and_forward(spec):
    original, migrated = source_and_migrated(spec)
    args = input_arguments(spec)
    if not isinstance(original, torch.nn.Module):
        rng = np.random.default_rng(82)
        state = dict(weight=rng.normal(size=1024), intercept=np.asarray(0.125),
                     mu=np.zeros(1024, dtype="float32"), sigma=np.ones(1024, dtype="float32"), epsilon=np.ones(1024, dtype="float32"))
        original.state = state
        migrated.state = {key: value.copy() for key, value in state.items()}
        np.testing.assert_array_equal(original.decision_function(*args), migrated.decision_function(*args))
        return
    left, right = original.state_dict(), migrated.state_dict()
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(left[name], right[name], atol=0, rtol=0, msg=name)
    original.eval()
    migrated.eval()
    with torch.inference_mode():
        torch.testing.assert_close(original(*args), migrated(*args), atol=0, rtol=0)
    migrated.load_state_dict(left, strict=True)


@pytest.mark.parametrize("spec", [row for row in INVENTORY if row.get("current")], ids=lambda row: row["id"])
def test_current_instance_gradient_parity(spec):
    original, migrated = source_and_migrated(spec)
    original.eval()
    migrated.eval()
    x, mask = input_arguments(spec)
    left = x.clone().requires_grad_()
    right = x.clone().requires_grad_()
    original(left, mask).sum().backward()
    migrated(right, mask).sum().backward()
    torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
    for (name, p), (other, q) in zip(original.named_parameters(), migrated.named_parameters()):
        assert name == other
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)


def test_instances_have_complete_local_interfaces_and_no_peer_imports():
    required = {"network.py", "data_loader.py", "preprocessing.py", "training.py", "inference.py", "model.yaml", "README.md", "provenance.json", "configuration.py"}
    for row in INVENTORY:
        package = PACKAGES / row["id"]
        assert required.issubset({p.name for p in package.iterdir()}), row["id"]
        cfg = yaml.safe_load((package / "model.yaml").read_text())
        assert cfg["model_id"] == row["id"]
        assert cfg["factory"] == "arc_model.model_instances." + row["id"] + ":create_model"
        assert cfg["legacy_state_compatible"] is True
        assert cfg["current_cohort_validated"] == bool(row.get("current"))
        assert cfg["evaluation"]["grouped_folds"] == {"self462": 4, "zenodo5501527": 5}
        for source in package.glob("*.py"):
            for node in ast.walk(ast.parse(source.read_text())):
                if isinstance(node, ast.ImportFrom):
                    imported = node.module or ""
                    assert not imported.startswith("arc_model.implementations"), (source, imported)
                    assert not imported.startswith("arc_model.model_instances"), (source, imported)
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith(("arc_model.implementations", "arc_model.model_instances")), (source, alias.name)


def test_factory_cannot_switch_structural_arm():
    from arc_model.model_instances.rcr_cnn_snn_original import create_model
    with pytest.raises(ValueError, match="fixed|fixes"):
        create_model(arm="dynamic")
