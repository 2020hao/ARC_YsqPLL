"""Baseline adapters preserve bypass and expose a separate shared branch."""
import importlib

import numpy as np
import pytest
import torch
from arc_model.contracts import FitIdentity


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def arguments(family):
    if family == "arc_fenet":
        return (torch.randn(2, 3, 100, 100),)
    if family == "arc_fusion":
        return torch.randn(2, 1, 3, 250), torch.randn(2, 1, 32)
    return (np.random.default_rng(7).normal(size=(2, 1024)).astype("float32"),)


@pytest.mark.parametrize("family", ["arc_fenet", "hydra", "arc_fusion"])
def test_explicit_shared_branch_and_bit_exact_disable(family):
    package = "arc_model.model_instances." + family + "_dynamic_correction_1m"
    model = importlib.import_module(package).create_model()
    adapter = importlib.import_module(package + ".inference").predict_with_branch
    args = arguments(family)
    if family == "hydra":
        model.state = dict(weight=np.ones(1024), intercept=np.asarray(.125), mu=np.zeros(1024, dtype="float32"), sigma=np.ones(1024, dtype="float32"), epsilon=np.ones(1024))
        original = model.decision_function(*args)
    else:
        model.eval()
        with torch.no_grad():
            original = model(*args)

    class OffsetBranch:
        identity = FitIdentity(model_id=family + "_dynamic_correction_1m",
                               parent_checkpoint_sha256="parent", dataset_id="self462",
                               split_sha256="fold0", fit_indices_sha256="train_rows",
                               train_groups_sha256="train_groups", seed=0,
                               score_kind="svm_margin" if family == "hydra" else "logit_margin")

        def predict(self, raw, dynamic, **kwargs):
            assert kwargs["enabled"] is True
            assert kwargs["score_kind"] == ("svm_margin" if family == "hydra" else "logit_margin")
            assert kwargs["expected_identity"] == self.identity
            return raw + .5

    # Transformer eval fast paths depend on grad mode; compare the same mode.
    with torch.no_grad():
        bypass = adapter(model, *args, branch=object(), enabled=False)
    if isinstance(original, np.ndarray):
        np.testing.assert_array_equal(bypass, original)
        np.testing.assert_array_equal(adapter(model, *args, branch=OffsetBranch(), dynamic=np.zeros((2, 36)), enabled=True,
                                              expected_identity=OffsetBranch.identity), original + .5)
    else:
        torch.testing.assert_close(bypass, original, atol=0, rtol=0)
        with torch.no_grad():
            corrected = adapter(model, *args, branch=OffsetBranch(), dynamic=np.zeros((2, 36)), enabled=True,
                                expected_identity=OffsetBranch.identity)
        torch.testing.assert_close(corrected[:, 0], original[:, 0], atol=0, rtol=0)
        torch.testing.assert_close(corrected[:, 1] - corrected[:, 0], original[:, 1] - original[:, 0] + .5)


@pytest.mark.parametrize("family", ["arc_fenet", "arc_fusion"])
@pytest.mark.parametrize("suffix", ["dynamic_correction_1m", "phase_aligned_input"])
def test_local_neural_backbone_batch_can_backpropagate(family, suffix):
    model = importlib.import_module("arc_model.model_instances." + family + "_" + suffix).create_model()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(*arguments(family)), torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    optimizer.step()


@pytest.mark.parametrize("suffix", ["dynamic_correction_1m", "phase_aligned_input"])
def test_local_hydra_small_fit_retains_margin_classifier(suffix):
    model = importlib.import_module("arc_model.model_instances.hydra_" + suffix).create_model()
    features = np.abs(np.random.default_rng(88).normal(size=(6, 1024))).astype("float32")
    diagnostics = model.fit(features, np.array([0, 1, 0, 1, 0, 1]), np.array([0, 1, 2, 3]))
    assert isinstance(diagnostics, dict)
    assert model.state is not None
    assert np.isfinite(model.decision_function(features)).all()
