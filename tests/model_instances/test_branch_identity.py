"""A fitted correction cannot silently move between models or evaluation folds."""
from dataclasses import replace
import hashlib
import importlib

import numpy as np
import pytest
import torch

from arc_model.contracts import FitIdentity
from arc_model.modules.rcr_dynamic_branch_1mhz import RCRDynamicBranch1MHz


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def identity(family, *, parent_hash="a" * 64):
    return FitIdentity(model_id=family + "_dynamic_correction_1m",
                       parent_checkpoint_sha256=parent_hash, dataset_id="self462",
                       split_sha256="fold0", fit_indices_sha256="train_rows",
                       train_groups_sha256="train_groups", seed=0,
                       score_kind="svm_margin" if family == "hydra" else "logit_margin")


def branch_state():
    return dict(keep=np.arange(36), mu=np.zeros(36), sd=np.ones(36), coef=np.ones(36) * .01,
                pca_components=np.eye(36)[:6], pca_mean=np.zeros(36), dimension=np.asarray(36),
                gate_threshold=np.asarray(1.), strength=np.asarray(.5), fusion=np.asarray("whole_path"),
                calibration_slope=np.asarray(1.), calibration_intercept=np.asarray(.05))


def prepare(family):
    package = "arc_model.model_instances." + family + "_dynamic_correction_1m"
    module = importlib.import_module(package + ".inference")
    model = importlib.import_module(package).create_model()
    if family == "hydra":
        model.state = dict(weight=np.ones(1024), intercept=np.asarray(.125), mu=np.zeros(1024, dtype="float32"),
                           sigma=np.ones(1024, dtype="float32"), epsilon=np.ones(1024))
        args = (np.ones((2, 1024), dtype="float32"),)
    elif family == "arc_fenet":
        model.eval()
        args = (torch.randn(2, 3, 100, 100),)
    else:
        model.eval()
        args = (torch.randn(2, 1, 3, 250), torch.randn(2, 1, 32))
    return module, model, args


@pytest.mark.parametrize("family", ["arc_fenet", "hydra", "arc_fusion"])
def test_enabled_adapter_requires_model_and_fold_bound_identity(family):
    module, model, args = prepare(family)
    expected = identity(family)
    branch = RCRDynamicBranch1MHz(branch_state(), expected)
    features = np.zeros((2, 36))
    with torch.no_grad():
        # Bypass accepts no branch state or identity at all.
        module.predict_with_branch(model, *args, branch=object(), enabled=False)
        with pytest.raises(ValueError, match="FitIdentity"):
            module.predict_with_branch(model, *args, branch=RCRDynamicBranch1MHz(branch_state()),
                                       dynamic=features, enabled=True, expected_identity=expected)
        with pytest.raises(ValueError, match="model_id"):
            wrong = RCRDynamicBranch1MHz(branch_state(), replace(expected, model_id="other_backbone"))
            module.predict_with_branch(model, *args, branch=wrong, dynamic=features,
                                       enabled=True, expected_identity=expected)
        with pytest.raises(ValueError, match="checkpoint"):
            module.predict_with_branch(model, *args, branch=branch, dynamic=features, enabled=True)
        for strength in (0., .5):
            state = branch_state()
            state["strength"] = np.asarray(strength)
            bound = RCRDynamicBranch1MHz(state, expected)
            for changed in [replace(expected, split_sha256="fold1"),
                            replace(expected, parent_checkpoint_sha256="b" * 64),
                            replace(expected, seed=1), replace(expected, dataset_id="zenodo5501527")]:
                with pytest.raises(ValueError, match="identity|Identity"):
                    module.predict_with_branch(model, *args, branch=bound, dynamic=features,
                                               enabled=True, expected_identity=changed)
        result = module.predict_with_branch(model, *args, branch=branch, dynamic=features,
                                            enabled=True, expected_identity=expected)
        assert np.isfinite(result if isinstance(result, np.ndarray) else result.numpy()).all()


@pytest.mark.parametrize("family", ["arc_fenet", "hydra", "arc_fusion"])
def test_loaded_checkpoint_receipt_binds_hash_and_full_identity(tmp_path, family):
    module, model, args = prepare(family)
    path = tmp_path / ("model.npz" if family == "hydra" else "model.pt")
    if family == "hydra":
        np.savez(path, **model.state)
    else:
        torch.save(model.state_dict(), path)
    expected = identity(family, parent_hash=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="checkpoint hash"):
        module.load_checkpoint(path, expected_identity=replace(expected, parent_checkpoint_sha256="bad_hash"))
    hash_only = module.load_checkpoint(path)
    with torch.no_grad(), pytest.raises(ValueError, match="full checkpoint receipt"):
        module.predict_with_branch(hash_only, *args, branch=RCRDynamicBranch1MHz(branch_state(), expected),
                                   dynamic=np.zeros((2, 36)), enabled=True)
    loaded = module.load_checkpoint(path, expected_identity=expected)
    assert loaded._parent_checkpoint_sha256 == expected.parent_checkpoint_sha256
    branch = RCRDynamicBranch1MHz(branch_state(), expected)
    with torch.no_grad():
        module.predict_with_branch(loaded, *args, branch=branch, dynamic=np.zeros((2, 36)), enabled=True)
        wrong = RCRDynamicBranch1MHz(branch_state(), replace(expected, split_sha256="wrong_fold"))
        with pytest.raises(ValueError, match="identity|Identity"):
            module.predict_with_branch(loaded, *args, branch=wrong, dynamic=np.zeros((2, 36)), enabled=True)
        with pytest.raises(ValueError, match="checkpoint hash"):
            module.predict_with_branch(loaded, *args, branch=branch, dynamic=np.zeros((2, 36)), enabled=True,
                                       expected_identity=replace(expected, parent_checkpoint_sha256="other_checkpoint"))
