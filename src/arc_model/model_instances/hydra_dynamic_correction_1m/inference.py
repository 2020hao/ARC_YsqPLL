"""Local, identity-bound shared1MHz adapter; bypass preserves the original path."""
import hashlib
from pathlib import Path

from arc_model.contracts import FitIdentity
from .network import Model

MODEL_ID = 'hydra_dynamic_correction_1m'


def create_predictor(**kwargs):
    return Model(**kwargs)


def _checkpoint_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_identity(model, branch, expected_identity):
    identity = getattr(branch, "identity", None)
    if not isinstance(identity, FitIdentity):
        raise ValueError("The enabled branch requires its fitted FitIdentity")
    if identity.model_id != MODEL_ID:
        raise ValueError("Branch model_id does not match this independent instance")
    loaded_hash = getattr(model, "_parent_checkpoint_sha256", None)
    if expected_identity is None:
        expected_identity = getattr(model, "_branch_expected_identity", None)
    if expected_identity is None:
        raise ValueError("Pass expected_identity or load_checkpoint(expected_identity=...) with a full checkpoint receipt")
    if not isinstance(expected_identity, FitIdentity) or expected_identity.model_id != MODEL_ID:
        raise ValueError("Expected model identity does not match this instance")
    if loaded_hash is not None and expected_identity.parent_checkpoint_sha256 != loaded_hash:
        raise ValueError("Expected parent checkpoint hash differs from the loaded checkpoint receipt")
    return expected_identity


def _bind_checkpoint(model, path, expected_identity):
    digest = _checkpoint_hash(path)
    if expected_identity is not None:
        if not isinstance(expected_identity, FitIdentity) or expected_identity.model_id != MODEL_ID:
            raise ValueError("Expected model identity does not match this instance")
        if expected_identity.parent_checkpoint_sha256 != digest:
            raise ValueError("Expected parent checkpoint hash differs from the checkpoint file")
    model._parent_checkpoint_sha256 = digest
    model._branch_expected_identity = expected_identity
    return model


def load_checkpoint(path, *, expected_identity=None):
    """Load this classifier and retain its actual parent-file SHA256 receipt."""
    model = Model.load(path)
    return _bind_checkpoint(model, path, expected_identity)


def predict_with_branch(model, features, *, branch=None, dynamic=None, enabled=False,
                        expected_identity=None):
    margin = model.decision_function(features, enabled=False)
    if not enabled:
        return margin
    expected = _expected_identity(model, branch, expected_identity)
    return branch.predict(margin, dynamic, enabled=True, score_kind="svm_margin", threshold=0.0,
                          expected_identity=expected)
