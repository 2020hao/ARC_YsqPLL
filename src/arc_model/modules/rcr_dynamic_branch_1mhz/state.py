"""Self-contained state bundles; learned state is bound to one fit identity."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from arc_model.contracts import FitIdentity

STATE_SCHEMA = "phase_dynamic_v2"


def indices_digest(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def groups_digest(groups):
    return hashlib.sha256(json.dumps(sorted(map(str, set(groups))), ensure_ascii=False).encode()).hexdigest()


def validate_state(state):
    required = {"keep", "mu", "sd", "coef", "pca_components", "pca_mean",
                "gate_threshold", "strength", "fusion", "calibration_slope",
                "calibration_intercept", "dimension"}
    if required - state.keys():
        raise ValueError(f"Incomplete phase_dynamic_v2 state: {sorted(required - state.keys())}")
    if int(state["dimension"]) != 36:
        raise ValueError("Current 1 MHz branch requires the explicit 36-feature schema")
    keep = np.asarray(state["keep"])
    n = len(keep)
    if keep.ndim != 1 or not np.issubdtype(keep.dtype, np.integer):
        raise ValueError("Integer feature column indices required")
    if not n or len(np.unique(keep)) != n or keep.min() < 0 or keep.max() >= 36:
        raise ValueError("Invalid selected columns")
    for key in ("mu", "sd", "coef", "pca_mean"):
        if np.asarray(state[key]).shape != (n,):
            raise ValueError(f"Invalid shape for {key}")
    pca = np.asarray(state["pca_components"])
    if pca.ndim != 2 or pca.shape[1] != n or not pca.shape[0]:
        raise ValueError("PCA shape does not match feature selection")
    for key, value in state.items():
        arr = np.asarray(value)
        if arr.dtype.kind in "biufc" and not np.isfinite(arr).all():
            raise ValueError(f"Non-finite fitted state: {key}")
    if (np.asarray(state["sd"]) <= 0).any() or float(state["gate_threshold"]) <= 0:
        raise ValueError("Positive fitted scales and gate threshold required")
    if float(state["calibration_slope"]) <= 0:
        raise ValueError("Positive score calibration required")
    fusion = str(state["fusion"])
    if fusion not in ("old_residual", "whole_path"):
        raise ValueError("Unknown fusion schema")
    strength = float(state["strength"])
    if strength < 0 or (fusion == "whole_path" and strength > 1):
        raise ValueError("Whole-path strength must be bounded to [0,1]")


def save_bundle(path, state, identity):
    validate_state(state)
    if identity is None:
        raise ValueError("A parent/data/split fit identity is required for persistence")
    metadata = dict(schema_version=2, state_schema=STATE_SCHEMA, identity=asdict(identity),
                    identity_sha256=identity.digest, performance_status="implemented_unvalidated")
    with Path(path).open("wb") as stream:
        np.savez(stream, **state, _metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))


def load_bundle(path, expected_identity=None):
    with np.load(path, allow_pickle=False) as bundle:
        if "_metadata_json" not in bundle:
            raise ValueError("Missing bundle identity; use legacy Correction for historical states")
        metadata = json.loads(str(bundle["_metadata_json"]))
        state = {key: bundle[key].copy() for key in bundle.files if key != "_metadata_json"}
    if metadata.get("state_schema") != STATE_SCHEMA or metadata.get("schema_version") != 2:
        raise ValueError("Unsupported branch state version")
    identity = FitIdentity(**metadata["identity"])
    if identity.digest != metadata["identity_sha256"]:
        raise ValueError("Corrupt identity metadata")
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("Wrong model/checkpoint/data/split/seed identity")
    validate_state(state)
    return state, identity
