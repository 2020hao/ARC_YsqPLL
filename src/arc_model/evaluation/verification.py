"""Compare reloaded predictions with stored window decisions and record ordering."""

from pathlib import Path
import json
import numpy as np
from arc_model.inference.predictor import predict_checkpoint
from arc_model.utils.io import sha256


def verify_checkpoint(path, dataset, device="cpu", legacy_indices=False, tolerance=1e-6):
    path = Path(path)
    meta = json.loads(path.read_text())
    if meta["protocol"] != "leave_appliance_out":
        raise ValueError("Only main-cohort artifacts are verifiable here")
    saved = np.load(path.with_name(path.stem + "_windows.npz"))
    indices = saved["indices"]
    if legacy_indices:
        mapping = {
            int(old): new for new, old in enumerate(dataset.array("original_window_indices"))
        }
        indices = np.array([mapping[int(i)] for i in indices])
    scores, _, _ = predict_checkpoint(path, dataset, indices, device)
    expected = saved["scores" if meta["model"] == "HYDRA" else "probabilities"]
    if not np.array_equal(dataset.record_ids[indices], saved["record_ids"]):
        raise ValueError("Record ordering changed")
    if not np.array_equal(dataset.labels[indices], saved["labels"]):
        raise ValueError("Window labels changed")
    error = float(np.max(np.abs(scores - expected)))
    return dict(
        checkpoint=str(path),
        model=meta["model"],
        variant=meta["variant"],
        mode=meta["mode"],
        seed=meta["seed"],
        fold=meta["fold_index"],
        windows=len(indices),
        max_abs_error=error,
        tolerance=tolerance,
        passed=bool(error <= tolerance),
    )


def audit_inputs(dataset, manifests, author_bundle, vendor):
    provenance = json.loads((Path(manifests) / "import_provenance.json").read_text())
    arrays = {}
    for name, info in provenance["arrays"].items():
        a = dataset.array(name)
        arrays[name] = dict(
            shape=list(a.shape),
            dtype=str(a.dtype),
            hash_matches=sha256(dataset.root / (name + ".npy")) == info["sha256"],
            finite=bool(np.isfinite(a).all()),
        )
    expected = "7d217eb7cae695fd6150129e2df772c031a530bd3bc36a719d9d0dc69215bd36"
    author_matches = sha256(vendor) == sha256(Path(author_bundle) / "hydra_class.py") == expected
    return dict(
        records=len(dataset.records),
        windows=len(dataset.record_ids),
        folds=len(dataset.splits),
        real_appliances_only=all(r["cohort"] == "main" for r in dataset.records),
        author_hydra_sha256=sha256(vendor),
        author_hydra_unchanged=author_matches,
        arrays=arrays,
        passed=author_matches and all(a["hash_matches"] and a["finite"] for a in arrays.values()),
    )
