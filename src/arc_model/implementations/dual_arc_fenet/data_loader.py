"""Independent ArcFENetLoader for one prepared continuous-current cohort.

Only standard/numpy/pandas/torch dependencies; no common project loader.
This loads model-specific prepared arrays, not raw BIN/DAT or inferred labels.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch


class ArcFENetLoader:
    def __init__(self, root, *, split_file=None, reviewed_indices=None):
        self.root = Path(root)
        self.table = pd.read_csv(self.root / "cycles.csv")
        self.splits = json.loads(
            (Path(split_file) if split_file is not None else self.root / "splits.json").read_text()
        )
        if reviewed_indices is None:
            raise ValueError(
                "Explicit current reviewed indices required; do not silently use historical extra-window splits"
            )
        self.reviewed = np.asarray(
            np.load(reviewed_indices)
            if isinstance(reviewed_indices, (str, Path))
            else reviewed_indices,
            dtype=np.int64,
        )
        required = {"sample_id", "label", "condition_key", "source_group", "record_id"}
        if not required.issubset(self.table.columns):
            raise ValueError("Missing sample labels or source/group identifiers")
        if self.table.sample_id.isna().any() or self.table.sample_id.duplicated().any():
            raise ValueError("Sample IDs must be nonempty and unique")
        if self.table[list(required)].isna().any().any():
            raise ValueError("Missing labels/group metadata")
        if not self.table.label.isin([0, 1]).all():
            raise ValueError("Only reviewed binary labels are supported")
        self.labels = self.table.label.to_numpy(dtype=np.int64)
        self.rgb = np.load(self.root / "rgb.npy", mmap_mode="r")
        if self.rgb.shape != (len(self.table), 3, 100, 100):
            raise ValueError("ArcFENet requires rgb.npy [N,3,100,100]")
        if (
            self.reviewed.ndim != 1
            or not len(self.reviewed)
            or len(np.unique(self.reviewed)) != len(self.reviewed)
            or self.reviewed.min() < 0
            or self.reviewed.max() >= len(self.table)
        ):
            raise ValueError("Invalid reviewed indices")
        self.eligible_indices = np.sort(self.reviewed)
        coverage = np.zeros(len(self.table), dtype=int)
        for fold in range(len(self.splits)):
            _, te = self.indices(fold)
            coverage[te] += 1
        if not np.all(coverage[self.eligible_indices] == 1) or np.any(coverage > 1):
            raise ValueError("Every eligible sample must appear in one outer test fold")

    def indices(self, fold):
        if not 0 <= fold < len(self.splits):
            raise ValueError("Unknown outer fold")
        split = self.splits[fold]
        arrays = []
        for key in ("train", "test"):
            ids = np.asarray(split[key])
            if ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids):
                raise ValueError("Nonempty integer split indices required")
            if ids.min() < 0 or ids.max() >= len(self.table) or len(np.unique(ids)) != len(ids):
                raise ValueError("Invalid or duplicate split indices")
            arrays.append(ids.astype(np.int64))
        tr, te = arrays
        tr = np.intersect1d(tr, self.reviewed)
        te = np.intersect1d(te, self.reviewed)
        if not np.array_equal(tr, np.setdiff1d(self.reviewed, te)):
            raise ValueError(
                "Training is not the exact reviewed complement of this outer test fold"
            )
        if np.intersect1d(tr, te).size:
            raise ValueError("Train/test row overlap")
        for field in ("condition_key", "source_group", "record_id"):
            if set(self.table.iloc[tr][field]) & set(self.table.iloc[te][field]):
                raise ValueError(f"{field} leakage")
        return tr, te

    def inputs(self, indices):
        ids = np.asarray(indices)
        if (
            ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or not len(ids)
            or ids.min() < 0
            or ids.max() >= len(self.table)
        ):
            raise ValueError("Nonempty integer row indices required")
        x = np.array(self.rgb[ids], dtype=np.float32, copy=True)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite RGB inputs")
        return {
            "image": torch.from_numpy(x),
            "label": torch.from_numpy(self.labels[ids]),
            "indices": ids.copy(),
        }


DataLoader = ArcFENetLoader


def from_project(dataset, project_root=None):
    import hashlib
    import yaml
    from .configuration import load_config

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[4]
    doc = yaml.safe_load((root / "configs/datasets.yaml").read_text())[dataset]
    split = root / doc["split_file"]
    reviewed = root / doc["reviewed_indices"]
    for p, key in [(split, "split_sha256"), (reviewed, "reviewed_sha256")]:
        if hashlib.sha256(p.read_bytes()).hexdigest() != doc[key]:
            raise ValueError("Frozen data/split identity changed: " + str(p))
    reference = load_config()["input_contract"]["reference"]
    prepared = root / doc["prepared_roots"][reference]
    if (
        hashlib.sha256((prepared / "cycles.csv").read_bytes()).hexdigest()
        != doc["prepared_metadata_sha256"][reference]
    ):
        raise ValueError("Prepared metadata identity mismatch")
    return DataLoader(prepared, split_file=split, reviewed_indices=reviewed)
