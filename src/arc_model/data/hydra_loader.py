"""Independent HydraLoader for one prepared continuous-current cohort.

Only standard/numpy/pandas/torch dependencies; no common project loader.
This loads model-specific prepared arrays, not raw BIN/DAT or inferred labels.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch


class HydraLoader:
    def __init__(self, root):
        self.root = Path(root)
        self.table = pd.read_csv(self.root / "cycles.csv")
        self.splits = json.loads((self.root / "splits.json").read_text())
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
        self.wave = np.load(self.root / "wave.npy", mmap_mode="r")
        if self.wave.shape != (len(self.table), 1000):
            raise ValueError("HYDRA requires wave.npy [N,1000]")
        self.eligible_indices = np.unique(
            np.concatenate([np.asarray(s["test"], dtype=np.int64) for s in self.splits])
        )
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
        x = np.array(self.wave[ids], dtype=np.float32, copy=True)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite current inputs")
        return {
            "wave": torch.from_numpy(x),
            "label": torch.from_numpy(self.labels[ids]),
            "indices": ids.copy(),
        }
