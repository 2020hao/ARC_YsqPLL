"""Local prepared-input loader; paths and cohort selection are explicit."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

class CycleDataLoader:
    def __init__(self, root, *, input_file, mask_file=None, keep_file=None, channels=5):
        self.root = Path(root)
        self.table = pd.read_csv(self.root / "cycles.csv")
        required = {"label", "condition_key", "record_id"}
        if not required.issubset(self.table) or self.table[list(required)].isna().any().any():
            raise ValueError("Reviewed labels, conditions and recording identities required")
        if not self.table.label.isin([0, 1]).all():
            raise ValueError("Binary reviewed labels required")
        self.inputs_array = np.load(self.root / input_file, mmap_mode="r")
        if self.inputs_array.ndim != 4 or self.inputs_array.shape[:2] != (len(self.table), 9) or self.inputs_array.shape[-1] != 64 or self.inputs_array.shape[2] < channels:
            raise ValueError("Expected model-native [N,9,C,64] input")
        self.channels = channels
        self.mask = np.load(self.root / mask_file, mmap_mode="r") if mask_file else np.ones((len(self.table), 9), dtype=np.float32)
        if self.mask.shape != (len(self.table), 9) or not np.isfinite(self.mask).all() or not np.isin(self.mask, (0, 1)).all():
            raise ValueError("Finite binary [N,9] history mask required")
        if keep_file is None and (self.root / "keep.npy").exists():
            raise ValueError("This prepared cohort has excluded candidates; pass keep_file explicitly")
        self.keep = np.arange(len(self.table)) if keep_file is None else np.load(self.root / keep_file, allow_pickle=False)
        self.keep = self._indices(self.keep)
        self.labels = self.table.label.to_numpy(dtype=np.int64)

    def _indices(self, values):
        ids = np.asarray(values)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids) or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(self.table):
            raise ValueError("Unique nonempty integer row indices required")
        return ids.astype(np.int64)

    def indices(self, fold):
        role = self.root / f"roles_f{fold}.npz"
        if role.exists():
            with np.load(role, allow_pickle=False) as state:
                train, test = self._indices(state["outer_train"]), self._indices(state["test"])
        elif "fold" in self.table:
            test = self.keep[self.table.iloc[self.keep].fold.to_numpy() == fold]
            train = np.setdiff1d(self.keep, test)
            train, test = self._indices(train), self._indices(test)
        else:
            raise ValueError("Explicit role files or reviewed fold column required")
        if not np.array_equal(np.sort(np.r_[train, test]), np.sort(self.keep)) or np.intersect1d(train, test).size:
            raise ValueError("Split must exactly partition reviewed rows")
        for field in ("condition_key", "record_id", "source_group"):
            if field in self.table and set(self.table.iloc[train][field]) & set(self.table.iloc[test][field]):
                raise ValueError(field + " leakage")
        return train, test

    def inputs(self, indices):
        ids = self._indices(indices)
        if not np.isin(ids, self.keep).all():
            raise ValueError("Requested rows are outside the reviewed cohort")
        x = np.array(self.inputs_array[ids, :, :self.channels], dtype=np.float32, copy=True)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite input")
        return {"x": torch.from_numpy(x), "mask": torch.from_numpy(np.array(self.mask[ids], dtype=np.float32, copy=True)), "label": torch.from_numpy(self.labels[ids]), "indices": ids.copy()}

DataLoader = CycleDataLoader
