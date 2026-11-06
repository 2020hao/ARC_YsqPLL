"""Load an explicit retained cohort and validate acquisition-disjoint outer folds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from arc_model.orchestration.configuration import (
    canonical_hash, file_hash, find_project_root, local_variables, read_yaml, resolve_variables,
)

CURRENT_DATASETS = {"self462": "self462_v1", "public": "public_v1",
                    "zenodo5501527": "public_v1"}


def _indices(values, length: int, name: str) -> np.ndarray:
    ids = np.asarray(values)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids):
        raise ValueError(f"{name}: nonempty integer indices required")
    if ids.min() < 0 or ids.max() >= length or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name}: invalid, duplicate or out-of-range indices")
    return ids.astype(np.int64, copy=False)


@dataclass
class PreparedCohort:
    dataset: dict[str, Any]
    protocol: dict[str, Any]
    directory: Path
    table: pd.DataFrame
    keep: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]]

    @classmethod
    def open(cls, directory, dataset, protocol, *, expected_counts=True):
        directory = Path(directory).expanduser().resolve()
        table = pd.read_csv(directory / dataset["prepared"]["metadata"])
        required = {"record_id", "condition_key", "label"}
        if not required <= set(table):
            raise ValueError(f"Missing required cohort columns: {sorted(required - set(table))}")
        if table[list(required)].isna().any().any() or not table.label.isin([0, 1]).all():
            raise ValueError("Missing source metadata or nonbinary labels")
        keep_name = dataset["prepared"].get("keep")
        keep = _indices(np.load(directory / keep_name, allow_pickle=False), len(table), "keep") if keep_name else np.arange(len(table), dtype=np.int64)
        folds = []
        split = dataset["prepared"]["splits"]
        for fold in range(protocol["outer_folds"]):
            if split["kind"] == "metadata_column":
                if split["column"] not in table:
                    raise ValueError("Missing explicit fold assignment column")
                test = keep[table.iloc[keep][split["column"]].to_numpy() == fold]
                train = np.setdiff1d(keep, test)
            elif split["kind"] == "npz_roles":
                with np.load(directory / split["pattern"].format(fold=fold), allow_pickle=False) as roles:
                    train, test = roles[split["train_key"]], roles[split["test_key"]]
            else:
                raise ValueError("Unknown explicit split storage")
            folds.append((_indices(train, len(table), "train"), _indices(test, len(table), "test")))
        result = cls(dataset, protocol, directory, table, keep, folds)
        result.audit(expected_counts=expected_counts)
        return result

    def audit(self, *, expected_counts=True):
        coverage = np.zeros(len(self.table), dtype=np.int64)
        for fold, (train, test) in enumerate(self.folds):
            if np.setdiff1d(test, self.keep).size:
                raise ValueError(f"Fold {fold}: test contains excluded candidate rows")
            if np.intersect1d(train, test).size or not np.array_equal(np.sort(train), np.setdiff1d(self.keep, test)):
                raise ValueError(f"Fold {fold}: train must be the exact kept complement of test")
            for name in self.protocol["isolation_columns"]:
                if name not in self.table:
                    raise ValueError(f"Missing isolation column: {name}")
                left, right = self.table.iloc[train][name], self.table.iloc[test][name]
                if left.isna().any() or right.isna().any() or set(left) & set(right):
                    raise ValueError(f"Fold {fold}: source leakage or missing groups in {name}")
            coverage[test] += 1
        if not (coverage[self.keep] == 1).all() or coverage[np.setdiff1d(np.arange(len(self.table)), self.keep)].any():
            raise ValueError("Every retained row must appear in exactly one outer test fold")
        retained = self.table.iloc[self.keep]
        counts = {"windows": len(retained), "records": retained.record_id.nunique(),
                  "groups": retained.condition_key.nunique(),
                  "normal": int((retained.label == 0).sum()), "arc": int((retained.label == 1).sum())}
        counts = {key: int(value) for key, value in counts.items()}
        if expected_counts:
            for key, expected in self.protocol["expected"].items():
                if counts.get(key) != expected:
                    raise ValueError(f"Cohort identity mismatch: {key}={counts.get(key)}, expected {expected}")
        return {"status": "passed", "dataset_id": self.dataset["dataset_id"],
                "protocol_id": self.protocol["protocol_id"], "counts": counts,
                "folds": [{"fold": fold, "train": len(train), "test": len(test)}
                          for fold, (train, test) in enumerate(self.folds)]}

    def open_array(self, name: str) -> np.ndarray:
        """Read-only memmap; individual model loaders still own representation selection."""
        descriptions = self.dataset["prepared"].get("arrays", {})
        if name not in descriptions:
            raise ValueError(f"Unknown representation: {name}")
        item = descriptions[name]
        values = np.load(self.directory / item["file"], mmap_mode="r", allow_pickle=False)
        if len(values) != len(self.table):
            raise ValueError("Prepared representation and candidate metadata row counts differ")
        if values.flags.writeable:
            raise ValueError("Prepared data must be read-only")
        return values

    def identity(self):
        files = [self.directory / self.dataset["prepared"]["metadata"]]
        if self.dataset["prepared"].get("keep"):
            files.append(self.directory / self.dataset["prepared"]["keep"])
        split = self.dataset["prepared"]["splits"]
        if split["kind"] == "npz_roles":
            files += [self.directory / split["pattern"].format(fold=fold) for fold in range(len(self.folds))]
        return {"dataset_id": self.dataset["dataset_id"], "protocol_id": self.protocol["protocol_id"],
                "dataset_schema_hash": canonical_hash(self.dataset),
                "protocol_hash": canonical_hash(self.protocol),
                "files": [{"name": path.name, "sha256": file_hash(path)} for path in files]}


def load_current_cohort(dataset_id: str, *, root=None, local_config=None, directory=None):
    if dataset_id not in CURRENT_DATASETS:
        raise ValueError("Retired or unknown dataset; current datasets are self462 and public")
    project = find_project_root(root)
    dataset = read_yaml(project / "configs/datasets" / (CURRENT_DATASETS[dataset_id] + ".yaml"))
    protocol = read_yaml(project / "configs/protocols" / (dataset["protocol_id"] + ".yaml"))
    if directory is None:
        directory = resolve_variables(dataset["prepared"]["root"], local_variables(project, local_config))
    return PreparedCohort.open(directory, dataset, protocol)

