"""Independent loaders reject cohort leakage and excluded candidate rows."""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("model_id", ["rcr_cnn_snn_original", "rcr_cnn_snn_peak_dynamic_affine_calibrated", "rcr_hierarchical_snn_compact"])
def test_local_loader_uses_explicit_keep_and_disjoint_roles(tmp_path, model_id):
    pd.DataFrame(dict(label=[0, 1, 0, 1, 0], condition_key=["A", "A", "B", "B", "excluded"],
                      record_id=["a", "a", "b", "b", "excluded"])).to_csv(tmp_path / "cycles.csv", index=False)
    np.save(tmp_path / "x.npy", np.ones((5, 9, 5, 64), dtype="float32"))
    np.save(tmp_path / "keep.npy", np.array([0, 1, 2, 3]))
    np.savez(tmp_path / "roles_f0.npz", outer_train=np.array([0, 1]), test=np.array([2, 3]))
    loader_class = importlib.import_module("arc_model.model_instances." + model_id + ".data_loader").DataLoader
    with pytest.raises(ValueError, match="keep_file"):
        loader_class(tmp_path, input_file="x.npy")
    loader = loader_class(tmp_path, input_file="x.npy", keep_file="keep.npy")
    tr, te = loader.indices(0)
    np.testing.assert_array_equal(tr, [0, 1])
    assert loader.inputs(te)["x"].shape == (2, 9, 5, 64)
    with pytest.raises(ValueError, match="reviewed"):
        loader.inputs([4])
    np.savez(tmp_path / "roles_f0.npz", outer_train=np.array([0, 2]), test=np.array([1, 3]))
    with pytest.raises(ValueError, match="leakage"):
        loader.indices(0)


def test_current_instances_have_distinct_loader_classes():
    first = importlib.import_module("arc_model.model_instances.rcr_cnn_snn_original.data_loader").DataLoader
    second = importlib.import_module("arc_model.model_instances.rcr_cnn_snn_dynamic_summary.data_loader").DataLoader
    assert first is not second
    assert Path(importlib.import_module(first.__module__).__file__).parent != Path(importlib.import_module(second.__module__).__file__).parent
