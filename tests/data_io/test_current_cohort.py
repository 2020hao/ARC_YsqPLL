import numpy as np
import pandas as pd
import pytest

from arc_model.data_io.cache import cache_key
from arc_model.data_io.cohort import PreparedCohort, load_current_cohort


def fixture_cohort(tmp_path):
    table = pd.DataFrame({"record_id": np.repeat(["r0", "r1", "r2", "r3"], 2),
                          "condition_key": np.repeat(["g0", "g1", "g2", "g3"], 2),
                          "source_group": np.repeat(["s0", "s1", "s2", "s3"], 2),
                          "label": [0, 1] * 4, "fold": [0, 0, 1, 1, 0, 0, 1, 1]})
    table.to_csv(tmp_path / "cycles.csv", index=False)
    np.save(tmp_path / "phase.npy", np.ones((8, 9, 5, 64), dtype=np.float32))
    ds = {"dataset_id": "synthetic", "prepared": {"metadata": "cycles.csv", "keep": None,
          "splits": {"kind": "metadata_column", "column": "fold"},
          "arrays": {"phase": {"file": "phase.npy"}}}}
    protocol = {"protocol_id": "synthetic_group2", "outer_folds": 2,
                "isolation_columns": ["record_id", "condition_key", "source_group"],
                "expected": {"records": 4, "groups": 4, "windows": 8, "normal": 4, "arc": 4}}
    return table, ds, protocol


def test_grouped_cohort_readonly_arrays_and_identity(tmp_path):
    _, ds, protocol = fixture_cohort(tmp_path)
    cohort = PreparedCohort.open(tmp_path, ds, protocol)
    assert cohort.audit()["counts"]["windows"] == 8
    assert not cohort.open_array("phase").flags.writeable
    assert cohort.identity()["files"][0]["name"] == "cycles.csv"
    for train, test in cohort.folds:
        assert not set(cohort.table.iloc[train].record_id) & set(cohort.table.iloc[test].record_id)


def test_acquisition_leakage_and_wrong_cohort_are_rejected(tmp_path):
    table, ds, protocol = fixture_cohort(tmp_path)
    table.loc[2, "record_id"] = "r0"
    table.to_csv(tmp_path / "cycles.csv", index=False)
    with pytest.raises(ValueError, match="source leakage"):
        PreparedCohort.open(tmp_path, ds, protocol)
    table.loc[2, "record_id"] = "r1"
    table.to_csv(tmp_path / "cycles.csv", index=False)
    protocol["expected"]["windows"] = 9
    with pytest.raises(ValueError, match="Cohort identity mismatch"):
        PreparedCohort.open(tmp_path, ds, protocol)


def test_saved_role_indices_apply_keep_and_cover_once(tmp_path):
    table, ds, protocol = fixture_cohort(tmp_path)
    np.save(tmp_path / "keep.npy", np.arange(6, dtype=np.int64))
    ds["prepared"]["keep"] = "keep.npy"
    ds["prepared"]["splits"] = {"kind": "npz_roles", "pattern": "roles_f{fold}.npz",
                                  "train_key": "outer_train", "test_key": "test"}
    np.savez(tmp_path / "roles_f0.npz", outer_train=[2, 3], test=[0, 1, 4, 5])
    np.savez(tmp_path / "roles_f1.npz", outer_train=[0, 1, 4, 5], test=[2, 3])
    protocol["expected"] = {"records": 3, "groups": 3, "windows": 6, "normal": 3, "arc": 3}
    assert PreparedCohort.open(tmp_path, ds, protocol).audit()["counts"]["windows"] == 6
    np.savez(tmp_path / "roles_f1.npz", outer_train=[0, 1, 4, 5], test=[2, 3, 6])
    with pytest.raises(ValueError, match="excluded candidate"):
        PreparedCohort.open(tmp_path, ds, protocol)


def test_retired_self_datasets_cannot_enter_current_loader():
    with pytest.raises(ValueError, match="Retired or unknown dataset"):
        load_current_cohort("continuous5m")


def test_cache_key_distinguishes_same_shape_different_sampling_and_reference():
    identity = dict(raw_source_hash="raw", channel="CH0", source_sample_rate_hz=2000000,
                    target_sample_rate_hz=1000000, antialias_version="aa103", reference_version="rcr_v2",
                    bands_hz=[[10000, 100000], [100000, 400000]], phase_bins=64, history=9,
                    feature_schema="dynamic36_v1", code_hash="source", config_hash="config")
    one = cache_key(**identity)
    assert one != cache_key(**{**identity, "target_sample_rate_hz": 2000000})
    assert one != cache_key(**{**identity, "reference_version": "sogi_v1"})
    del identity["antialias_version"]
    with pytest.raises(ValueError, match="Incomplete cache identity"):
        cache_key(**identity)

