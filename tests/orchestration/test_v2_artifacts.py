from dataclasses import replace
import json

import pytest

from arc_model.orchestration.artifacts import RunIdentity, initialize_run


def identity():
    return RunIdentity("study_v1", "self462_group4_125051", "model_v1", 0, 1,
                       "model_hash", "data_hash", "split_hash", "source_hash", "schema_hash")


def test_resume_requires_same_source_data_configuration_and_split(tmp_path):
    item = identity()
    directory = initialize_run(tmp_path, item, {"epochs": 50})
    receipt = json.loads((directory / "run_manifest.json").read_text())
    assert receipt["status"] == "prepared_not_trained"
    assert initialize_run(tmp_path, item, {"epochs": 50}) == directory
    with pytest.raises(ValueError, match="Run identity changed"):
        initialize_run(tmp_path, replace(item, split_hash="other_split"), {"epochs": 50})
    with pytest.raises(ValueError, match="Run identity changed"):
        initialize_run(tmp_path, item, {"epochs": 100})


def test_run_path_traversal_and_empty_provenance_are_rejected():
    with pytest.raises(ValueError, match="path traversal"):
        replace(identity(), study_id="../../private")
    with pytest.raises(ValueError, match="Explicit run identity"):
        replace(identity(), source_hash="")

