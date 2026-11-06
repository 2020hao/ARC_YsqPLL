import pytest
import yaml

from arc_model.orchestration.configuration import (
    canonical_hash, find_project_root, read_yaml, resolve_variables,
)
from arc_model.orchestration.registry import discover_models, validate_registry
from arc_model.orchestration.runtime import device_environment, validate_runtime


def checkout(tmp_path):
    (tmp_path / "src/arc_model").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/registry.yaml").write_text(yaml.safe_dump({
        "schema_version": 2, "discovery": ["src/arc_model/model_instances/*/model.yaml"],
        "sealed_baselines": [],
    }))
    return tmp_path


def add_model(root, name):
    folder = root / "src/arc_model/model_instances" / name
    folder.mkdir(parents=True)
    for filename in ("network.py", "data_loader.py", "preprocessing.py", "training.py",
                     "inference.py", "README.md", "provenance.json", "__init__.py"):
        (folder / filename).write_text("{}" if filename.endswith(".json") else "")
    cfg = {"schema_version": 2, "model_id": name, "display_name": name,
           "role": "direct_classifier", "package": f"arc_model.model_instances.{name}",
           "factory": f"arc_model.model_instances.{name}:create_model", "network": {},
           "input": {}, "training": {}, "evaluation": {}, "status": "test", "provenance": {}}
    (folder / "model.yaml").write_text(yaml.safe_dump(cfg))
    return folder


def test_registry_is_extensible_and_requires_local_model_files(tmp_path):
    root = checkout(tmp_path)
    for index in range(11):
        add_model(root, f"independent_{index}")
    assert validate_registry(root)["models"] == 11
    assert validate_registry(root)["status"] == "passed"
    (root / "src/arc_model/model_instances/independent_3/data_loader.py").unlink()
    result = validate_registry(root)
    assert result["status"] == "failed"
    assert "Missing independent model file: data_loader.py" in result["errors"]["independent_3"]


def test_duplicate_ids_and_yaml_keys_are_rejected(tmp_path):
    root = checkout(tmp_path)
    first = add_model(root, "first")
    second = add_model(root, "second")
    cfg = read_yaml(second / "model.yaml")
    cfg["model_id"] = "first"
    (second / "model.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="Duplicate model ID"):
        discover_models(root)
    (first / "model.yaml").write_text("model_id: first\nmodel_id: second\n")
    with pytest.raises(ValueError, match="Duplicate configuration key"):
        read_yaml(first / "model.yaml")


def test_registry_rejects_hidden_cross_model_imports_and_factories(tmp_path):
    root = checkout(tmp_path)
    folder = add_model(root, "independent")
    (folder / "network.py").write_text("from arc_model.implementations.dynamic_state_net.network import DynamicStateNet\n")
    assert "Forbidden shared model import" in str(validate_registry(root)["errors"])
    (folder / "network.py").write_text("from arc_model.model_instances.other.network import Model\n")
    assert "Cross-instance import" in str(validate_registry(root)["errors"])
    (folder / "network.py").write_text("from arc_model.modules.rcr_dynamic_branch_1mhz import Correction\n")
    assert validate_registry(root)["status"] == "passed"
    cfg = read_yaml(folder / "model.yaml")
    cfg["factory"] = "arc_model.model_instances.other:create_model"
    (folder / "model.yaml").write_text(yaml.safe_dump(cfg))
    assert "Factory must be defined inside" in str(validate_registry(root)["errors"])


def test_root_search_and_variables_do_not_guess_private_paths(tmp_path, monkeypatch):
    root = checkout(tmp_path)
    nested = root / "unusual/deep/path"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == root
    monkeypatch.delenv("ARC_UNSET_TEST_PATH", raising=False)
    with pytest.raises(ValueError, match="Unconfigured path variable"):
        resolve_variables("${ARC_UNSET_TEST_PATH}")
    assert resolve_variables({"root": "${EXPLICIT}"}, {"EXPLICIT": "/local"}) == {"root": "/local"}
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})


def test_gpu0_is_forbidden_without_initializing_cuda():
    with pytest.raises(ValueError, match="Forbidden physical GPU"):
        device_environment(0)
    assert device_environment(2)["CUDA_VISIBLE_DEVICES"] == "2"
    assert device_environment()["CUDA_VISIBLE_DEVICES"] == ""
    with pytest.raises(ValueError, match="requires physical"):
        validate_runtime({"allowed_physical_gpus": [0, 1, 2, 3], "forbidden_physical_gpus": []})

