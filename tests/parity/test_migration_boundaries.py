"""Recovery and public-distribution boundaries of the structural migration."""

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def script(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_guard_includes_untracked_legacy_sources(tmp_path, monkeypatch):
    module = script("audit_project_layout_v2")
    project = tmp_path / "project"
    path = project / "src/arc_model/local_untracked.py"
    path.parent.mkdir(parents=True)
    path.write_text("original = True\n")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    entries = [{"path": "src/arc_model/local_untracked.py",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]
    (snapshot / "SOURCE_MANIFEST.json").write_text(json.dumps(entries))
    monkeypatch.setattr(module, "ROOT", project)
    assert module.preserved_sources(snapshot)["original_files_checked"] == 1
    path.write_text("original = False\n")
    with pytest.raises(ValueError, match="changed"):
        module.preserved_sources(snapshot)
    path.unlink()
    with pytest.raises(ValueError, match="changed"):
        module.preserved_sources(snapshot)


def test_portable_export_includes_instance_configs_without_local_data():
    module = script("export_code_snapshot")
    files = {path.relative_to(ROOT).as_posix() for path in module.public_files()}
    assert "configs/registry.yaml" in files
    assert "configs/datasets/self462_v1.yaml" in files
    assert "configs/protocols/public_group5_1357.yaml" in files
    assert "src/arc_model/model_instances/rcr_cnn_snn_peak_dynamic_affine_calibrated/model.yaml" in files
    assert "scripts/project.py" in files
    assert "scripts/prepare_self462_1m_signals_20261105.py" in files
    for name in ("report_model_upgrade_20261105.py", "run_dynamic_state_20261105.py",
                 "run_phase_hierarchy_20261105.py", "run_folded_calibration_20261105.py"):
        assert f"scripts/{name}" in files
    assert not any(path.startswith(("outputs/", "artifacts/", "data/")) for path in files)
    assert not any(path.startswith("configs/local/") and not path.endswith("paths.example.yaml")
                   for path in files)
    for name in ("README.md", "docs/QUICKSTART.md", "docs/DATASETS.md", "docs/MODEL_CATALOG.md"):
        source = ROOT / name
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", source.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            linked = (source.parent / target.split("#")[0]).resolve()
            assert linked.relative_to(ROOT).as_posix() in files, (name, target)
