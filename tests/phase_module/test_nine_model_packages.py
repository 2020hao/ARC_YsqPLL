import ast
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from arc_model.implementations.registry import load_catalog, create_model
from arc_model.models.standalone.arc_fenet import FENet
from arc_model.models.standalone.arc_fusion import FusionNet
from arc_model.models.standalone.hydra import SparseScaler
from arc_model.models.standalone.hydra_paper import paper_transform

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ROOT / "src/arc_model/implementations"


def test_nine_folders_and_no_peer_component_imports():
    specs = load_catalog()
    assert len(specs) == 9
    for s in specs.values():
        folder = PACKAGES / s.implementation
        for n in [
            "model.py",
            "network.py",
            "data_loader.py",
            "configuration.py",
            "training.py",
            "backbone_training.py",
            "evaluation.py",
            "performance.json",
        ]:
            assert (folder / n).is_file()
        for path in folder.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert node.level <= 1
                    assert not (node.module or "").startswith("arc_model")
                if isinstance(node, ast.Import):
                    assert not any(a.name.startswith("arc_model") for a in node.names)
        perf = json.loads((folder / "performance.json").read_text())
        if s.role == "baseline":
            assert len(perf["scores"]) == 2
        else:
            if perf["status"] == "not_validated_on_locked_parent":
                assert perf["scores"] == []
            else:
                assert perf["status"] == "seed0_screen_complete_on_locked_parent"
                assert perf["seeds"] == [0] and perf["outer_folds"] == 5
                assert perf["source"] == "outputs/locked_routes_20261030/main_results.csv"
                assert perf["no_legacy_score_inheritance"] is True
                assert perf["promoted_to_default"] is False
                assert {r["dataset"] for r in perf["scores"]} == {"continuous5m", "zenodo5501527"}
                assert all(r["seed"] == 0 and r["N"] in (38374, 1361) for r in perf["scores"])


@pytest.mark.parametrize("backbone,original", [("arc_fenet", FENet), ("arc_fusion", FusionNet)])
def test_refactor_and_both_disabled_variants_preserve_network(backbone, original):
    torch.set_num_threads(2)
    torch.manual_seed(44)
    source = original(extra=0).eval()
    x = torch.randn(2, 3, 100, 100) if backbone == "arc_fenet" else torch.randn(2, 1, 3, 250)
    spectral = None if backbone == "arc_fenet" else torch.randn(2, 1, 32)
    with torch.no_grad():
        expected = source(x, spectral)
    for spec in load_catalog().values():
        if spec.backbone != backbone:
            continue
        model = create_model(spec.name).eval()
        model.load_state_dict(source.state_dict(), strict=True)
        with torch.no_grad():
            actual = model(x, spectral)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if spec.role == "plugin":
            aligned = x if spectral is None else dict(temporal=x, spectrum=spectral)
            with torch.no_grad():
                same = model(x, spectral, aligned=aligned, enabled=True)
            torch.testing.assert_close(same, expected, rtol=0, atol=0)
        if spec.role == "dual":
            with pytest.raises(ValueError, match="No fitted correction"):
                model(x, spectral, dynamic=np.zeros((2, 36)), enabled=True)


def test_hydra_author_transform_and_scores_preserved():
    torch.set_num_threads(2)
    rng = np.random.default_rng(1)
    waves = rng.uniform(0, 1, (5, 1000)).astype("float32")
    features = paper_transform(waves)
    sc = SparseScaler()
    sc.fit(torch.tensor(features[:3]))
    state = dict(
        weight=rng.normal(0, 0.01, 1024),
        intercept=np.asarray(0.3),
        mu=sc.mu.numpy(),
        sigma=sc.sigma.numpy(),
        epsilon=sc.epsilon.numpy(),
    )
    expected = sc.transform(torch.tensor(features)).numpy() @ state["weight"] + state["intercept"]
    for spec in load_catalog().values():
        if spec.backbone != "hydra":
            continue
        model = create_model(spec.name)
        model.state = deepcopy(state)
        np.testing.assert_array_equal(model.extract(waves), features)
        np.testing.assert_array_equal(model.decision_function(features), expected)
        if spec.role == "plugin":
            np.testing.assert_array_equal(
                model.decision_function(features, aligned=features, enabled=True), expected
            )
        if spec.role == "dual":
            with pytest.raises(ValueError, match="No fitted correction"):
                model.decision_function(features, dynamic=np.zeros((5, 36)), enabled=True)


def test_yaml_nodes_are_independent_and_constructor_values_are_active(tmp_path):
    p = ROOT / "configs/models.yaml"
    data = yaml.safe_load(p.read_text())
    data["ArcFusionNet_Current_Plugin"]["model_kwargs"]["dropout"] = 0.27
    f = tmp_path / "models.yaml"
    f.write_text(yaml.safe_dump(data))
    changed = create_model("ArcFusionNet_Current_Plugin", config_path=f)
    unchanged = create_model("ArcFusionNet_Current_Baseline", config_path=f)
    assert changed.te.layers[0].dropout.p == 0.27
    assert unchanged.te.layers[0].dropout.p == 0.1
    data["ArcFENet_DualBranch"]["model_kwargs"]["channels"] = [16, 24, 48, 96]
    f.write_text(yaml.safe_dump(data))
    model = create_model("ArcFENet_DualBranch", config_path=f)
    assert model.backbone.head.in_features == 96 * 4


def test_duplicate_yaml_keys_rejected(tmp_path):
    f = tmp_path / "dup.yaml"
    f.write_text("A: 1\nA: 2\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_catalog(f)


def test_prepared_loader_uses_current_mask_without_changing_source(tmp_path):
    import pandas as pd

    folder = tmp_path / "data"
    folder.mkdir()
    pd.DataFrame(
        dict(
            sample_id=["a0", "a1", "a2", "b0", "b1", "b2"],
            record_id=[0, 0, 0, 1, 1, 1],
            condition_key=["a"] * 3 + ["b"] * 3,
            source_group=["a"] * 3 + ["b"] * 3,
            label=[0, 1, 0, 0, 1, 0],
        )
    ).to_csv(folder / "cycles.csv", index=False)
    # Third row per record is an old unreviewed row and must not enter training.
    splits = [dict(train=[3, 4, 5], test=[0, 1, 2]), dict(train=[0, 1, 2], test=[3, 4, 5])]
    (folder / "splits.json").write_text(json.dumps(splits))
    np.save(folder / "wave.npy", np.zeros((6, 1000), dtype="float32"))
    loader = importlib.import_module(
        "arc_model.implementations.baseline_hydra.data_loader"
    ).DataLoader(folder, reviewed_indices=np.array([0, 1, 3, 4]))
    tr, te = loader.indices(0)
    np.testing.assert_array_equal(tr, [3, 4])
    np.testing.assert_array_equal(te, [0, 1])
    np.testing.assert_array_equal(loader.eligible_indices, [0, 1, 3, 4])


def test_baseline_release_guard_and_candidate_independence(tmp_path):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from check_baseline_release import verify

    assert verify()["baseline_packages"] == 3
    cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    cfg["ArcFENet_Plugin"]["model_kwargs"]["channels"] = [16, 24, 48, 96]
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert verify(config_path=p)["status"] == "pass"
    cfg["ArcFENet_Baseline"]["model_training_param"]["lr"] = 0.002
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="Frozen baseline YAML"):
        verify(config_path=p)
    with pytest.raises(ValueError, match="Frozen baseline configuration"):
        create_model("ArcFENet_Baseline", config_path=p)


def test_independent_hydra_training_api_records_resolved_yaml(tmp_path, monkeypatch):
    module = importlib.import_module("arc_model.implementations.dual_hydra.backbone_training")
    original = module.load_config()
    cfg = deepcopy(original)
    cfg["model_kwargs"].update(k=2, g=2)
    monkeypatch.setattr(module, "load_config", lambda: cfg)
    rng = np.random.default_rng(42)
    features = rng.uniform(0, 1, (10, 64)).astype("float32")
    model = module.fit(
        features,
        np.arange(10) % 2,
        np.arange(8),
        np.arange(8, 10),
        tmp_path / "fit",
        groups=np.array(["train"] * 8 + ["test"] * 2),
    )
    assert (tmp_path / "fit/resolved_model.yaml").is_file()
    assert model.decision_function(features[8:]).shape == (2,)
    with pytest.raises(ValueError, match="disjoint"):
        module.fit(
            features,
            np.arange(10) % 2,
            np.arange(8),
            np.arange(8, 10),
            tmp_path / "bad",
            groups=np.array(["one"] * 10),
        )
