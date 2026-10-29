"""Guard independent model files against architecture drift and data leakage."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from arc_model.models.arc_fenet.network import FENet as OldFENet
from arc_model.models.arc_fusion.network import FusionNet as OldFusion
from arc_model.models.standalone.arc_fenet import FENet
from arc_model.models.standalone.arc_fusion import FusionNet
from arc_model.models.standalone.hydra import transform_numpy
from arc_model.vendor.hydra_class import Hydra as OldHydra
from arc_model.data.arc_fenet_loader import ArcFENetLoader
from arc_model.data.arc_fusion_loader import ArcFusionLoader
from arc_model.data.hydra_loader import HydraLoader
from arc_model.methods.phase_representation.input_frontend import pack_current_energy


@pytest.mark.parametrize(
    "old,new,shape", [(OldFENet, FENet, (2, 3, 100, 100)), (OldFusion, FusionNet, (2, 1, 3, 250))]
)
def test_neural_state_outputs_and_gradients_match(old, new, shape):
    torch.set_num_threads(2)
    torch.manual_seed(2)
    a = old().eval()
    b = new().eval()
    assert list(a.state_dict()) == list(b.state_dict())
    b.load_state_dict(a.state_dict(), strict=True)
    x = torch.randn(shape)
    s = torch.randn(2, 1, 32) if new is FusionNet else None
    ya, yb = a(x, s), b(x, s)
    torch.testing.assert_close(ya, yb, rtol=0, atol=0)
    ya.sum().backward()
    yb.sum().backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)
    assert a.extra is None and b.extra is None


def test_hydra_transform_matches_original():
    torch.set_num_threads(2)
    x = np.random.default_rng(3).normal(size=(4, 1000)).astype("float32")
    a = OldHydra(1000, k=8, g=4, stride=2, seed=0)(torch.from_numpy(x[:, None])).numpy()
    b = transform_numpy(x, seed=0)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", ["arc_fenet", "arc_fusion", "hydra"])
def test_model_file_importable_without_package(tmp_path, name):
    # Module imports must work directly by file, with no package context.
    root = Path(__file__).resolve().parents[2] / "src/arc_model/models/standalone"
    spec = importlib.util.spec_from_file_location("isolated_" + name, root / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)


@pytest.fixture
def prepared(tmp_path):
    n = 4
    pd.DataFrame(
        dict(
            sample_id=["a", "b", "c", "d"],
            label=[0, 1, 0, 1],
            condition_key=["a", "a", "b", "b"],
            source_group=["a", "a", "b", "b"],
            record_id=[0, 0, 1, 1],
        )
    ).to_csv(tmp_path / "cycles.csv", index=False)
    (tmp_path / "splits.json").write_text(
        json.dumps([dict(train=[0, 1], test=[2, 3]), dict(train=[2, 3], test=[0, 1])])
    )
    np.save(tmp_path / "rgb.npy", np.zeros((n, 3, 100, 100), np.float32))
    np.save(tmp_path / "wave.npy", np.zeros((n, 1000), np.float32))
    np.save(tmp_path / "temporal.npy", np.zeros((n, 1, 3, 250), np.float32))
    np.save(tmp_path / "spectrum.npy", np.zeros((n, 125), np.float32))
    return tmp_path


@pytest.mark.parametrize("cls", [ArcFENetLoader, ArcFusionLoader, HydraLoader])
def test_loader_shapes_and_leakage(prepared, cls):
    d = cls(prepared)
    tr, te = d.indices(0)
    if cls is ArcFusionLoader:
        b = d.inputs(te, selected_bins=np.arange(32))
        assert b["temporal"].shape == (2, 1, 3, 250) and b["spectrum"].shape == (2, 1, 32)
        with pytest.raises(ValueError):
            d.inputs(te)
    else:
        b = d.inputs(te)
        key = "image" if cls is ArcFENetLoader else "wave"
        assert b[key].shape[0] == 2
    t = pd.read_csv(prepared / "cycles.csv")
    t.loc[2, "source_group"] = "a"
    t.to_csv(prepared / "cycles.csv", index=False)
    with pytest.raises(ValueError, match="leakage"):
        cls(prepared)


def test_input_frontend_contract():
    w = np.ones((3, 1000))
    e = np.ones((3, 2, 20)) * 0.001
    a = pack_current_energy(w, e, np.full(3, 0.02))
    assert a.shape == (3, 1000) and np.isfinite(a).all()
    np.testing.assert_allclose(a[:, 600], np.log1p(1), rtol=1e-6)
    with pytest.raises(ValueError):
        pack_current_energy(w, -e, np.full(3, 0.02))
