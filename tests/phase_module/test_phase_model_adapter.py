import numpy as np
import pytest
import torch
from arc_model.models.factory import make_model
from arc_model.models.phase_adapter import PhaseFeatureAdapter


@pytest.mark.parametrize("name", ["ArcFENet", "HYDRA", "ArcFusionNet"])
def test_adapter_baseline_parity_and_real_feature_gradient(name):
    torch.manual_seed(9)
    base = torch.nn.Linear(448, 1) if name == "HYDRA" else make_model(name)
    base.eval()
    adapter = PhaseFeatureAdapter(base, name).eval()
    x = (
        torch.randn(2, 448)
        if name == "HYDRA"
        else torch.randn(2, 3, 100, 100)
        if name == "ArcFENet"
        else torch.randn(2, 1, 3, 250)
    )
    s = torch.randn(2, 1, 32) if name == "ArcFusionNet" else None
    f = torch.randn(2, 10)
    expected = base(x) if name == "HYDRA" else base(x, s)
    torch.testing.assert_close(adapter(x, s, enabled=False), expected, rtol=0, atol=0)
    torch.testing.assert_close(adapter(x, s, f), expected, rtol=1e-6, atol=1e-7)
    assert all(p.requires_grad for p in base.parameters())
    with torch.no_grad():
        adapter.fusion_head.weight[:, -16:] = 0.05
    a = adapter(x, s, f)
    b = adapter(x, s, f + 2)
    assert not torch.allclose(a, b)
    masked = adapter(x, s, f, available=torch.zeros(2, dtype=torch.bool))
    torch.testing.assert_close(masked, expected, rtol=0, atol=0)
    adapter.zero_grad()
    a.sum().backward()
    assert adapter.feature_encoder[0].weight.grad.abs().sum() > 0
    assert np.isfinite(a.detach().numpy()).all()
