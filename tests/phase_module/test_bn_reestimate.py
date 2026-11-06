import copy
import torch
from torch import nn
from arc_model.implementations.map_input_calibration.bn_reestimate import reestimate_bn


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(3)
        self.fc = nn.Linear(3, 4)
        self.bn2 = nn.BatchNorm1d(4)
    def forward(self, x):
        return self.bn2(self.fc(torch.relu(self.bn1(x))))


def test_population_moments_and_fixed_weights():
    torch.manual_seed(10)
    model = Toy().eval()
    x = torch.randn(23, 3) * 2 + 4
    before = {n: v.clone() for n, v in model.named_parameters()}
    receipt = reestimate_bn(model, lambda: ((b,) for b in x.split(7)))
    torch.testing.assert_close(model.bn1.running_mean, x.mean(0))
    torch.testing.assert_close(model.bn1.running_var, x.var(0, unbiased=False))
    with torch.no_grad():
        middle = model.fc(torch.relu(model.bn1(x)))
    torch.testing.assert_close(model.bn2.running_mean, middle.mean(0))
    torch.testing.assert_close(model.bn2.running_var, middle.var(0, unbiased=False))
    assert all(torch.equal(before[n], v) for n, v in model.named_parameters())
    assert all(r['count_per_channel'] == len(x) for r in receipt)


def test_reestimate_independent_of_batch_boundaries_and_order():
    torch.manual_seed(12)
    a = Toy().eval(); b = copy.deepcopy(a)
    x = torch.randn(31, 3)
    reestimate_bn(a, lambda: ((v,) for v in x.split(8)))
    reverse = x.flip(0)
    reestimate_bn(b, lambda: ((v,) for v in reverse.split(13)))
    for name in ('bn1', 'bn2'):
        torch.testing.assert_close(getattr(a, name).running_mean, getattr(b, name).running_mean, atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(getattr(a, name).running_var, getattr(b, name).running_var, atol=2e-7, rtol=2e-6)
