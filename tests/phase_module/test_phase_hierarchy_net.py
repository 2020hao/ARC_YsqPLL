import numpy as np
import pytest
import torch
from torch import nn

from arc_model.implementations.dynamic_state_net import native_dynamic_summary
from arc_model.implementations.phase_hierarchy_net import (
    LIF_UPDATES,
    MATRIX_MACS,
    PARAMETER_COUNTS,
    PhaseHierarchyNet,
    build_model,
)
from arc_model.implementations.phase_hierarchy_net.network import _PhaseLIF


def test_lif_analytical_soft_reset_trace_and_gap_reset():
    cell = _PhaseLIF(1)
    current = torch.zeros(1, 9, 1, 1)
    current[:, 0] = 0.75
    current[:, 1] = 0.4
    current[:, 3] = 0.2
    mask = torch.ones(1, 9)
    mask[:, 2] = 0
    events, membrane, trace, threshold, activity = cell(current, mask)
    # Initial threshold=.5 and leak=.8: .75→spike/.25; .8*.25+.4→spike/.1.
    torch.testing.assert_close(threshold, torch.tensor([[[0.5]]]), atol=1e-7, rtol=0)
    torch.testing.assert_close(events.flatten(), torch.tensor([1., 1., 0., 0., 0., 0., 0., 0., 0.]))
    torch.testing.assert_close(membrane.flatten(), torch.tensor([.2 * .8**5]), atol=1e-7, rtol=0)
    torch.testing.assert_close(trace, torch.zeros_like(trace), atol=0, rtol=0)
    torch.testing.assert_close(activity, torch.tensor(2 / 8), atol=0, rtol=0)


@pytest.mark.parametrize('size', ['small', 'wide'])
def test_exact_parameters_matrix_hooks_and_lif_updates(size):
    torch.manual_seed(11)
    net = PhaseHierarchyNet(size).eval()
    assert sum(p.numel() for p in net.parameters()) == PARAMETER_COUNTS[size]
    batch = 2
    x = torch.randn(batch, 9, 5, 64)
    mask = torch.ones(batch, 9)
    mac = []

    def hook(module, inputs, output):
        if isinstance(module, nn.Conv1d):
            mac.append(output.numel() * module.in_channels // module.groups * module.kernel_size[0])
        elif isinstance(module, nn.Linear):
            mac.append(output.numel() * module.in_features)

    handles = [module.register_forward_hook(hook) for module in net.modules() if isinstance(module, (nn.Conv1d, nn.Linear))]
    with torch.no_grad():
        logits, details = net(x, mask, return_details=True)
    for handle in handles:
        handle.remove()
    assert logits.shape == (batch,)
    assert sum(mac) // batch == MATRIX_MACS[size]
    assert sum(spikes.numel() for spikes in details['layer_spikes']) // batch == LIF_UPDATES[size]
    assert tuple(spikes.shape[-1] for spikes in details['layer_spikes']) == (64, 32, 16)
    assert details['representation'].shape == (batch, net.channels[-1] * 8)
    assert net.cost_accounting()['nominal_matrix_FLOPs'] == 2 * MATRIX_MACS[size]


@pytest.mark.parametrize('size', ['small', 'wide'])
def test_finite_training_gradients_across_all_three_spike_stages(size):
    torch.manual_seed(31)
    net = PhaseHierarchyNet(size)
    x = torch.randn(4, 9, 5, 64)
    mask = torch.ones(4, 9)
    mask[1, :3] = 0
    logits, activity = net(x, mask, return_activity=True)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0., 1.])) + .001 * activity
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    for name in ('stem', 'pw1', 'pw2'):
        assert getattr(net, name).weight.grad.abs().sum() > 0
    assert net.readout_hidden.weight.grad[:, -36:].abs().sum() > 0


def test_reference_gap_blocks_earlier_memory_and_dynamic_history():
    torch.manual_seed(77)
    net = PhaseHierarchyNet('small').eval()
    with torch.no_grad():
        net.readout_hidden.weight[:, -36:].normal_(0, .05)
    x = torch.randn(3, 9, 5, 64)
    mask = torch.ones(3, 9)
    mask[:, 5] = 0
    changed = x.clone()
    changed[:, :5] = torch.randn_like(changed[:, :5]) * 100
    changed[:, 5] = float('nan')
    with torch.no_grad():
        a, original = net(x, mask, return_details=True)
        b, reset = net(changed, mask, return_details=True)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(original['representation'], reset['representation'], atol=0, rtol=0)
    assert not reset['native_summary'][:, 16:32].any()
    assert not reset['native_summary'][:, 34:].any()


def test_zero_support_clears_all_states_and_rates_without_nan():
    torch.manual_seed(44)
    net = PhaseHierarchyNet('small').eval()
    x = torch.full((2, 9, 5, 64), float('nan'))
    mask = torch.zeros(2, 9)
    with torch.no_grad():
        logits, details = net(x, mask, return_details=True)
    assert torch.isfinite(logits).all()
    assert not details['native_summary'].any()
    assert not details['normalized_summary'].any()
    assert not details['representation'].any()
    assert not details['layer_rates'].any()
    for spikes, membrane, trace in zip(details['layer_spikes'], details['layer_membranes'], details['layer_traces']):
        assert not spikes.any() and not membrane.any() and not trace.any()


@pytest.mark.parametrize('size', ['small', 'wide'])
def test_native_cache_parity_training_statistics_and_mask_guards(size):
    torch.manual_seed(14)
    raw = torch.randn(4, 9, 5, 64)
    raw[:, :, 1:3] -= 6
    mask = torch.ones(4, 9)
    mask[1, :7] = 0
    mask[2, 5] = 0
    mean5 = torch.tensor([.2, -7, -4, .002, -.001])
    std5 = torch.tensor([.7, 1.8, 2.1, .01, .03])
    native = native_dynamic_summary(raw, mask)
    mean36 = native.mean(0)
    std36 = native.std(0, unbiased=False).clamp_min(1e-6)
    net = PhaseHierarchyNet(size).set_statistics(mean5, std5, mean36, std36).eval()
    with torch.no_grad():
        net.readout_hidden.weight[:, -36:].normal_(0, .02)
    x = (raw - mean5[None, None, :, None]) / std5[None, None, :, None]
    with torch.no_grad():
        a, online = net(x, mask, return_details=True)
        b, cached = net(x, mask, return_details=True, summary=native)
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(online['native_summary'], cached['native_summary'], atol=2e-6, rtol=2e-5)
    assert not cached['normalized_summary'][1, 8:32].any()
    assert not cached['normalized_summary'][2, 16:32].any()
    torch.testing.assert_close(cached['normalized_summary'][:, 32:], native[:, 32:], atol=0, rtol=0)
    with pytest.raises(ValueError, match='strictly positive'):
        net.set_statistics(np.zeros(5), np.zeros(5), np.zeros(36), np.ones(36))


def test_public_builder_and_invalid_input_contract():
    assert type(build_model('small')) is PhaseHierarchyNet
    with pytest.raises(ValueError, match='Unknown'):
        PhaseHierarchyNet('enormous')
    net = PhaseHierarchyNet().eval()
    with pytest.raises(ValueError, match='Expected'):
        net(torch.zeros(2, 9, 8, 64), torch.ones(2, 9))
    with pytest.raises(ValueError, match='mask'):
        net(torch.zeros(2, 9, 5, 64), torch.ones(2, 8))
