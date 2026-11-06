import math

import numpy as np
import pytest
import torch

from arc_model.implementations.dynamic_state_net import (
    DYNAMIC_FEATURE_NAMES,
    PARAMETER_COUNTS,
    DynamicStateNet,
    build_model,
    native_dynamic_summary,
)
from arc_model.implementations.map_input_calibration.deep_network import DeepCodedPhaseMap


def test_summary_identical_cycles_and_analytical_disjoint_support():
    raw = torch.zeros(2, 9, 5, 64, dtype=torch.float64)
    mask = torch.ones(2, 9, dtype=torch.float64)
    # Every old cycle is a point mass at bin0; current cycle is at bin1.
    raw[1, :, 1:3] = -10000
    raw[1, :-1, 1:3, 0] = 0
    raw[1, -1, 1:3, 1] = 0
    result = native_dynamic_summary(raw, mask)
    assert result.shape == (2, 36) and len(DYNAMIC_FEATURE_NAMES) == 36
    identical = result[0, :32].reshape(4, 2, 4)
    torch.testing.assert_close(identical[..., 0], torch.ones(4, 2, dtype=raw.dtype))
    torch.testing.assert_close(identical[..., 1:], torch.zeros(4, 2, 3, dtype=raw.dtype))
    changed = result[1, :32].reshape(4, 2, 4)
    expected = torch.tensor([0, 2 / 64, math.log(2), 0], dtype=raw.dtype)
    torch.testing.assert_close(changed, expected.expand(4, 2, 4), atol=2e-10, rtol=0)
    assert result[:, 32:].eq(1).all()


def test_lag_order_native_probability_oracle_and_gain_invariance():
    generator = torch.Generator().manual_seed(35)
    raw = torch.randn(3, 9, 5, 64, generator=generator, dtype=torch.float64)
    mask = torch.ones(3, 9, dtype=raw.dtype)
    result = native_dynamic_summary(raw, mask)
    power = np.power(10.0, raw[:, :, 1:3].numpy())
    p = power / power.sum(-1, keepdims=True)
    expected = []
    for lag in (1, 2, 4, 8):
        for band in (0, 1):
            u, v = p[:, -1, band], p[:, -1 - lag, band]
            m = (u + v) / 2
            expected.extend([
                (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-12),
                np.abs(u - v).mean(-1),
                0.5 * (u * np.log((u + 1e-12) / (m + 1e-12))).sum(-1)
                + 0.5 * (v * np.log((v + 1e-12) / (m + 1e-12))).sum(-1),
                (u.max(-1) - v.max(-1)) / (u.max(-1) + v.max(-1) + 1e-12),
            ])
    np.testing.assert_allclose(result[:, :32], np.stack(expected, axis=-1), rtol=1e-12, atol=1e-12)
    shifted = raw.clone()
    shifted[:, :, 1:3] += torch.randn(3, 9, 2, 1, generator=generator, dtype=raw.dtype) * 1000
    torch.testing.assert_close(native_dynamic_summary(shifted, mask), result, atol=1e-10, rtol=0)


def test_missing_history_never_wraps_or_crosses_a_gap():
    torch.manual_seed(13)
    raw = torch.randn(4, 9, 5, 64)
    mask = torch.zeros(4, 9)
    mask[1, -2:] = 1
    mask[2] = 1
    mask[2, 5] = 0
    mask[3, :-1] = 1
    raw[0] = float('nan')  # Invalid source samples cannot create NaN features.
    result = native_dynamic_summary(raw, mask)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[0], torch.zeros(36))
    torch.testing.assert_close(result[3], torch.zeros(36))
    torch.testing.assert_close(result[1, 32:], torch.tensor([1., 0., 0., 0.]))
    torch.testing.assert_close(result[2, 32:], torch.tensor([1., 1., 0., 0.]))
    assert not result[1, 8:32].any()
    assert not result[2, 16:32].any()
    changed = raw.clone()
    changed[1, :-2] = 100
    torch.testing.assert_close(native_dynamic_summary(changed, mask)[1], result[1], rtol=0, atol=0)


def test_summary_batch_independence_and_finite_zero_probability_gradients():
    torch.manual_seed(4)
    raw = torch.randn(3, 9, 5, 64, requires_grad=True)
    mask = torch.ones(3, 9)
    result = native_dynamic_summary(raw, mask)
    separate = torch.cat([native_dynamic_summary(raw[i:i+1], mask[i:i+1]) for i in range(3)])
    torch.testing.assert_close(result, separate, rtol=0, atol=0)
    result[:, :32].sum().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    extreme = torch.full((2, 9, 5, 64), -10000.)
    extreme[:, :, 1:3, 0] = 0
    extreme[:, -1, 1:3, 1] = 0
    extreme.requires_grad_()
    native_dynamic_summary(extreme, mask[:2]).sum().backward()
    assert torch.isfinite(extreme.grad).all()
    half_summary = native_dynamic_summary(extreme.detach().half(), mask[:2])
    assert half_summary.dtype == torch.float16 and torch.isfinite(half_summary).all()


@pytest.mark.parametrize('arm', tuple(PARAMETER_COUNTS))
def test_parameter_counts_and_training_gradients(arm):
    torch.manual_seed(17)
    net = DynamicStateNet(arm)
    assert sum(p.numel() for p in net.parameters()) == PARAMETER_COUNTS[arm]
    x = torch.randn(3, 9, 5, 64)
    mask = torch.ones(3, 9)
    logits, activity = net(x, mask, return_activity=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.])) + .001 * activity
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    if arm in ('dynamic', 'maxpool_dynamic'):
        assert net.readout_hidden.weight.grad[:, 128:].abs().sum() > 0


@pytest.mark.parametrize('training', [False, True])
def test_added_zero_columns_preserve_original_initial_function(training):
    torch.manual_seed(6)
    base = DeepCodedPhaseMap('raw')
    torch.manual_seed(6)
    dynamic = DynamicStateNet('dynamic')
    base.train(training)
    dynamic.train(training)
    x = torch.randn(4, 9, 5, 64)
    mask = torch.ones(4, 9)
    mask[1, :3] = 0
    with torch.no_grad():
        old_logits, old = base(x, mask, return_details=True)
        logits, new = dynamic(x, mask, return_details=True)
    torch.testing.assert_close(old['representation'], new['representation'], atol=0, rtol=0)
    torch.testing.assert_close(old_logits, logits, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(old['activity'], new['activity'], atol=0, rtol=0)
    assert not dynamic.readout_hidden.weight[:, 128:].count_nonzero()


def test_cached_native_summary_matches_online_after_training_statistics():
    torch.manual_seed(3)
    raw = torch.randn(4, 9, 5, 64)
    raw[:, :, 1:3] -= 7
    mask = torch.ones(4, 9)
    mask[1, :7] = 0
    mask[2, 5] = 0
    mean5 = torch.tensor([.1, -5, -7, .002, -.001])
    std5 = torch.tensor([.5, 2, 1.2, .01, .03])
    native = native_dynamic_summary(raw, mask)
    mean36 = native.mean(0)
    std36 = native.std(0, unbiased=False).clamp_min(1e-6)
    x = (raw - mean5[None, None, :, None]) / std5[None, None, :, None]
    net = DynamicStateNet('dynamic').set_statistics(mean5, std5, mean36, std36).eval()
    with torch.no_grad():
        # Exercise the new path after its added weights are no longer zero.
        net.readout_hidden.weight[:, 128:].normal_(0, .02)
        a, online = net(x, mask, return_details=True)
        b, cached = net(x, mask, return_details=True, summary=native)
    torch.testing.assert_close(online['native_summary'], cached['native_summary'], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    assert not cached['normalized_summary'][1, 8:32].any()
    assert not cached['normalized_summary'][2, 16:32].any()
    torch.testing.assert_close(cached['normalized_summary'][:, 32:], native[:, 32:], atol=0, rtol=0)
    # Cached flags cannot fabricate support missing from the current input mask.
    fabricated = native.clone()
    fabricated[:, 32:] = 1
    _, guarded = net(x, mask, return_details=True, summary=fabricated)
    assert not guarded['native_summary'][1, 8:32].any()


def test_statistics_persist_without_gradients_and_base_builder_is_original():
    net = DynamicStateNet('dynamic')
    net.set_statistics(np.zeros(5), np.ones(5), np.zeros(36), np.ones(36))
    assert {'input_mean', 'input_std', 'dynamic_mean', 'dynamic_std'} <= net.state_dict().keys()
    assert not any(value.requires_grad for value in net.buffers())
    assert type(build_model('base')) is DeepCodedPhaseMap
    with pytest.raises(ValueError, match='strictly positive'):
        net.set_statistics(np.zeros(5), np.zeros(5), np.zeros(36), np.ones(36))
    with pytest.raises(ValueError, match='finite'):
        net.set_statistics(np.zeros(5), np.ones(5), np.full(36, np.nan), np.ones(36))
