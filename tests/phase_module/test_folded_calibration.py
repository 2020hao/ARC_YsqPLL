import inspect
import json

import numpy as np
import pytest
import torch
from scipy.special import expit
from torch import nn

from arc_model.implementations.folded_calibration import (
    REGULARIZATION,
    condition_class_weights,
    fit_affine,
    fit_bias,
    fold_to_readout,
    predict_logits,
    predict_proba,
)


@pytest.mark.parametrize('fit', [fit_bias, fit_affine])
def test_zero_scores_balanced_groups_have_identity_optimum_and_certificate(fit):
    logits = np.zeros(12)
    y = np.array([0, 1, 1, 0] * 3)
    group = np.repeat(['a', 'b', 'c'], 4)
    result = fit(logits, y, group)
    assert result['success'] and result['optimizer_success']
    assert result['certificate']['passed']
    assert result['certificate']['projected_gradient_inf_norm'] < 1e-8
    assert result['scale'] == 1 and result['bias'] == 0
    assert result['lambda'] == REGULARIZATION == .01
    assert result['objective'] == pytest.approx(np.log(2))
    assert json.loads(json.dumps(result))['mode'] == result['mode']


def test_global_normalization_and_group_replication_do_not_change_prior_strength():
    logits = np.array([-.5, .7, 2.1, 3., -1.1, 1.7])
    y = np.array([0, 0, 1, 1, 0, 1])
    group = np.array(['a'] * 4 + ['b'] * 2)
    weights = condition_class_weights(y, group)
    assert weights.sum() == pytest.approx(1)
    for g in np.unique(group):
        assert weights[group == g].sum() == pytest.approx(.5)
        for label in (0, 1):
            assert weights[(group == g) & (y == label)].sum() == pytest.approx(.25)
    repeat = np.r_[np.tile(np.flatnonzero(group == 'a'), 7), np.flatnonzero(group == 'b')]
    for fit in (fit_bias, fit_affine):
        a = fit(logits, y, group)
        b = fit(logits[repeat], y[repeat], group[repeat])
        # Independent float64 optimization paths can stop at slightly different
        # points inside the same certified minimum; the objective stays equal.
        np.testing.assert_allclose([a['scale'], a['bias']], [b['scale'], b['bias']], atol=1e-7, rtol=0)
        assert a['objective'] == pytest.approx(b['objective'], abs=1e-14)
        assert a['success'] and b['success']


@pytest.mark.parametrize('fit', [fit_bias, fit_affine])
def test_synthetic_score_offset_improves_balanced_mean_bce(fit):
    rng = np.random.default_rng(42)
    latent = rng.normal(0, 1.6, 3000)
    y = (rng.random(len(latent)) < expit(latent)).astype(int)
    group = np.repeat(np.arange(6), 500)
    logits = latent + 2.3
    result = fit(logits, y, group)
    weights = condition_class_weights(y, group)
    identity_bce = np.dot(weights, np.logaddexp(0, logits) - y * logits)
    assert result['success'] and result['certificate']['passed']
    assert result['bias'] < -.5
    assert result['weighted_BCE'] < identity_bce - .15
    expected_penalty = .01 * (result['bias'] ** 2 + (result['scale'] - 1) ** 2)
    assert result['objective'] == pytest.approx(result['weighted_BCE'] + expected_penalty)


def test_active_bounds_have_projected_kkt_certificate_and_finite_objective():
    y = np.tile([0, 1], 40)
    groups = np.repeat(np.arange(4), 20)
    scores = np.where(y == 1, -200., 200.) + 500
    for fit in (fit_bias, fit_affine):
        result = fit(scores, y, groups)
        assert result['success'] and result['certificate']['passed']
        assert -.4e1 <= result['bias'] <= 4
        assert .25 <= result['scale'] <= 4
        assert result['certificate']['bound_violation'] == 0
        assert np.isfinite(result['objective'])
        assert result['objective'] <= result['identity_objective']


def test_inference_is_label_free_and_identity_is_exact():
    scores = np.array([-1000., -1, 0, 3, 1000])
    identity = {'scale': 1., 'bias': 0.}
    np.testing.assert_array_equal(predict_logits(scores, identity), scores)
    np.testing.assert_array_equal(predict_proba(scores, identity), expit(scores))
    assert tuple(inspect.signature(predict_logits).parameters) == ('logits', 'calibration')
    assert tuple(inspect.signature(predict_proba).parameters) == ('logits', 'calibration')


class _SingleHead(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.encoder = nn.Linear(3, 5)
        self.readout = nn.Linear(5, 1, bias=bias)

    def forward(self, x):
        return self.readout(torch.tanh(self.encoder(x))).flatten()


def test_affine_folding_matches_linear_sigmoid_and_preserves_original_and_parameters():
    torch.manual_seed(8)
    original = _SingleHead().double().eval()
    old_state = {k: v.clone() for k, v in original.state_dict().items()}
    parameters = sum(p.numel() for p in original.parameters())
    x = torch.randn(7, 3, dtype=torch.float64)
    folded = fold_to_readout(original, .7, -1.2)
    assert folded is not original
    with torch.no_grad():
        expected = original(x) * .7 - 1.2
        actual = folded(x)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(torch.sigmoid(actual).numpy(), predict_proba(original(x).detach().numpy(), {'scale': .7, 'bias': -1.2}), rtol=1e-14, atol=1e-14)
    assert sum(p.numel() for p in folded.parameters()) == parameters
    assert tuple(folded.state_dict()) == tuple(original.state_dict())
    for key in old_state:
        torch.testing.assert_close(original.state_dict()[key], old_state[key], atol=0, rtol=0)


def test_explicit_inplace_and_unbiased_readout_contract():
    net = _SingleHead().eval()
    weight = net.readout.weight.detach().clone()
    bias = net.readout.bias.detach().clone()
    assert fold_to_readout(net, 2., .3, inplace=True) is net
    torch.testing.assert_close(net.readout.weight, weight * 2)
    torch.testing.assert_close(net.readout.bias, bias * 2 + .3)
    unbiased = _SingleHead(bias=False)
    with pytest.raises(ValueError, match='unbiased'):
        fold_to_readout(unbiased, 1., .5)
    assert fold_to_readout(unbiased, 1., 0.).readout.bias is None


@pytest.mark.parametrize('fit', [fit_bias, fit_affine])
def test_degenerate_or_unaligned_calibration_support_is_rejected(fit):
    with pytest.raises(ValueError, match='nonempty'):
        fit([], [], [])
    with pytest.raises(ValueError, match='align'):
        fit([0, 1], [0, 1], ['a'])
    with pytest.raises(ValueError, match='finite'):
        fit([0, np.nan], [0, 1], ['a', 'a'])
    with pytest.raises(ValueError, match='binary'):
        fit([0, 1], [0, 2], ['a', 'a'])
    with pytest.raises(ValueError, match='lacks class'):
        fit([0, 1], [0, 0], ['a', 'a'])
    with pytest.raises(ValueError, match='lacks class'):
        fit([0, 1], [0, 1], ['a', 'b'])
    with pytest.raises(ValueError, match='missing'):
        fit([0, 1], [0, 1], [None, None])


def test_prediction_and_folding_reject_invalid_transform_parameters():
    with pytest.raises(ValueError, match='scale'):
        predict_logits([0, 1], {'scale': -1, 'bias': 0})
    with pytest.raises(ValueError, match='bias'):
        predict_logits([0, 1], {'scale': 1, 'bias': 5})
    with pytest.raises(ValueError, match='finite'):
        predict_logits([0, 1], {'scale': 1, 'bias': np.nan})
    with pytest.raises(ValueError, match='single-output'):
        fold_to_readout(nn.Linear(3, 1), 1., 0.)
    with pytest.raises(TypeError, match='boolean'):
        fold_to_readout(_SingleHead(), 1., 0., inplace='yes')
