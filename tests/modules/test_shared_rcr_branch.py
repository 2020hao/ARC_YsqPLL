from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.signal import upfirdn

from arc_model.contracts import CurrentChunk, FitIdentity, ScorePacket
from arc_model.modules.rcr_dynamic_branch_1mhz import Correction, RCRDynamicBranch1MHz
from arc_model.modules.rcr_dynamic_branch_1mhz.features import DYNAMIC_FEATURE_NAMES
from arc_model.modules.rcr_dynamic_branch_1mhz.sampling import CausalOneMHz, antialias_coefficients
from arc_model.modules.rcr_dynamic_branch_1mhz.state import groups_digest, indices_digest

torch.set_num_threads(1)


def identity(kind="logit_margin", ids=None, groups=None):
    ids = np.arange(20) if ids is None else ids
    groups = np.repeat(np.arange(4), 5) if groups is None else groups
    return FitIdentity("model", "parent-checkpoint-hash", "self462", "split-hash",
                       indices_digest(ids), groups_digest(groups), 0, kind)


def state(fusion="whole_path", strength=.4):
    return dict(keep=np.arange(36), mu=np.zeros(36), sd=np.ones(36),
                coef=np.arange(36, dtype=np.float32) / 100,
                pca_components=np.eye(36)[:6], pca_mean=np.zeros(36),
                gate_threshold=np.array(1.3), strength=np.array(strength),
                fusion=np.array(fusion), calibration_slope=np.array(1.2),
                calibration_intercept=np.array(.3), dimension=np.array(36))


@pytest.mark.parametrize("rate", [1_000_000, 2_000_000])
def test_sampling_is_chunk_invariant_and_matches_validated_causal_aa(rate):
    x = np.random.default_rng(14).normal(size=10043)
    sampler = CausalOneMHz(rate)
    pieces = []
    start = 0
    for count in [1, 3, 102, 11, 3401, 1, 2, 1999, 2, 4521]:
        if start == len(x):
            break
        end = min(len(x), start + count)
        pieces.append(sampler.update(x[start:end]))
        start = end
    if start < len(x):
        pieces.append(sampler.update(x[start:]))
    actual = np.concatenate(pieces)
    expected = x if rate == 1_000_000 else upfirdn(antialias_coefficients(), x, down=2)[:(len(x)+1)//2]
    np.testing.assert_array_equal(actual, expected)
    assert len(sampler.history) == 102
    assert sampler.delay_seconds == (0 if rate == 1_000_000 else 25.5e-6)


def test_antialias_coefficients_match_existing_preparation_script():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts/prepare_self462_1m_signals_20261105.py"
    spec = importlib.util.spec_from_file_location("old_sampling_for_parity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert len(antialias_coefficients()) == 103
    np.testing.assert_array_equal(antialias_coefficients(), module.antialias())


@pytest.mark.parametrize("kind,threshold", [("svm_margin", 0), ("logit_margin", 0), ("probability", .5)])
def test_disabled_scores_and_decisions_are_exact_identity(kind, threshold):
    x = np.array([0, .2, .5, 1], dtype=np.float32)
    packet = ScorePacket(x, kind, threshold)
    branch = RCRDynamicBranch1MHz()
    result = branch.correct(packet)
    assert result is packet
    assert result.values is x
    np.testing.assert_array_equal(result.decisions, packet.decisions)
    assert branch.predict(x, score_kind=kind, threshold=threshold) is x


def test_whole_path_zero_is_identity_but_residual_zero_keeps_calibration():
    x = np.array([-1, .1], dtype=np.float32)
    packet = ScorePacket(x)
    branch = RCRDynamicBranch1MHz(state(strength=0), identity())
    assert branch.correct(packet, enabled=True) is packet
    residual = RCRDynamicBranch1MHz(state("old_residual", 0), identity())
    got = residual.correct(packet, np.zeros((2, 36)), enabled=True)
    np.testing.assert_array_equal(got.values, 1.2 * x + .3)


def test_new_fusion_matches_original_v2_and_does_not_sigmoid_svm_margin():
    from arc_model.implementations.dual_hydra.phase_dynamic_v2 import Correction as Old

    raw = np.array([-2, .2, 4], dtype=np.float64)
    x = np.random.default_rng(33).normal(size=(3, 36)).astype("float32")
    for fusion in ("old_residual", "whole_path"):
        values = state(fusion)
        old_state = dict(values, baseline_id=np.asarray("baseline"), score_kind=np.asarray("svm_margin"))
        expected = Old(old_state, baseline_id="baseline", score_kind="svm_margin").score(raw, x)
        branch = RCRDynamicBranch1MHz(values, identity("svm_margin"))
        got = branch.correct(ScorePacket(raw, "svm_margin", 0), x, enabled=True)
        np.testing.assert_array_equal(got.values, expected)
        assert got.kind == "svm_margin" and got.values.max() > 1


def test_legacy_alpha_correction_is_separate_and_exact():
    from arc_model.implementations.dual_arc_fenet.enhancement import Correction as Old

    values = state()
    values.update(alpha=np.asarray(.7), baseline_id=np.asarray("baseline"), score_kind=np.asarray("logit"))
    x = np.random.default_rng(2).normal(size=(5, 36)).astype("float32")
    expected = Old(values, baseline_id="baseline", score_kind="logit").delta(x)
    actual = Correction(values, baseline_id="baseline", score_kind="logit").delta(x)
    np.testing.assert_array_equal(actual, expected)


def test_features_are_exact_copy_and_clear_noncontiguous_history():
    from arc_model.implementations.dynamic_state_net.summary import native_dynamic_summary

    raw = torch.randn(3, 9, 5, 64, generator=torch.Generator().manual_seed(32))
    mask = torch.ones(3, 9)
    mask[1, 6] = 0
    mask[2, :8] = 0
    actual = RCRDynamicBranch1MHz.transform(raw, mask)
    np.testing.assert_array_equal(actual, native_dynamic_summary(raw, mask).numpy())
    assert len(DYNAMIC_FEATURE_NAMES) == 36
    np.testing.assert_array_equal(actual[1, 32:], [1, 0, 0, 0])
    np.testing.assert_array_equal(actual[2], np.zeros(36))


def test_saved_state_is_bound_to_model_data_split_seed(tmp_path):
    branch = RCRDynamicBranch1MHz(state(), identity())
    path = tmp_path / "branch.npz"
    branch.save(path)
    loaded = RCRDynamicBranch1MHz.load(path, expected_identity=identity())
    for key in branch.state:
        np.testing.assert_array_equal(loaded.state[key], branch.state[key])
    with pytest.raises(ValueError, match="Wrong"):
        RCRDynamicBranch1MHz.load(path, expected_identity=replace(identity(), model_id="other"))
    with pytest.raises(ValueError, match="conventions"):
        branch.correct(ScorePacket(np.ones(2), "svm_margin"), np.zeros((2, 36)), enabled=True)


def test_enabled_correction_requires_identity_and_rejects_wrong_expected_fit():
    packet = ScorePacket(np.array([-1., 1.]))
    x = np.zeros((2, 36))
    unbound = RCRDynamicBranch1MHz(state())
    with pytest.raises(ValueError, match="FitIdentity"):
        unbound.correct(packet, x, enabled=True)
    assert unbound.correct(packet, enabled=False) is packet
    branch = RCRDynamicBranch1MHz(state(), identity())
    for replacement in (dict(model_id="other"), dict(parent_checkpoint_sha256="wrong"),
                        dict(dataset_id="public"), dict(split_sha256="other-fold"), dict(seed=1)):
        with pytest.raises(ValueError, match="identity"):
            branch.predict(packet.values, x, enabled=True,
                           expected_identity=replace(identity(), **replacement))
    np.testing.assert_array_equal(branch.predict(packet.values, x, enabled=True,
                                               expected_identity=identity()),
                                  branch.correct(packet, x, enabled=True).values)


def test_fit_rejects_group_leakage_and_does_not_read_heldout_labels():
    x = np.random.default_rng(22).normal(size=(40, 36)).astype("float32")
    y = np.tile([0, 1], 20)
    g = np.repeat(np.arange(8), 5)
    ids, held = np.arange(20), np.arange(20, 40)
    ident = identity(ids=ids, groups=g[ids])
    scores = ScorePacket(np.linspace(-1, 1, 40).astype("float32"))
    options = dict(identity=ident, heldout_indices=held, honest_oof=True, epochs=1)
    a = RCRDynamicBranch1MHz().fit(x, y, scores, ids, g, **options)
    changed = y.copy()
    changed[held] = 1 - changed[held]
    b = RCRDynamicBranch1MHz().fit(x, changed, scores, ids, g, **options)
    for key in a.state:
        np.testing.assert_array_equal(a.state[key], b.state[key])
    with pytest.raises(ValueError, match="overlap"):
        RCRDynamicBranch1MHz().fit(x, y, scores, ids, g, **{**options, "heldout_indices": np.arange(19, 40)})
    with pytest.raises(ValueError, match="OOF"):
        RCRDynamicBranch1MHz().fit(x, y, scores, ids, g, **{**options, "honest_oof": False})
    with pytest.raises(ValueError, match="Calibration"):
        RCRDynamicBranch1MHz().fit(x, y, scores, ids, g, **{**options, "calibration_fit_indices": held})
    with pytest.raises(ValueError, match="integer"):
        RCRDynamicBranch1MHz().fit(x, y, scores, ids + .2, g, **options)
    with pytest.raises(ValueError, match="integer"):
        RCRDynamicBranch1MHz().fit(x, y, scores, ids, g, **{**options, "heldout_indices": held + .2})
    broken_groups = g.astype(object)
    broken_groups[held[0]] = ""
    with pytest.raises(ValueError, match="empty"):
        RCRDynamicBranch1MHz().fit(x, y, scores, ids, broken_groups, **options)
    broken_y = y.copy()
    broken_y[0] = 2
    with pytest.raises(ValueError, match="classes"):
        RCRDynamicBranch1MHz().fit(x, broken_y, scores, ids, g, **options)


def stream_values(values, sizes):
    branch = RCRDynamicBranch1MHz()
    events, start = [], 0
    for count in sizes:
        if start >= len(values):
            break
        end = min(len(values), start + count)
        events += branch.observe(CurrentChunk(values[start:end], 1_000_000, "record", start))
        start = end
    if start < len(values):
        events += branch.observe(CurrentChunk(values[start:], 1_000_000, "record", start))
    return events, branch


def test_streaming_chunk_parity_causality_and_record_reset():
    fs = 1_000_000
    t = np.arange(180_000) / fs
    values = np.sin(2*np.pi*50*t) + .02*np.sin(2*np.pi*30_000*t)
    first, branch = stream_values(values, [len(values)])
    second, _ = stream_values(values, [1, 1999, 4111, 128, 34001, 40001])
    assert first and len(first) == len(second)
    for a, b in zip(first, second):
        assert (a.start_sample, a.stop_sample_exclusive, a.coordinate, a.phase_status) == (
            b.start_sample, b.stop_sample_exclusive, b.coordinate, b.phase_status)
        np.testing.assert_array_equal(a.features, b.features)
        assert a.stop_sample_exclusive/fs <= a.available_at_seconds
    assert len(branch._stream.prefixes) <= 30
    assert len(branch._pending) < 2000
    assert {row.coordinate for row in first} == {"native", "fallback"}
    with pytest.raises(ValueError, match="consecutive"):
        branch.observe(CurrentChunk(np.ones(2000), fs, "record", len(values)+1))
    reset = branch.observe(CurrentChunk(np.zeros(20_000), fs, "new-record", 0))
    assert len(reset) == 1 and reset[0].coordinate == "fallback"
    np.testing.assert_array_equal(reset[0].history_mask, [0, 0, 0, 0, 0, 0, 0, 0, 1])
