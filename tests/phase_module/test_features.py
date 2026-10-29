import numpy as np
import pytest
from arc_model.methods.phase_representation import phase_features
from arc_model.training.phase_features import PhaseFeatureScaler


def test_position_and_intensity_are_separate():
    e = np.zeros((2, 2, 20))
    e[0, :, 0] = 2
    e[1, :, 5] = 2
    f = phase_features(e)
    np.testing.assert_array_equal(f.values[0, :2], f.values[1, :2])
    assert f.values[0, 2] == 1 and f.values[1, 2] == 0
    assert f.values[0, 4] > 0 and f.values[1, 4] < 0
    scaled = phase_features(e * 7)
    np.testing.assert_allclose(f.values[:, 2:], scaled.values[:, 2:])
    np.testing.assert_allclose(scaled.values[:, :2] - f.values[:, :2], np.log(7))


def test_half_cycle_polarity_and_shift_controls():
    rng = np.random.default_rng(5)
    e = rng.uniform(0.1, 4, (7, 2, 20))
    f = phase_features(e)
    half = phase_features(np.roll(e, 10, axis=-1))
    np.testing.assert_allclose(f.values, half.values, atol=1e-14)
    free = phase_features(e, mode="phase_free")
    for s in range(20):
        np.testing.assert_allclose(
            free.values,
            phase_features(np.roll(e, s, axis=-1), mode="phase_free").values,
            atol=1e-14,
        )
    shift = phase_features(e, mode="shifted", shifts=np.arange(7))
    np.testing.assert_array_equal(shift.values[:, :2], f.values[:, :2])
    assert not np.allclose(shift.values[:, 2:], f.values[:, 2:])


def test_invalid_energy_and_nominal_coordinate_are_explicit():
    z = phase_features(np.zeros((3, 2, 20)), phase_valid=np.zeros(3, bool))
    assert not z.valid.any() and not z.phase_valid.any() and not z.values.any()
    f = phase_features(np.ones((3, 2, 20)), phase_valid=np.zeros(3, bool))
    assert f.valid.all() and not f.phase_valid.any()
    with pytest.raises(ValueError):
        phase_features(-np.ones((1, 2, 20)))
    with pytest.raises(ValueError):
        phase_features(np.ones((1, 2, 19)))


def test_scaling_cannot_see_test_rows():
    rng = np.random.default_rng(8)
    e = rng.uniform(0.1, 3, (9, 2, 20))
    a = phase_features(e)
    ids = np.array([0, 1, 2, 3])
    sc = PhaseFeatureScaler.fit(a, ids)
    e[4:] *= 1000
    b = phase_features(e)
    other = PhaseFeatureScaler.fit(b, ids)
    np.testing.assert_array_equal(sc.mean, other.mean)
    np.testing.assert_array_equal(sc.scale, other.scale)
    np.testing.assert_allclose(sc.transform(a)[ids].mean(0), 0, atol=1e-6)
    restored = PhaseFeatureScaler.from_state(sc.state())
    np.testing.assert_array_equal(restored.transform(a), sc.transform(a))
    with pytest.raises(ValueError):
        PhaseFeatureScaler.fit(a, np.array([0, 0]))
