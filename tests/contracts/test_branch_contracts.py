import numpy as np
import pytest

from arc_model.contracts import BranchEvent, CurrentChunk, FitIdentity, ScorePacket


def test_reference_rate_cannot_be_mislabelled_as_hf_input():
    with pytest.raises(ValueError, match="reference-only"):
        CurrentChunk(np.ones(10), 5000, "record", 0)


def test_score_packets_preserve_svm_margin_and_native_threshold():
    margin = np.array([-4, 0, 3], dtype=np.float32)
    packet = ScorePacket(margin, "svm_margin", 0)
    assert packet.values is margin
    np.testing.assert_array_equal(packet.decisions, [False, True, True])
    with pytest.raises(ValueError, match="Probability"):
        ScorePacket(margin, "probability", 0.5)
    probability = ScorePacket(np.array([.1, .7]), kind="probability")
    assert probability.threshold == .5
    np.testing.assert_array_equal(probability.decisions, [False, True])


def test_event_rejects_future_support():
    with pytest.raises(ValueError, match="before"):
        BranchEvent("r", 0, 20000, 1000000, 0, .019, "fallback", -1, -1,
                    "NOMINAL", np.zeros(36), np.ones(9))


@pytest.mark.parametrize("index", [.5, True, "0"])
def test_fractional_or_ambiguous_acquisition_indices_are_rejected(index):
    with pytest.raises(ValueError, match="integer"):
        CurrentChunk(np.ones(10), 1_000_000, "record", index)


@pytest.mark.parametrize("seed", [-1, .5, True])
def test_invalid_fit_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        FitIdentity("m", "parent", "self462", "split", "indices", "groups", seed, "svm_margin")
