import struct
import numpy as np
import pytest
from arc_model.data.current_recording import read_current
from arc_model.preprocessing.phase_windows import prepare_phase_windows


def test_net9770_header_and_current_only(tmp_path):
    p = tmp_path / "test.bin"
    n = 40_000
    a = np.zeros((n, 8))
    a[:, 0] = np.sin(np.arange(n) * 2 * np.pi * 50 / 2e6)
    with p.open("wb") as f:
        f.write(struct.pack("<32sQIIQ", b"NET9770_114", 0, 8, 2_000_000, n))
        f.write(a.astype("<f8").tobytes())
    r = read_current(p, 2_000_000)
    np.testing.assert_array_equal(r.current, a[:, 0])
    with pytest.raises(ValueError):
        read_current(p, 1_000_000)
    q = tmp_path / "C2_test.dat"
    np.savetxt(q, np.c_[np.arange(20), np.ones(20)])
    with pytest.raises(ValueError, match="C2"):
        read_current(q, 1_000_000)


@pytest.mark.parametrize("method", ["fixed20", "sogi", "rcr", "cft"])
def test_native_hf_and_prefix_causality(method):
    fs = 1_000_000
    t = np.arange(500_000) / fs
    raw = np.sin(2 * np.pi * 50 * t) + 0.04 * np.sin(2 * np.pi * 180_000 * t)
    anchors = np.array([0.3, 0.32, 0.34, 0.36, 0.38, 0.4])
    full = prepare_phase_windows(raw, fs, reference=method, anchors=anchors)
    prefix = prepare_phase_windows(raw[:400_000], fs, reference=method, anchors=anchors)
    np.testing.assert_array_equal(full.energy, prefix.energy)
    np.testing.assert_array_equal(full.wave, prefix.wave)
    np.testing.assert_array_equal(full.native, prefix.native)
    assert full.energy[:, 1].sum() > 1e-7
    assert (full.metadata.available_seconds <= full.metadata.anchor_seconds + 1e-9).all()
    if method == "fixed20":
        assert not full.metadata.phase_valid.any()
