"""Ordered 42-feature names and physical groups (checkpoint contract)."""

NAMES = (
    [
        "log_rms2",
        "crest",
        "kurtosis",
        "near_zero_fraction",
        "log_hf_10_100k_ratio",
        "log_hf_100_400k_ratio",
    ]
    + [f"hf{b}_phasebin{k}" for b in (1, 2) for k in range(8)]
    + ["mean_abs_difference", "mean_squared_difference", "tkeo_abs", "dc_normalized"]
    + [f"harmonic_{k}_ratio" for k in range(2, 10)]
    + [
        "thd_2_20",
        "spectral_entropy",
        "halfcycle_asymmetry",
        "positive_duty",
        "slope_crest",
        "spectral_flatness",
        "top5_energy_fraction",
        "zero_region_energy_ratio",
    ]
)
GROUPS = (
    ["amplitude_shape"] * 4
    + ["hf_energy"] * 2
    + ["hf_phase_structure"] * 16
    + ["transient_shape"] * 4
    + ["harmonic_structure"] * 8
    + ["harmonic_structure"] * 2
    + ["cycle_shape"] * 2
    + ["transient_shape", "harmonic_structure", "transient_shape", "cycle_shape"]
)
