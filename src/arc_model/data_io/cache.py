"""Cache identity depends on signal provenance and processing, never just shape."""

from arc_model.orchestration.configuration import canonical_hash

REQUIRED = {"raw_source_hash", "channel", "source_sample_rate_hz", "target_sample_rate_hz",
            "antialias_version", "reference_version", "bands_hz", "phase_bins", "history",
            "feature_schema", "code_hash", "config_hash"}


def cache_key(**identity):
    missing = REQUIRED - identity.keys()
    if missing:
        raise ValueError(f"Incomplete cache identity: {sorted(missing)}")
    if any(identity[name] is None for name in REQUIRED):
        raise ValueError("Cache identities must be explicit; use identity/no_filter where appropriate")
    return canonical_hash(identity)

