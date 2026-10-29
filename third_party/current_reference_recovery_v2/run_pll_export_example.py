"""NEW documentation adapter, not a historical training/analysis script.

Run locally on CPU. Input: finite, real, one-dimensional current .npy sampled
at exactly 1 MHz or 2 MHz. Caller must establish units/calibration/channel.
The offline symmetric-FIR reference is not a causal deployment implementation.
This adapter has only been statically checked in this documentation export.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from partition_current import partition
from reference_recovery_recommended import complete_cycles, estimate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--fs", required=True, type=int, choices=(1_000_000, 2_000_000))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    spec_path = here / "pll_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec["reference_fs_hz"] != 2000:
        raise ValueError("This adapter is limited to the exported 2 kHz reference spec")
    if args.output.exists():
        raise FileExistsError("Use a new output directory; existing outputs are not overwritten")
    if args.input.suffix.lower() != ".npy":
        raise ValueError("Only explicit single-channel .npy input is supported")

    input_sha = sha256_file(args.input)
    raw = np.load(args.input, mmap_mode="r", allow_pickle=False)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iuf":
        raise ValueError("Input must be nonempty, one-dimensional, real numeric current")
    if not np.isfinite(raw).all():
        raise ValueError("Input contains NaN or infinity; do not silently fill raw data")
    x = np.asarray(raw, dtype=np.float64)
    args.output.mkdir(parents=True, exist_ok=False)

    # Same two-stage offline reference preprocessing as V4 clocks.py.
    low = resample_poly(x, 1, args.fs // 10000, window=("kaiser", 5.0))
    ref = resample_poly(low, 1, 5, window=("kaiser", 5.0))
    time_s = np.arange(len(ref), dtype=np.float64) / spec["reference_fs_hz"]
    # Wrapper forces dynamic=False despite the underlying V2 default being True.
    clock, events = estimate(time_s, ref, spec)
    good = complete_cycles(clock, args.fs, spec)
    ledger = pd.DataFrame(partition(good, len(x), args.fs))

    ledger["all_zero"] = [
        not bool(np.any(x[int(a):int(b)]))
        for a, b in zip(ledger.start_sample, ledger.stop_sample_exclusive)
    ]
    ledger["complete"] = ledger["kind"].ne("partial_gap")
    # Only a label-free candidate flag; no arc labels/classification are produced.
    ledger["analysis_candidate"] = ledger["complete"] & ~ledger["all_zero"]
    if sha256_file(args.input) != input_sha:
        raise RuntimeError("Input changed during processing; do not use this run")

    clock.to_csv(args.output / "clock.csv", index=False)
    good.to_csv(args.output / "pll_cycles.csv", index=False)
    ledger.to_csv(args.output / "partition.csv", index=False)
    save_json(args.output / "events.json", events)
    code_names = [
        "reference_recovery_recommended.py", "reference_recovery_v2.py",
        "dual_reference_pll.py", "partition_current.py", "run_pll_export_example.py",
    ]
    save_json(args.output / "run_info.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input),
        "input_sha256": input_sha,
        "input_dtype": str(raw.dtype),
        "sample_count": len(x),
        "raw_fs_hz": args.fs,
        "first_sample_time_s": 0.0,
        "reference_fs_hz": spec["reference_fs_hz"],
        "spec_sha256": sha256_file(spec_path),
        "code_sha256": {name: sha256_file(here / name) for name in code_names},
        "python_version": platform.python_version(),
        "package_versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas")
        },
        "dynamic": False,
        "compute": "local CPU; no GPU framework imported",
        "phase_truth_certified": False,
        "causal_frontend": False,
        "fallback_recent_resets_per_epoch": False,
        "limits": [
            "Input calibration and physical units are caller responsibilities",
            "phase_valid is an estimator gate, not a true physical phase certificate",
            "Fallback coordinates are estimates and partial_gap is not a full cycle",
            "Reference morphology origin is not grid-voltage phase",
            "No labels, detector, high-frequency features or event accuracy are produced",
        ],
    })
    print(f"Saved reference diagnostics and gapless cycle ledger to {args.output}")


if __name__ == "__main__":
    main()
