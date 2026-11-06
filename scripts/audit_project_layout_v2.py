"""Audit a structural migration, optionally replaying existing private checkpoints.

This command never trains, chooses a threshold, regenerates labels, or moves
evidence. Private evidence paths are explicit arguments, not runtime defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import _bootstrap as boot
import numpy as np

ROOT = boot.ROOT
CURRENT = {
    "rcr_cnn_snn_original": ("dynamic_state_20261105", "base"),
    "rcr_cnn_snn_dynamic_summary": ("dynamic_state_20261105", "dynamic"),
    "rcr_cnn_snn_capacity_control": ("dynamic_state_20261105", "capacity"),
    "rcr_cnn_snn_peak_dynamic": ("dynamic_state_20261105", "maxpool_dynamic"),
    "rcr_cnn_snn_peak_dynamic_augmented": ("phase_hierarchy_20261105", "gain_phase"),
    "rcr_hierarchical_snn_compact": ("phase_hierarchy_20261105", "hierarchy_small"),
    "rcr_hierarchical_snn_wide": ("phase_hierarchy_20261105", "hierarchy_wide"),
    "rcr_cnn_snn_peak_dynamic_bias_calibrated": ("folded_calibration_20261105", "folded_bias"),
    "rcr_cnn_snn_peak_dynamic_affine_calibrated": ("folded_calibration_20261105", "folded_affine"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preserved_sources(snapshot):
    """Check original algorithms and all legacy runners, including untracked files."""
    entries = json.loads((Path(snapshot) / "SOURCE_MANIFEST.json").read_text())
    selected = [entry for entry in entries if entry["path"].startswith((
        "src/", "configs/", "scripts/", "tests/", "releases/"))
        and entry["path"] not in {"scripts/README.md", "scripts/export_code_snapshot.py"}]
    changed = [entry["path"] for entry in selected
               if not (ROOT / entry["path"]).is_file()
               or sha(ROOT / entry["path"]) != entry["sha256"]]
    if changed:
        raise ValueError(f"Original implementation or evidence entry changed: {changed}")
    return {"status": "passed", "original_files_checked": len(selected)}


def replay_current(evidence, self_prepared, public_prepared):
    """Load all nine current variants on both datasets and compare CPU outputs."""
    import pandas as pd
    import torch
    from arc_model.orchestration.registry import build_model, discover_models

    torch.set_num_threads(1)
    specs = discover_models(ROOT)
    results = []
    for dataset, prepared, folds in (("self462", self_prepared, 4),
                                      ("zenodo5501527", public_prepared, 5)):
        table = pd.read_csv(prepared / "cycles.csv")
        if dataset == "self462":
            raw = np.load(prepared / "cnn_snn_2m_control.npy", mmap_mode="r")
            mask = np.ones((len(table), 9), dtype=np.float32)
            keep = np.arange(len(table))
        else:
            raw = np.load(prepared / "realtime_raw.npy", mmap_mode="r")[:, :, :5]
            mask = np.load(prepared / "realtime_mask.npy", mmap_mode="r")
            keep = np.load(prepared / "keep.npy")
        if len(keep) != (125051 if dataset == "self462" else 1357):
            raise ValueError("Only the current reviewed cohorts may be replayed")
        summaries = np.load(evidence / "dynamic_state_20261105" /
                            f"{dataset}_dynamic36.npy", mmap_mode="r")
        for model_id, (stage, arm) in CURRENT.items():
            provenance = specs[model_id].config["provenance"]
            module, name = provenance["legacy_class"].split(":")
            old_class = getattr(importlib.import_module(module), name)
            for seed in range(3):
                for fold in range(folds):
                    run = evidence / stage / "runs" / dataset
                    if stage == "folded_calibration_20261105":
                        run = run / f"f{fold}_s{seed}"
                        checkpoint = run / f"{arm}_model.pt"
                        predictions = run / f"{arm}_predictions.npz"
                    else:
                        run = run / arm / f"f{fold}_s{seed}"
                        checkpoint = run / "model.pt"
                        predictions = run / "predictions.npz"
                    with np.load(predictions, allow_pickle=False) as saved:
                        ids = saved["indices"]
                        slots = np.unique(np.linspace(0, len(ids) - 1, 8).astype(int))
                        probe = ids[slots]
                        saved_probability = saved["probability"][slots]
                    if not np.isin(probe, keep).all():
                        raise ValueError("Checkpoint test rows do not match current cohort")
                    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    original = old_class(**provenance["legacy_kwargs"]).eval()
                    migrated = build_model(model_id, root=ROOT).eval()
                    original.load_state_dict(state, strict=True)
                    migrated.load_state_dict(state, strict=True)
                    native = torch.from_numpy(np.array(raw[probe], dtype=np.float32, copy=True))
                    available = torch.from_numpy(np.array(mask[probe], dtype=np.float32, copy=True))
                    x = ((native - original.input_mean[None, None, :, None]) /
                         original.input_std[None, None, :, None])
                    summary = torch.from_numpy(np.array(summaries[probe], copy=True))
                    with torch.inference_mode():
                        expected = original(x, available, summary=summary)
                        actual = migrated(x, available, summary=summary)
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                    probability = actual.sigmoid().numpy()
                    results.append({
                        "model_id": model_id, "dataset": dataset, "fold": fold,
                        "seed": seed, "checkpoint_sha256": sha(checkpoint),
                        "probe_windows": len(probe), "CPU_old_new_max_abs_error": 0.0,
                        "CPU_old_new_decisions_equal": True,
                        "CPU_saved_scoring_max_abs_error": float(np.max(np.abs(
                            probability - saved_probability))),
                        "CPU_saved_scoring_decision_differences": int(np.count_nonzero(
                            (probability >= .5) != (saved_probability >= .5))),
                    })
    if len(results) != 243:
        raise ValueError(f"Expected243 existing outer states, got{len(results)}")
    return {"status": "passed", "checkpoints": len(results), "rows": results,
            "new_training_fits": 0, "GPU_used": False,
            "scope": "CPU old/new exact parity; saved GPU scores are a separate numerical diagnostic"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--checkpoint-parity", action="store_true")
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--self-prepared", type=Path)
    parser.add_argument("--public-prepared", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    from arc_model.orchestration.registry import validate_registry

    report = {"registry": validate_registry(ROOT), "training_started": False,
              "GPU_used": False}
    if report["registry"]["status"] != "passed":
        raise ValueError(report["registry"])
    if args.snapshot:
        report["preserved_sources"] = preserved_sources(args.snapshot)
    if args.checkpoint_parity:
        if not all((args.evidence_root, args.self_prepared, args.public_prepared)):
            parser.error("Checkpoint replay requires explicit evidence and both prepared directories")
        report["checkpoint_parity"] = replay_current(
            args.evidence_root, args.self_prepared, args.public_prepared)
    report["status"] = "passed"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: ({k: v for k, v in value.items() if k != "rows"}
                           if isinstance(value, dict) else value)
                      for key, value in report.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
