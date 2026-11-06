"""Current-cohort project CLI. Listing and validation never launch training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from arc_model.data_io import load_current_cohort
from arc_model.orchestration.configuration import find_project_root, read_yaml
from arc_model.orchestration.registry import build_model, discover_models, validate_registry
from arc_model.orchestration.runtime import load_runtime


def _print(document):
    print(json.dumps(document, ensure_ascii=False, indent=2, default=str))


def _current_configs(root):
    datasets = []
    for name in ("self462_v1", "public_v1"):
        dataset = read_yaml(root / "configs/datasets" / (name + ".yaml"))
        protocol = read_yaml(root / "configs/protocols" / (dataset["protocol_id"] + ".yaml"))
        if protocol["dataset_id"] != dataset["dataset_id"]:
            raise ValueError("Dataset and protocol identity mismatch")
        expected_windows = 125051 if dataset["dataset_id"] == "self462" else 1357
        if protocol["expected"]["windows"] != expected_windows:
            raise ValueError("Only current self462125051 and public1357 cohorts are active")
        datasets.append({"dataset_id": dataset["dataset_id"], "protocol_id": protocol["protocol_id"],
                         "windows": expected_windows, "outer_folds": protocol["outer_folds"]})
    load_runtime(root / "configs/runtime/server.yaml")
    return datasets


def report_runs(directory):
    """Report identified V2 runs, without scanning legacy output folders for scores."""
    records = []
    for path in sorted(Path(directory).glob("runs/*/*/*/fold-*/seed-*/run_manifest.json")):
        item = json.loads(path.read_text())
        identity = item["identity"]
        metrics = path.parent / "metrics.json"
        records.append({"path": str(path.parent), "identity": identity,
                        "status": item["status"],
                        "metrics": json.loads(metrics.read_text()) if metrics.exists() else None})
    return {"runs": records, "count": len(records), "training_started": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Project checkout, or set ARC_PROJECT_ROOT")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List all independent registered model instances")
    listing.add_argument("--role", choices=("baseline", "output_correction", "input_alignment", "direct_classifier"))
    listing.add_argument("--recommended", action="store_true", help="Show sealed comparators and current recommended candidate")
    commands.add_parser("validate", help="Validate model schemas, frozen sources and current protocols")
    config = commands.add_parser("config", help="Show complete canonical model configuration")
    config.add_argument("model_id")
    build = commands.add_parser("build", help="Construct one model on CPU; no fitting or inference")
    build.add_argument("model_id")
    build.add_argument("--seed", type=int, default=0)
    data = commands.add_parser("data-audit", help="Read-only current-cohort/source-isolation audit")
    data.add_argument("dataset", choices=("self462", "public", "zenodo5501527"))
    data.add_argument("--prepared-dir", help="Explicit prepared data directory, otherwise configured local path")
    data.add_argument("--local-config")
    plan = commands.add_parser("plan", help="Show an explicitly named current study; never run it")
    plan.add_argument("--experiment", default="configs/experiments/current_comparison.yaml")
    report = commands.add_parser("report", help="Read identified V2 run receipts and metrics")
    report.add_argument("--artifacts", required=True)
    args = parser.parse_args(argv)
    try:
        root = find_project_root(args.root)
        if args.command == "list":
            models = discover_models(root)
            if args.role:
                models = {key: spec for key, spec in models.items() if spec.config["role"] == args.role}
            if args.recommended:
                models = {key: spec for key, spec in models.items() if spec.config.get("recommended", False)}
            _print({"models": [{"model_id": key, "display_name": spec.config["display_name"],
                                 "role": spec.config["role"], "status": spec.config["status"],
                                 "directory": str(spec.package_path.relative_to(root))}
                                for key, spec in models.items()], "count": len(models)})
        elif args.command == "validate":
            result = validate_registry(root)
            result["current_datasets"] = _current_configs(root)
            _print(result)
            return 0 if result["status"] == "passed" else 1
        elif args.command == "config":
            models = discover_models(root)
            if args.model_id not in models:
                raise ValueError(f"Unknown model instance: {args.model_id}")
            spec = models[args.model_id]
            _print({"config_file": str(spec.config_path), "config_hash": spec.digest,
                    "resolved_model": spec.config})
        elif args.command == "build":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            import torch
            torch.manual_seed(args.seed)
            model = build_model(args.model_id, root=root)
            parameters = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else None
            _print({"model_id": args.model_id, "status": "constructed_not_trained", "device": "cpu",
                    "parameters": parameters, "seed": args.seed})
        elif args.command == "data-audit":
            cohort = load_current_cohort(args.dataset, root=root, local_config=args.local_config,
                                         directory=args.prepared_dir)
            _print({**cohort.audit(), "identity": cohort.identity()})
        elif args.command == "plan":
            study = read_yaml(root / args.experiment)
            if study.get("schema_version") != 2 or set(study.get("datasets", [])) != {"self462", "public"}:
                raise ValueError("A V2 current two-dataset experiment configuration is required")
            _print({"status": "plan_only", "experiment": study, "current_datasets": _current_configs(root),
                    "training_started": False})
        elif args.command == "report":
            _print(report_runs(args.artifacts))
        return 0
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.exit(2, f"arc-project: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

