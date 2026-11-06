"""Extensible V2 registry; sealed baseline packages and V1 registry stay intact."""

from __future__ import annotations

import ast
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .configuration import canonical_hash, file_hash, find_project_root, read_yaml

REQUIRED = {"schema_version", "model_id", "display_name", "role", "package", "factory",
            "network", "input", "training", "evaluation", "status", "provenance"}
ROLES = {"baseline", "output_correction", "input_alignment", "direct_classifier"}


@dataclass(frozen=True)
class ModelSpec:
    config: dict[str, Any]
    config_path: Path
    package_path: Path

    @property
    def model_id(self):
        return self.config["model_id"]

    @property
    def digest(self):
        return canonical_hash(self.config)


def _sealed_baseline(root: Path, entry: dict[str, Any]) -> ModelSpec:
    """Expose the canonical sealed node, without making a second competing YAML."""
    document = read_yaml(root / entry["config_file"])
    node = document[entry["config_key"]]
    lock = json.loads((root / entry["release"] / "baseline.lock.json").read_text())
    sealed = json.loads((root / entry["release"] / "SEALED.json").read_text())
    if file_hash(root / entry["release"] / "baseline.lock.json") != sealed["baseline_lock_sha256"]:
        raise ValueError("Sealed baseline release identity changed")
    if canonical_hash(node) != lock["baseline_yaml_nodes"][entry["config_key"]]:
        raise ValueError(f"Sealed baseline configuration changed: {entry['model_id']}")
    for source in lock["files"]:
        if file_hash(root / source["path"]) != source["sha256"]:
            raise ValueError(f"Sealed baseline source changed: {source['path']}")
    config = {
        "schema_version": 2, "model_id": entry["model_id"],
        "display_name": entry["display_name"], "role": "baseline",
        "package": entry["package"], "factory": entry["factory"],
        "factory_kwargs": node["model_kwargs"],
        "network": {"source": entry["package"] + ".network", "frozen": True},
        "input": node["input_contract"], "training": node["model_training_param"],
        "evaluation": {"current_protocols": ["self462_group4_125051", "public_group5_1357"],
                       "current_public_replay_status": "requires_current_protocol_verification"},
        "status": "sealed_source_current_self462_1mhz_verified",
        "recommended": True,
        "provenance": {"sealed_config_file": entry["config_file"],
                       "sealed_config_key": entry["config_key"],
                       "sealed_config_hash": canonical_hash(node),
                       "sealed_release": entry["release"],
                       "historical_evaluation_is_not_current_protocol": True},
    }
    return ModelSpec(config, root / entry["config_file"],
                     root / "src" / entry["package"].replace(".", "/"))


def discover_models(root: str | Path | None = None) -> dict[str, ModelSpec]:
    project = find_project_root(root)
    catalog = read_yaml(project / "configs/registry.yaml")
    if catalog.get("schema_version") != 2:
        raise ValueError("V2 registry schema required")
    models = {}
    for entry in catalog.get("sealed_baselines", []):
        spec = _sealed_baseline(project, entry)
        models[spec.model_id] = spec
    for pattern in catalog.get("discovery", []):
        for path in sorted(project.glob(pattern)):
            config = read_yaml(path)
            if config.get("model_id") in models:
                raise ValueError(f"Duplicate model ID: {config.get('model_id')}")
            spec = ModelSpec(config, path, path.parent)
            models[config.get("model_id")] = spec
    return models


def validate_spec(spec: ModelSpec) -> list[str]:
    config = spec.config
    errors = [f"Missing model fields: {sorted(REQUIRED - config.keys())}"] if REQUIRED - config.keys() else []
    if config.get("schema_version") != 2:
        errors.append("schema_version must equal 2")
    if config.get("role") not in ROLES:
        errors.append("Unknown model role")
    model_id = config.get("model_id")
    if not isinstance(model_id, str) or not model_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in model_id):
        errors.append("model_id must contain lowercase letters, digits and underscores")
    if not spec.package_path.is_dir():
        errors.append("Model package directory is missing")
    if config.get("role") != "baseline":
        if spec.package_path.name != model_id:
            errors.append("Model ID and independent directory must match")
        for filename in ("network.py", "data_loader.py", "preprocessing.py", "training.py",
                         "inference.py", "README.md", "provenance.json", "__init__.py"):
            if not (spec.package_path / filename).is_file():
                errors.append(f"Missing independent model file: {filename}")
        if "arc_model.implementations" in config.get("factory", ""):
            errors.append("V2 factory must resolve to the independent instance")
        expected_package = f"arc_model.model_instances.{model_id}"
        if config.get("package") != expected_package:
            errors.append("Package must identify this independent model instance")
        factory_module = config.get("factory", "").split(":", 1)[0]
        if factory_module != expected_package and not factory_module.startswith(expected_package + "."):
            errors.append("Factory must be defined inside this independent model directory")
        for path in spec.package_path.rglob("*.py"):
            try:
                source = ast.parse(path.read_text(), filename=str(path))
            except SyntaxError as exc:
                errors.append(f"Invalid Python in {path.name}: {exc.msg}")
                continue
            for node in ast.walk(source):
                imported = []
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    imported = [node.module or ""]
                for name in imported:
                    if name.startswith(("arc_model.implementations", "arc_model.models", "arc_model.methods")):
                        errors.append(f"Forbidden shared model import in {path.name}: {name}")
                    elif name.startswith("arc_model.model_instances") and name != expected_package and not name.startswith(expected_package + "."):
                        errors.append(f"Cross-instance import in {path.name}: {name}")
    for field in ("network", "input", "training", "evaluation", "provenance"):
        if not isinstance(config.get(field), dict):
            errors.append(f"{field} must be an explicit mapping")
    if ":" not in config.get("factory", ""):
        errors.append("factory must use module:callable notation")
    return errors


def validate_registry(root: str | Path | None = None) -> dict[str, Any]:
    models = discover_models(root)
    problems = {key: errors for key, spec in models.items() if (errors := validate_spec(spec))}
    return {"status": "passed" if not problems else "failed", "models": len(models),
            "roles": {role: sum(s.config["role"] == role for s in models.values()) for role in sorted(ROLES)},
            "errors": problems, "private_data_required": False}


def build_model(model_id: str, *, root: str | Path | None = None, **kwargs):
    models = discover_models(root)
    if model_id not in models:
        raise ValueError(f"Unknown model instance: {model_id}")
    spec = models[model_id]
    if errors := validate_spec(spec):
        raise ValueError("; ".join(errors))
    module, name = spec.config["factory"].split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    options = dict(spec.config.get("factory_kwargs", {}))
    options.update(kwargs)
    return factory(**options)

