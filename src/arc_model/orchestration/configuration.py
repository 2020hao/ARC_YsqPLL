"""Portable paths and canonical configuration identities for current experiments."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


def find_project_root(start: str | Path | None = None) -> Path:
    """Find a checkout by markers, without assuming a package nesting depth."""
    supplied = start if start is not None else os.environ.get("ARC_PROJECT_ROOT")
    candidates = [Path(supplied)] if supplied is not None else [Path.cwd(), Path(__file__)]
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file():
            candidate = candidate.parent
        for folder in (candidate, *candidate.parents):
            if (folder / "pyproject.toml").is_file() and (folder / "src/arc_model").is_dir():
                return folder
    raise ValueError("Project checkout not found; pass --root or set ARC_PROJECT_ROOT")


def read_yaml(path: str | Path) -> dict[str, Any]:
    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"Duplicate configuration key: {key!r} in {path}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    document = yaml.load(Path(path).read_text(), Loader=UniqueLoader)
    if not isinstance(document, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return document


def canonical_hash(document: Any) -> str:
    payload = json.dumps(document, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_variables(document: Any, values: Mapping[str, str] | None = None) -> Any:
    """Expand explicit ${NAME} values and fail rather than guessing data paths."""
    variables = dict(os.environ)
    variables.update(values or {})

    def replace(match):
        name = match.group(1)
        if name not in variables or not str(variables[name]):
            raise ValueError(f"Unconfigured path variable: {name}")
        return str(variables[name])

    if isinstance(document, str):
        return re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", replace, document)
    if isinstance(document, list):
        return [resolve_variables(item, variables) for item in document]
    if isinstance(document, dict):
        return {key: resolve_variables(value, variables) for key, value in document.items()}
    return document


def local_variables(root: Path, path: str | Path | None = None) -> dict[str, str]:
    location = Path(path) if path is not None else root / "configs/local/paths.yaml"
    if not location.exists():
        return {}
    document = read_yaml(location)
    values = document.get("variables", {})
    if not isinstance(values, dict) or any(not isinstance(value, str) for value in values.values()):
        raise ValueError("Local paths must define string values under variables")
    # Explicit environment settings take precedence over a workstation's local file.
    return {key: os.environ.get(key, value) for key, value in values.items()}

