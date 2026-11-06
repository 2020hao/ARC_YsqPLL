"""Write explicit run identities; never infer experiment completion from a folder."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .configuration import canonical_hash


def _component(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_\-.]+", value) or value in (".", ".."):
        raise ValueError("Run path identifiers must be simple names without path traversal")
    return value


@dataclass(frozen=True)
class RunIdentity:
    study_id: str
    protocol_id: str
    model_id: str
    fold: int
    seed: int
    model_config_hash: str
    data_identity_hash: str
    split_hash: str
    source_hash: str
    feature_schema_hash: str
    parent_checkpoint_hash: str | None = None

    def __post_init__(self):
        for key in ("study_id", "protocol_id", "model_id"):
            _component(getattr(self, key))
        if self.fold < 0 or self.seed < 0:
            raise ValueError("fold and seed must be nonnegative")
        for key in ("model_config_hash", "data_identity_hash", "split_hash", "source_hash", "feature_schema_hash"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError(f"Explicit run identity required: {key}")

    @property
    def digest(self):
        return canonical_hash(asdict(self))

    def path(self, artifacts_root):
        return Path(artifacts_root) / "runs" / self.study_id / self.protocol_id / self.model_id / f"fold-{self.fold}" / f"seed-{self.seed}"


def initialize_run(artifacts_root, identity: RunIdentity, resolved_config):
    """Create a prepared receipt, or reject an incompatible resume in that directory."""
    directory = identity.path(artifacts_root)
    receipt = {"status": "prepared_not_trained", "identity": asdict(identity),
               "identity_hash": identity.digest, "resolved_config_hash": canonical_hash(resolved_config)}
    target = directory / "run_manifest.json"
    if target.exists():
        previous = json.loads(target.read_text())
        if previous["identity_hash"] != identity.digest or previous["resolved_config_hash"] != receipt["resolved_config_hash"]:
            raise ValueError("Run identity changed; use a new study/model/protocol version")
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False))
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return directory

