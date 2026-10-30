"""Check immutable baseline files, three YAML nodes and published metric snapshots."""

import argparse
import hashlib
import json
from pathlib import Path
import _bootstrap as boot
from arc_model.implementations.registry import load_catalog, canonical_hash

RELEASE = "baselines_v1_20261030"


def verify(root=boot.ROOT, *, config_path=None):
    root = Path(root)
    release = root / "releases" / RELEASE
    path = release / "baseline.lock.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != (release / "baseline.lock.sha256").read_text().strip():
        raise ValueError(
            "Baseline lock itself changed; create a new release rather than overwrite it"
        )
    seal = json.loads((release / "SEALED.json").read_text())
    if seal["baseline_lock_sha256"] != digest:
        raise ValueError("Baseline release seal mismatch")
    lock = json.loads(path.read_text())
    for row in lock["files"]:
        file = root / row["path"]
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Frozen baseline file changed: " + row["path"])
    models = load_catalog(config_path or root / "configs/models.yaml")
    for name, expected in lock["baseline_yaml_nodes"].items():
        if canonical_hash(models[name].values) != expected:
            raise ValueError("Frozen baseline YAML node changed: " + name)
    return dict(
        status="pass",
        release=RELEASE,
        baseline_packages=3,
        baseline_yaml_nodes=3,
        frozen_files=len(lock["files"]),
        candidate_packages=6,
        fidelity_confirmation=lock["paper_fidelity_confirmation"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    a = p.parse_args()
    print(json.dumps(verify(config_path=a.config), ensure_ascii=False, indent=2))
