"""Local YAML reader; this implementation imports no peer model package."""

from pathlib import Path
import hashlib
import json
import yaml

CONFIG_KEY = "ArcFENet_Baseline"


def load_config(path=None):
    p = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[4] / "configs/models.yaml"
    )
    with p.open() as f:
        document = yaml.safe_load(f)
    if not isinstance(document, dict) or CONFIG_KEY not in document:
        raise ValueError("Missing model config: " + CONFIG_KEY)
    cfg = document[CONFIG_KEY]
    if cfg["implementation"] != "baseline_arc_fenet":
        raise ValueError("Config implementation mismatch")
    root = Path(__file__).resolve().parents[4]
    release = root / "releases/baselines_v1_20261030"
    if (release / "SEALED.json").exists():
        lock_path = release / "baseline.lock.json"
        digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        if digest != json.loads((release / "SEALED.json").read_text())["baseline_lock_sha256"]:
            raise ValueError("Baseline release lock changed")
        lock = json.loads(lock_path.read_text())
        canonical = json.dumps(
            cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        if (
            hashlib.sha256(canonical.encode()).hexdigest()
            != lock["baseline_yaml_nodes"][CONFIG_KEY]
        ):
            raise ValueError("Frozen baseline configuration changed: " + CONFIG_KEY)
        package = Path(__file__).resolve().parent.name
        for row in lock["files"]:
            if Path(row["path"]).parent.name == package:
                source = root / row["path"]
                if (
                    not source.is_file()
                    or hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]
                ):
                    raise ValueError("Frozen baseline source changed: " + row["path"])
    return cfg
