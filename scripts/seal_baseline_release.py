"""One-time seal of current baseline source/config/metrics; never edits scores."""

import hashlib
import json
import _bootstrap as boot
from arc_model.implementations.registry import load_catalog, canonical_hash

ROOT = boot.ROOT
VERSION = "baselines_v1_20261030"


def main():
    release = ROOT / "releases" / VERSION
    if (release / "SEALED.json").exists():
        raise RuntimeError("Baseline release already sealed; an update needs a NEW version")
    audit = ROOT / "outputs/nine_model_refactor_20261030"
    audit.mkdir(parents=True, exist_ok=True)
    old = release / "baseline.lock.json"
    if old.exists():
        (audit / "pre_release_lock.json").write_bytes(old.read_bytes())
    specs = load_catalog()
    files = []
    nodes = {}
    for name, spec in specs.items():
        if spec.role != "baseline":
            continue
        folder = ROOT / "src/arc_model/implementations" / spec.implementation
        for p in sorted(folder.iterdir()):
            if p.is_file():
                files.append(
                    dict(
                        path=str(p.relative_to(ROOT)),
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    )
                )
        nodes[name] = canonical_hash(spec.values)
    lock = dict(
        release=VERSION,
        status="code_and_reported_metrics_locked",
        paper_fidelity_confirmation=dict(
            arc_fenet="conditional_pending",
            hydra="author_configuration_supported",
            arc_fusion="current_only_not_complete_author_method",
        ),
        revision_policy="Never overwrite this release; fixes need an explicit new version and paired baseline/enhanced reevaluation",
        files=files,
        baseline_yaml_nodes=nodes,
        metrics_source="outputs/grouped_baseline_standard_20261029/main_results.csv",
        refactor_verification="outputs/nine_model_refactor_20261030; baseline metrics are not rerun or edited by this seal",
    )
    text = json.dumps(lock, ensure_ascii=False, indent=2) + "\n"
    old.write_text(text)
    digest = hashlib.sha256(old.read_bytes()).hexdigest()
    (release / "baseline.lock.sha256").write_text(digest + "\n")
    seal = dict(
        release=VERSION,
        baseline_lock_sha256=digest,
        baseline_packages=3,
        active_model_config="configs/models.yaml",
        protected_baseline_nodes=list(nodes),
        candidate_nodes_mutable=True,
        fidelity_confirmation=lock["paper_fidelity_confirmation"],
    )
    (release / "SEALED.json").write_text(json.dumps(seal, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(seal, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
