"""Export an allowlisted portable public tree, without raw data or private history."""

from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    ".gitignore",
    ".gitattributes",
    "README.md",
    "pyproject.toml",
    "requirements-verified.txt",
    ".github/workflows/phase-module.yml",
    "configs/modules/*.yaml",
    "configs/models.yaml",
    "configs/datasets.yaml",
    "configs/experiment_config.yaml",
    "src/arc_model/implementations/catalog.json",
    "src/arc_model/implementations/*/README.md",
    "src/arc_model/implementations/*/provenance.json",
    "src/arc_model/implementations/*/performance.json",
    "releases/baselines_v1_20261030/*",
    "docs/NINE_MODELS_AND_CONFIG.md",
    "docs/AUTHOR_ASSET_AVAILABILITY_20261030.md",
    "scripts/check_baseline_release.py",
    "scripts/run_experiment_suite.py",
    "scripts/qualify_fenet_20261030.py",
    "scripts/qualify_fenet_prepare_20261030.py",
    "scripts/fenet_vmd_graph_20261030.py",
    "scripts/fenet_stream_prepare_20261030.py",
    "scripts/summarize_fenet_qualification_20261030.py",
    "scripts/qualify_fusion_current_20261030.py",
    "docs/BASELINE_QUALIFICATION_20261030.md",
    "docs/DATA_SPLIT_PROTOCOL_20261030.md",
    "scripts/seal_baseline_release.py",
    "scripts/audit_nine_model_baselines.py",
    "docs/PHASE_MODULE_DESIGN.md",
    "docs/CODEBASE_GUIDE.md",
    "docs/GIT_WORKFLOW.md",
    "docs/THIRD_PARTY_NOTICES.md",
    "docs/INDEPENDENT_MODELS.md",
    "docs/CHANGELOG.md",
    "docs/SHARED_EIGHT_STUDY.md",
    "docs/SHARED_EIGHT_E2E.md",
    "docs/DYNAMIC_CORRECTION_RESTORE.md",
    "docs/BASELINE_DEFINITION.md",
    "docs/DATASETS.md",
    "docs/BRANCH_VS_REPLACEMENT.md",
    "docs/CLASSIFICATION_METRICS.md",
    "docs/HYDRA_ALTERNATING.md",
    "docs/GROUPED_BASELINE_STANDARD.md",
    "docs/BASELINE_FIDELITY_v2.json",
    "scripts/phase_module.py",
    "scripts/export_code_snapshot.py",
    "scripts/_bootstrap.py",
    "scripts/validate_input_frontend.py",
    "scripts/validate_shared_eight.py",
    "scripts/summarize_shared_eight.py",
    "scripts/train_shared_eight_e2e.py",
    "scripts/summarize_shared_eight_e2e.py",
    "scripts/audit_shared_eight_e2e.py",
    "scripts/report_shared_eight_e2e.py",
    "scripts/validate_dynamic_restore.py",
    "scripts/summarize_dynamic_restore.py",
    "scripts/export_dynamic_modules.py",
    "scripts/audit_current_datasets.py",
    "scripts/complete_paper_baselines.py",
    "scripts/summarize_paper_baselines.py",
    "scripts/audit_paper_baseline_forward.py",
    "scripts/report_classification_metrics.py",
    "scripts/validate_hydra_alternating.py",
    "scripts/report_hydra_alternating.py",
    "scripts/freeze_grouped_baselines.py",
    "scripts/summarize_grouped_baselines.py",
    "scripts/report_grouped_baselines.py",
    "scripts/audit_grouped_baseline_forward.py",
    "scripts/audit_fenet_numerics.py",
    "src/**/*.py",
    "tests/phase_module/*.py",
    "third_party/current_reference_recovery_v2/*.py",
    "third_party/current_reference_recovery_v2/pll_spec.json",
    "third_party/hydra/README.txt",
    "third_party/hydra/hydra_class.py",
    "third_party/hydra/hydra_features.py",
    "third_party/hydra/author_source_integrity.json",
]
SECRET = re.compile(
    r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|sk-[A-Za-z0-9]{40,}"
)


def public_files():
    paths = sorted(
        {
            p
            for pattern in PATTERNS
            for p in ROOT.glob(pattern)
            if p.is_file() and "__pycache__" not in p.parts
        }
    )
    for p in paths:
        if p.is_symlink() or p.stat().st_size > 1024 * 1024:
            raise ValueError(f"Unexpected symlink/large file: {p.relative_to(ROOT)}")
        if SECRET.search(p.read_text()):
            raise ValueError(f"Credential-like text: {p.relative_to(ROOT)}")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if out == ROOT or out in ROOT.parents or (out.exists() and any(out.iterdir())):
        raise ValueError("Use a new empty export directory")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in public_files():
        rel = p.relative_to(ROOT)
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        rows.append(
            dict(
                path=str(rel),
                bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            )
        )
    (out / "CODE_SNAPSHOT_MANIFEST.json").write_text(json.dumps(rows, indent=2))
    print(
        json.dumps(dict(status="exported_not_uploaded", files=len(rows), output=str(out)), indent=2)
    )


if __name__ == "__main__":
    main()
