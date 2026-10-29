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
    "docs/PHASE_MODULE_DESIGN.md",
    "docs/CODEBASE_GUIDE.md",
    "docs/GIT_WORKFLOW.md",
    "docs/THIRD_PARTY_NOTICES.md",
    "docs/INDEPENDENT_MODELS.md",
    "docs/CHANGELOG.md",
    "scripts/phase_module.py",
    "scripts/export_code_snapshot.py",
    "scripts/_bootstrap.py",
    "scripts/validate_input_frontend.py",
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
