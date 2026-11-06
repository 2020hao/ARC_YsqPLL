# Git / GitHub workflow

Remote: https://github.com/2020hao/ARC_YsqPLL . The repository was public when
initialized; the connected repository is private as verified on2026-11-06.
The portable-code-only publication policy still applies. Git manages local changes; GitHub hosts the portable code snapshot.

Raw data, labels, weights, manuscripts and metrics remain outside versioned code.
Use `scripts/export_code_snapshot.py` to inspect/export an allowlisted code tree
before public synchronization. It writes a manifest with SHA256 fingerprints and
rejects oversized files and common credential patterns. It is not an uploader and
contains no token. Third-party declarations remain part of the export.

The local `research-baseline-20261029` tag and `research/local-main` branch retain
the private pre-refactor state. Do not push all branches or tags to the public
repository. Normal code releases should follow only the public main history.

Recommended day-to-day workflow after Git credentials are configured:

```bash
git switch -c feature/short-description
# implement, run targeted tests, inspect the diff
git add src tests configs docs
git commit -m "Describe the concrete change"
git push -u origin HEAD
# open a pull request targeting main
```

Do not commit raw/derived arrays just because they make reproduction convenient.
Record external data source, native sampling rate, current channel, annotation
revision, split indices and checksums in each local experiment receipt. Freeze
code/version before training; a code refactor does not upgrade prior evidence.
