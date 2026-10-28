# Initial verification — 2026-10-28

This records scaffold checks only, not model/feature/evaluation test results.

Observed inventory: 67 scaffold files, 24 Python placeholders, 3 YAML configs,
6 discoverable Python packages and 45 resolved local documentation links.
Ignore probes covered 24 excluded paths and 73 retained file/marker paths,
including hypothetical nested documentation locations. Python 3.13.5 was used
for syntax/configuration inspection; no packages were installed.

## Safe checks completed

- [x] Preferred directories, task scripts, guides and directory markers exist.
- [x] Every package directory contains `__init__.py`.
- [x] All Python files parse; placeholders state responsibility, owner and TODOs.
- [x] TOML and all YAML files parse; package discovery uses `src/`.
- [x] Standard/extended punctuation have 4/8 unique labels; C is exactly the
  ordered 4×3 product with 12 unique joint labels and no 24-class target.
- [x] Proposed collapse covers all extended labels, targets standard labels and
  remains explicitly unapproved; dataset configuration remains unverified.
- [x] Shared data/model/feature/evaluation contracts and all six roles/branches
  are documented with ownership, dependencies, acceptance criteria and handoffs.
- [x] Relative documentation links resolve to created files.
- [x] Git ignores representative datasets, processed/split outputs, model files,
  caches, environments, secrets and generated results, while retaining markers/docs.
- [x] Results/output directories contain only documentation/empty markers;
  no data downloads, training, predictions or numerical results were produced.
- [x] No package installation, GPU framework setup or pretrained download occurred.
- [x] Empty independent workspace used; no unrelated files or NALTRA writes.

## Local Git setup

- [x] Initialized this independent directory with `main`.
- [x] Used the existing configured Git identity; no identity was invented/changed.
- [x] Created the initial scaffolding commit locally.
- [x] Created all six feature branches from that same commit; left `main` checked out.
- [x] Verified a clean working tree, no configured remote and no pushes.

The exact commit can be retrieved with `git rev-parse main`; it is not embedded
here to avoid a self-referential commit hash. Branch names are in
[GIT_WORKFLOW.md](GIT_WORKFLOW.md). No Git operation was unavailable during setup.

## Unresolved implementation work

Annotation/tokenization/casing edge cases, source/license/download verification,
split compatibility/leakage, B→A approval, CRF feature agreement, trained models,
real metrics and end-to-end inference are pending. See the decision register and
team board. Runtime dependency compatibility and an experiment lock are not yet
established. The test modules are plans only; no functional model tests were run
or claimed. Publication awaits a separately authorized Project 2 remote.
