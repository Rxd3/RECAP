# Data storage and release plan

Owner: Member 1. No datasets or generated split files are present.

- `raw/`: immutable source downloads, source revision and license evidence.
- `processed/`: versioned JSONL releases following the [contract](../docs/DATA_CONTRACT.md).
- `splits/`: immutable sample/group assignments and release checksum manifests.

These directories retain markers in Git; their generated contents are ignored.
Keep a small reviewed release description in documentation. Share actual data
and manifests through an agreed versioned artifact location after verifying
usage conditions; record the location and SHA-256 in the release description.
Do not add large data with `git add -f`.

Download instructions are pending source verification. Start with the two source
pages in [DATASET_PLAN.md](../docs/DATASET_PLAN.md), inspect metadata and a small
permitted sample, and record exact configuration, fields, upstream revision,
license, and download command there. Do not treat placeholder nulls as loader
arguments or execute an unreviewed remote loader script.

TODO: approved download steps; cleaning/tokenization/label policies; exclusions
and rejection log; group-aware split release; duplicate audit; statistics and
class distributions; clean/noisy provenance; environment and regeneration command.
