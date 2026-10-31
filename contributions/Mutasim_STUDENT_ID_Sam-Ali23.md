# Mutasim — Member 1: dataset and preprocessing

| Field | Value |
| --- | --- |
| Student ID | `<STUDENT_ID>` (filled in by the member) |
| GitHub | [@Sam-Ali23](https://github.com/Sam-Ali23) |
| Git author name(s) | `mutesim` (GitHub profile name: Almutesim Alakili) |
| Branch | `feat/m1-data-preprocessing` |
| Pull request | [#1](https://github.com/Rxd3/RECAP/pull/1) — feat(data): validate SponSpeech preprocessing and corrected v2 release (merged on 2026-10-29) |

## Contributions in this repository

- Built the SponSpeech data pipeline and release (eaed76b): source extraction, labeling, preprocessing, release building, split assignment and validation in `src/punct_restore/data/`, the `scripts/prepare_data.py` CLI, and the data docs and contract tests.
- PR #1 also carries five v2-correction commits by Rxd3 (d30df5e, f259b4a, 9b91467, 1e638d3, 263472e); they are listed in `Rasit_STUDENT_ID_Rxd3.md`.

## Commits

| Commit | Date | Subject |
| --- | --- | --- |
| `eaed76b` | 2026-10-29 | feat(data): complete SponSpeech preprocessing and dataset release |

## Files added or changed

- `configs/`: `datasets.yaml` (M in eaed76b), `label_sets.yaml` (M in eaed76b)
- `data/`: `README.md` (M in eaed76b)
- `docs/`: `DATASET_PLAN.md` (M in eaed76b), `DATA_CONTRACT.md` (M in eaed76b), `DECISIONS.md` (M in eaed76b)
- `scripts/`: `prepare_data.py` (M in eaed76b)
- `src/punct_restore/data/`: `__init__.py` (M in eaed76b), `extraction.py` (A in eaed76b), `labeling.py` (M in eaed76b), `preprocessing.py` (M in eaed76b), `release.py` (A in eaed76b), `splits.py` (A in eaed76b), `validation.py` (A in eaed76b)
- `tests/`: `test_data_contract.py` (M in eaed76b)

Generated from `git log HEAD --author="mutesim"` at commit `4b2c616` of `release/final` on 2026-10-31. Only the member's own commits are listed; later changes by others are not shown. The member fills in the student ID.
