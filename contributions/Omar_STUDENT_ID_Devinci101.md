# Omar Alzyod — Member 3: CRF main model

| Field | Value |
| --- | --- |
| Student ID | `<STUDENT_ID>` (filled in by the member) |
| GitHub | [@Devinci101](https://github.com/Devinci101) |
| Git author name(s) | `Devinci101`, `Omar Alzyod` |
| Branch | `feat/m3-crf-model` |
| Pull request | none (no pull request was opened); merged into `release/final` by 611f3f4 |

## Contributions in this repository

- Implemented the CRF model (`src/punct_restore/models/crf.py`), the CRF CLI (`scripts/train_crf.py`) and the CRF feature extractor with the `basic` and `improved` presets (`src/punct_restore/features/crf_features.py`), with their tests (aa232be).
- Trained the six full-data CRF models and documented them in `docs/CRF_EXPERIMENTS.md` (c3b0f8a). The handoff README he added in c3b0f8a (`experiments/runs/recap_m3_handoff_eval/README.md`) was later removed in 766aeb7.
- Shared the six trained `.joblib` models with a `SHA256SUMS` file that verified. Member 5 reproduced all six validation prediction files from them byte for byte and used them for the test and test_amb predictions (`docs/EVALUATION_RESULTS_CRF.md`, Provenance); his training manifests are committed in `results/scores/crf_training_manifests/` (15e5ebf).
- Updated the team board (9015b62).

## Commits

| Commit | Date | Subject |
| --- | --- | --- |
| `c3b0f8a` | 2026-10-29 | Add CRF experiment notes and evaluation guide |
| `aa232be` | 2026-10-29 | feat(crf): implement and validate CRF models for modes A/B/C |
| `9015b62` | 2026-10-28 | Update team member assignments and ownership details |

## Files added or changed

- repository root: `TEAM_TASKS.md` (M in 9015b62)
- `docs/`: `CRF_EXPERIMENTS.md` (A in c3b0f8a)
- `experiments/`: `CRF_NOTES.md` (M in aa232be), `FEATURE_NOTES.md` (M in aa232be)
- `experiments/runs/recap_m3_handoff_eval/`: `README.md` (A in c3b0f8a, later removed in 766aeb7)
- `scripts/`: `train_crf.py` (M in aa232be)
- `src/punct_restore/features/`: `crf_features.py` (M in aa232be)
- `src/punct_restore/models/`: `crf.py` (M in aa232be)
- `tests/`: `test_crf_features.py` (A in aa232be), `test_crf_model.py` (A in aa232be), `test_train_crf_cli.py` (A in aa232be)

Generated from `git log HEAD --author="Devinci101\|Omar"` at commit `4b2c616` of `release/final` on 2026-10-31. Only the member's own commits are listed; later changes by others are not shown. The member fills in the student ID.
