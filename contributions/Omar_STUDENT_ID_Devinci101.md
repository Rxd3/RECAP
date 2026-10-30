# Omar Alzyod — Member 3: CRF main model

| Field | Value |
| --- | --- |
| Student ID | `<STUDENT_ID>` (filled in by the member) |
| GitHub | [@Devinci101](https://github.com/Devinci101) |
| Git author name(s) | `Devinci101`, `Omar Alzyod` |
| Branch | `feat/m3-crf-model` |
| Pull request | none (no pull request was opened). Merge commit 611f3f4 merged the branch into `release/final`. |

## Contributions in this repository

- Implemented the CRF model (`src/punct_restore/models/crf.py`), the CRF CLI (`scripts/train_crf.py`) and the CRF feature extractor (`src/punct_restore/features/crf_features.py`) (aa232be).
- Implemented the improved features: the `improved` preset is in `crf_features.py` (aa232be).
- Added `tests/test_crf_features.py`, `tests/test_crf_model.py` and `tests/test_train_crf_cli.py`; changed `experiments/CRF_NOTES.md` and `experiments/FEATURE_NOTES.md` (aa232be).
- Added `docs/CRF_EXPERIMENTS.md` and `experiments/runs/recap_m3_handoff_eval/README.md` (c3b0f8a).
- Updated `TEAM_TASKS.md` (9015b62).
- Shared the six trained CRF models (`.joblib`, with a `SHA256SUMS` file that verified). Member 5 reproduced all six validation prediction files from them byte for byte and used them for the test and test_amb predictions (`docs/EVALUATION_RESULTS_CRF.md`, Provenance).

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
- `experiments/runs/recap_m3_handoff_eval/`: `README.md` (A in c3b0f8a)
- `scripts/`: `train_crf.py` (M in aa232be)
- `src/punct_restore/features/`: `crf_features.py` (M in aa232be)
- `src/punct_restore/models/`: `crf.py` (M in aa232be)
- `tests/`: `test_crf_features.py` (A in aa232be), `test_crf_model.py` (A in aa232be), `test_train_crf_cli.py` (A in aa232be)

Source: `git log HEAD --author="Devinci101\|Omar"` (HEAD is `ddba805`, the `release/final` merge) on 2026-10-30. The member fills in the student ID.
