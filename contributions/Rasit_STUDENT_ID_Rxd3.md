# Raşit — Member 2: HMM baseline and team lead

| Field | Value |
| --- | --- |
| Student ID | `<STUDENT_ID>` (filled in by the member) |
| GitHub | [@Rxd3](https://github.com/Rxd3) |
| Git author name(s) | `Rxd3`, `Rasit isaoglu` |
| Branch | `feat/m2-hmm-baseline` |
| Pull request | [#4](https://github.com/Rxd3/RECAP/pull/4) — Complete validated HMM baseline and SponSpeech v2 experiments (open on 2026-10-30) |

## Contributions in this repository

- Created the project scaffold: 67 files added in 71e55ae. Commit 56babb9 then changed 6 files (subject: name project RECAP and prepare GitHub publication).
- Wrote the HMM baseline: `models/hmm.py`, `scripts/train_hmm.py`, `tests/test_hmm.py`, `tests/test_hmm_cli.py` (2b530a2).
- Added the HMM selection, experiment and reproduction workflow: `models/hmm_workflow.py`, `scripts/predict_hmm.py`, `scripts/run_hmm_experiments.py`, `scripts/verify_hmm_experiments.py`, `configs/hmm_sponspeech_v2.yaml`, `requirements-hmm.txt` (e4c3183). Added `experiments/HMM_RESULTS.md`, `docs/MEMBER2_COMPLETION.md` and `scripts/package_hmm_handoff.py` (34849b9). Changed 9 files in 10bbaa8, including `experiments/HMM_RESULTS.md` and `docs/MEMBER2_COMPLETION.md`.
- Made the v2 data-release corrections on `feat/m1-data-preprocessing` (d30df5e, f259b4a, 9b91467, 1e638d3, 263472e). These five commits are part of PR #1.
- Merged PR #1 (1810c48) and merged that result into the HMM branch (7d735f1).

## Commits

| Commit | Date | Subject |
| --- | --- | --- |
| `10bbaa8` | 2026-10-29 | Verify portable HMM handoff and record completion evidence |
| `34849b9` | 2026-10-29 | Record verified HMM results and portable handoff procedure |
| `e4c3183` | 2026-10-29 | Add validated HMM selection, experiment and reproduction workflow |
| `263472e` | 2026-10-29 | Record verified data handoff archive fingerprint |
| `1e638d3` | 2026-10-29 | Document verified SponSpeech v2 release and runtime setup |
| `9b91467` | 2026-10-29 | Keep small source checks sequential |
| `f259b4a` | 2026-10-29 | Bound source I/O and stream cross-view release validation |
| `d30df5e` | 2026-10-29 | Fix portable data releases and unresolved split selection |
| `2b530a2` | 2026-10-28 | feat(hmm): implement supervised baseline and dataset-independent checks |
| `56babb9` | 2026-10-28 | chore: name project RECAP and prepare GitHub publication |
| `71e55ae` | 2026-10-28 | chore: initialize sequence labeling project scaffold |

Merge commits:

| Commit | Date | Subject |
| --- | --- | --- |
| `7d735f1` | 2026-10-29 | Merge validated SponSpeech v2 data release into HMM branch |
| `1810c48` | 2026-10-29 | Merge pull request #1 from Rxd3/feat/m1-data-preprocessing |

## Files added or changed

- `71e55ae`: 67 files, all status A (project scaffold). Later changes to those files are shown below as M.
- repository root: `CONTRIBUTING.md` (M in 10bbaa8, 2b530a2), `README.md` (M in 34849b9, e4c3183, 1e638d3, 2b530a2, 56babb9), `TEAM_TASKS.md` (M in 34849b9, 1e638d3, 2b530a2), `pyproject.toml` (M in 1e638d3, 56babb9), `requirements-hmm.txt` (A in e4c3183)
- `artifacts/`: `README.md` (M in 10bbaa8)
- `configs/`: `datasets.yaml` (M in d30df5e), `hmm_sponspeech_v2.yaml` (A in e4c3183), `label_sets.yaml` (M in d30df5e)
- `data/`: `README.md` (M in 1e638d3)
- `docs/`: `ARCHITECTURE.md` (M in 10bbaa8, 2b530a2), `DATASET_PLAN.md` (M in 1e638d3), `DATA_CONTRACT.md` (M in 1e638d3), `DATA_RELEASE_VALIDATION.md` (A in 1e638d3; M in 263472e), `DECISIONS.md` (M in 34849b9, 1e638d3), `GIT_WORKFLOW.md` (M in 10bbaa8, 56babb9), `INITIAL_VERIFICATION.md` (M in 56babb9), `MEMBER2_COMPLETION.md` (A in 34849b9; M in 10bbaa8), `MODEL_CONTRACT.md` (M in 34849b9, 2b530a2), `PROJECT_SCOPE.md` (M in 10bbaa8, 2b530a2, 56babb9), `README.md` (M in 1e638d3), `REPRODUCIBILITY.md` (M in 34849b9, 2b530a2)
- `experiments/`: `HMM_NOTES.md` (M in 34849b9, e4c3183, 2b530a2), `HMM_RESULTS.md` (A in 34849b9; M in 10bbaa8), `HMM_RUNBOOK.md` (A in e4c3183; M in 34849b9), `README.md` (M in 34849b9)
- `presentation/`: `README.md` (M in 10bbaa8)
- `results/`: `README.md` (M in 10bbaa8)
- `scripts/`: `README.md` (M in 34849b9, e4c3183, 1e638d3, 2b530a2), `package_hmm_handoff.py` (A in 34849b9), `predict_hmm.py` (A in e4c3183), `prepare_data.py` (M in d30df5e), `run_hmm_experiments.py` (A in e4c3183), `train_hmm.py` (M in e4c3183, 2b530a2), `verify_hmm_experiments.py` (A in e4c3183)
- `src/punct_restore/`: `__init__.py` (M in 56babb9)
- `src/punct_restore/data/`: `extraction.py` (M in d30df5e), `labeling.py` (M in d30df5e), `preprocessing.py` (M in 9b91467, f259b4a, d30df5e), `release.py` (M in f259b4a, d30df5e), `splits.py` (M in d30df5e)
- `src/punct_restore/models/`: `__init__.py` (M in 2b530a2), `hmm.py` (M in 2b530a2), `hmm_workflow.py` (A in e4c3183)
- `tests/`: `README.md` (M in e4c3183, 1e638d3, 2b530a2), `test_data_contract.py` (M in d30df5e), `test_data_release_regressions.py` (A in d30df5e; M in f259b4a), `test_hmm.py` (A in 2b530a2), `test_hmm_cli.py` (A in 2b530a2; M in e4c3183), `test_hmm_experiments.py` (A in e4c3183), `test_hmm_workflow.py` (A in e4c3183), `test_model_contract.py` (M in 2b530a2)

Source: `git log HEAD --author="Rxd3\|Rasit"` (HEAD is `ddba805`, the `release/final` merge) on 2026-10-30. The member fills in the student ID.
