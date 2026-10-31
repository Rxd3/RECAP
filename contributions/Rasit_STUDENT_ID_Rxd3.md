# Raşit — Member 2: HMM baseline and team lead

| Field | Value |
| --- | --- |
| Student ID | `<STUDENT_ID>` (filled in by the member) |
| GitHub | [@Rxd3](https://github.com/Rxd3) |
| Git author name(s) | `Rxd3`, `Rasit isaoglu` |
| Branch | `feat/m2-hmm-baseline` |
| Pull request | [#4](https://github.com/Rxd3/RECAP/pull/4) — Complete validated HMM baseline and SponSpeech v2 experiments (open on 2026-10-30; merged into `release/final` by 0790467) |

## Contributions in this repository

- Created the project scaffold (71e55ae) and prepared it for GitHub (56babb9).
- Wrote the HMM baseline: `src/punct_restore/models/hmm.py`, `scripts/train_hmm.py` and their tests (2b530a2).
- Added the validated HMM selection, experiment and reproduction workflow: `hmm_workflow.py`, `predict_hmm.py`, `run_hmm_experiments.py`, `verify_hmm_experiments.py`, `configs/hmm_sponspeech_v2.yaml`, `requirements-hmm.txt` (e4c3183); recorded the verified results and the handoff packager (34849b9, 10bbaa8). The completion audit he added in 34849b9 was later folded into `experiments/HMM_RESULTS.md` and removed in 766aeb7.
- Made the v2 data-release corrections that shipped in PR #1 (d30df5e, f259b4a, 9b91467, 1e638d3, 263472e) and merged PR #1 (1810c48).

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

- `71e55ae`: the project scaffold, 67 files added (not listed one by one)
- repository root: `CONTRIBUTING.md` (M in 10bbaa8, 2b530a2), `README.md` (M in 34849b9, e4c3183, 1e638d3, 2b530a2, 56babb9), `TEAM_TASKS.md` (M in 34849b9, 1e638d3, 2b530a2), `pyproject.toml` (M in 1e638d3, 56babb9), `requirements-hmm.txt` (A in e4c3183)
- `artifacts/`: `README.md` (M in 10bbaa8)
- `configs/`: `datasets.yaml` (M in d30df5e), `hmm_sponspeech_v2.yaml` (A in e4c3183), `label_sets.yaml` (M in d30df5e)
- `data/`: `README.md` (M in 1e638d3)
- `docs/`: 12 files (2 added); commits 10bbaa8, 1e638d3, 263472e, 2b530a2, 34849b9, 56babb9
- `experiments/`: `HMM_NOTES.md` (M in 34849b9, e4c3183, 2b530a2), `HMM_RESULTS.md` (A in 34849b9; M in 10bbaa8), `HMM_RUNBOOK.md` (A in e4c3183; M in 34849b9), `README.md` (M in 34849b9)
- `presentation/`: `README.md` (M in 10bbaa8)
- `results/`: `README.md` (M in 10bbaa8)
- `scripts/`: `README.md` (M in 34849b9, e4c3183, 1e638d3, 2b530a2), `package_hmm_handoff.py` (A in 34849b9), `predict_hmm.py` (A in e4c3183), `prepare_data.py` (M in d30df5e), `run_hmm_experiments.py` (A in e4c3183), `train_hmm.py` (M in e4c3183, 2b530a2), `verify_hmm_experiments.py` (A in e4c3183)
- `src/punct_restore/`: `__init__.py` (M in 56babb9)
- `src/punct_restore/data/`: `extraction.py` (M in d30df5e), `labeling.py` (M in d30df5e), `preprocessing.py` (M in 9b91467, f259b4a, d30df5e), `release.py` (M in f259b4a, d30df5e), `splits.py` (M in d30df5e)
- `src/punct_restore/models/`: `__init__.py` (M in 2b530a2), `hmm.py` (M in 2b530a2), `hmm_workflow.py` (A in e4c3183)
- `tests/`: `README.md` (M in e4c3183, 1e638d3, 2b530a2), `test_data_contract.py` (M in d30df5e), `test_data_release_regressions.py` (A in d30df5e; M in f259b4a), `test_hmm.py` (A in 2b530a2), `test_hmm_cli.py` (A in 2b530a2; M in e4c3183), `test_hmm_experiments.py` (A in e4c3183), `test_hmm_workflow.py` (A in e4c3183), `test_model_contract.py` (M in 2b530a2)

Generated from `git log HEAD --author="Rxd3\|Rasit"` at commit `f889d0f` of `release/final` on 2026-10-31. Only the member's own commits are listed; later changes by others are not shown. The member fills in the student ID.
