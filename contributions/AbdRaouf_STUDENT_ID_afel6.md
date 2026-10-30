# Abd Raouf — Member 5: evaluation and error analysis

| Field | Value |
| --- | --- |
| Student ID | `<STUDENT_ID>` (filled in by the member) |
| GitHub | [@afel6](https://github.com/afel6) |
| Git author name(s) | `Raouf` |
| Branch | `feat/m5-evaluation` |
| Pull request | [#2](https://github.com/Rxd3/RECAP/pull/2) — feat(eval): evaluation pipeline, shortcut baseline and CLI (Member 5) (open on 2026-10-30) |

## Contributions in this repository

- Added the evaluation pipeline, shortcut baseline and CLI: `evaluation/baseline.py`, `inputs.py`, `report.py`, `views.py` and 10 files under `tests/`; changed `evaluation/metrics.py` and `scripts/evaluate_models.py` (680438f).
- Added grouped bootstrap CIs, paired compare, error examples and a manifest check: `evaluation/comparison.py`, `manifest.py`, `uncertainty.py` (6356916). Recorded the gold data policy version in reports (ec37d19).
- Fixed the manifest check for the HMM bundle's nested `inputs.input` (9155a7e) and the Markdown percentage rounding (6340b59).
- Added `docs/EVALUATION_RESULTS.md` (51bd400) and `docs/EVALUATION_RESULTS_CRF.md` (255fc68); changed `docs/EVALUATION_PROTOCOL.md` (680438f, 6356916, ec37d19, 51bd400, 255fc68).
- Created six merge commits: four into `release/final` (ddba805, 611f3f4, 0790467, 097c87b), 3044a58 (main into `feat/m5-evaluation`) and aca40c6 (into `feat/m5-eval-uncertainty`).

## Commits

| Commit | Date | Subject |
| --- | --- | --- |
| `255fc68` | 2026-10-30 | docs(eval): add CRF validation results, feature ablations and hypothesis outcomes |
| `51bd400` | 2026-10-29 | docs(eval): add HMM evaluation results with confusion matrices, error analysis and runtime |
| `6340b59` | 2026-10-29 | fix(eval): round Markdown percentages half-up from the stored 6-decimal value |
| `9155a7e` | 2026-10-29 | fix(eval): accept the HMM bundle's nested inputs.input in the manifest check |
| `ec37d19` | 2026-10-29 | feat(eval): record the gold data policy version in reports |
| `6356916` | 2026-10-29 | feat(eval): add grouped bootstrap CIs, paired compare, error examples and manifest check |
| `680438f` | 2026-10-29 | feat(eval): add evaluation pipeline, shortcut baseline and CLI |

Merge commits:

| Commit | Date | Subject |
| --- | --- | --- |
| `ddba805` | 2026-10-30 | chore: merge feat/m6-integration-presentation into release/final |
| `611f3f4` | 2026-10-30 | chore: merge feat/m3-crf-model into release/final |
| `0790467` | 2026-10-30 | chore: merge feat/m2-hmm-baseline into release/final |
| `097c87b` | 2026-10-30 | chore: merge feat/m5-evaluation into release/final |
| `aca40c6` | 2026-10-29 | chore: merge feat/m5-evaluation (main with m1-v2 data) into feat/m5-eval-uncertainty |
| `3044a58` | 2026-10-29 | chore: merge main into feat/m5-evaluation |

## Files added or changed

- `docs/`: `EVALUATION_PROTOCOL.md` (M in 255fc68, 51bd400, ec37d19, 6356916, 680438f), `EVALUATION_RESULTS.md` (A in 51bd400; M in 255fc68), `EVALUATION_RESULTS_CRF.md` (A in 255fc68)
- `results/`: `README.md` (M in 680438f)
- `scripts/`: `evaluate_models.py` (M in ec37d19, 6356916, 680438f)
- `src/punct_restore/evaluation/`: `__init__.py` (M in 6356916, 680438f), `baseline.py` (A in 680438f), `comparison.py` (A in 6356916; M in ec37d19), `error_analysis.py` (M in 6356916), `inputs.py` (A in 680438f; M in ec37d19, 6356916), `manifest.py` (A in 6356916; M in 9155a7e), `metrics.py` (M in 6356916, 680438f), `report.py` (A in 680438f; M in 6340b59, ec37d19, 6356916), `uncertainty.py` (A in 6356916), `views.py` (A in 680438f; M in 6356916)
- `tests/`: `README.md` (M in 6356916, 680438f), `cli_analysis_cases.py` (A in 6356916), `conftest.py` (A in 680438f; M in 6356916), `eval_fixtures.py` (A in 680438f; M in 6356916), `test_evaluate_models_cli.py` (A in 680438f), `test_evaluate_models_cli_analysis.py` (A in 6356916), `test_evaluate_models_cli_safety.py` (A in 680438f), `test_evaluate_models_cli_score_extras.py` (A in 6356916), `test_evaluation_baseline.py` (A in 680438f; M in 6356916), `test_evaluation_comparison.py` (A in 6356916; M in ec37d19), `test_evaluation_errors.py` (A in 6356916), `test_evaluation_input_hardening.py` (A in 680438f), `test_evaluation_inputs.py` (A in 680438f; M in ec37d19), `test_evaluation_manifest.py` (A in 6356916; M in 9155a7e), `test_evaluation_metrics.py` (A in 680438f; M in 6356916), `test_evaluation_report.py` (A in 680438f; M in 6340b59, ec37d19, 6356916), `test_evaluation_uncertainty.py` (A in 6356916), `test_evaluation_views.py` (A in 680438f; M in 6356916)

Source: `git log HEAD --author="Raouf"` (HEAD is `ddba805`, the `release/final` merge) on 2026-10-30. The member fills in the student ID.
