# Tests

Run `python -m pytest -q` from the repository root. Every test uses tiny local fixtures: no
downloads, no real data, no trained weights.

| Group | Files | What they check |
| --- | --- | --- |
| Data (Member 1) | `test_data_contract.py`, `test_data_release_regressions.py` | Records, labels, IDs, annotation policies, masks, split integrity, checksums and release audits. |
| HMM (Member 2) | `test_hmm.py`, `test_hmm_cli.py`, `test_hmm_workflow.py`, `test_hmm_experiments.py` | Hand-computed probabilities, Viterbi, unknown words, gold-field invariance, persistence, CLI manifests, frozen selections. |
| CRF (Member 3) | `test_crf_features.py`, `test_crf_model.py`, `test_train_crf_cli.py`, `test_train_crf_standalone.py` | Feature extraction, CRF fit/predict/save/load, the training CLI and running it from a plain checkout. |
| Evaluation (Member 5) | `test_evaluation_*.py`, `test_evaluate_models_cli*.py`, with `conftest.py`, `eval_fixtures.py`, `cli_analysis_cases.py` | Metrics, views, masks, bootstrap, paired comparison, errors, manifest check, reports and the evaluator CLI. |
| Final-push extensions | `test_threshold.py`, `test_run_ablations.py`, `test_project_joint.py`, `test_build_leakage_subsets.py`, `test_compare_classes.py`, `test_reporting.py`, `test_reporting_safety.py` (with `reporting_fixtures.py`) | Threshold decoding, feature ablations, the H1 projection, leakage subsets and the table builders. |
| Integration (Member 6) | `test_model_contract.py`, `test_pipeline.py`, `test_prediction_outputs.py` | Loading saved models, the model contract, text reconstruction, the demo CLI and safe prediction outputs. |
| Repository | `test_dependencies.py`, `test_repo_hygiene.py` | Exact dependency pins and no scaffold leftovers. |
