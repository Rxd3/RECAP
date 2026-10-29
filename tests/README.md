# Contract test plans

Member 5's evaluation tests are implemented: `test_evaluation_*.py` and
`test_evaluate_models_cli*.py`, with shared fixtures in `conftest.py`,
`eval_fixtures.py` and `cli_analysis_cases.py`. The other contract files still
contain only descriptions and TODOs.

- Member 1: records, labels, IDs, policy edge cases and fixed split integrity.
- Member 6 with Members 2–3: adapters, gold-free inputs, alignment and persistence.
- Member 5: hand-checkable metrics, O handling, zero division and projections.

Add tiny meaningful fixtures as behavior is implemented. Do not download full
datasets, weights or live API resources in unit tests. Initial structure/config
checks are documented separately in `docs/INITIAL_VERIFICATION.md`.
