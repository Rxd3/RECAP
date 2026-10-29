# Contract tests

Run `python -m pytest -q` from the repository root. Member 1's data contract and
release regression suites use tiny local fixtures and cover annotation policies,
alignment, masks, split integrity, checksums, extraction and immutable releases.
Member 5's evaluation tests are implemented: `test_evaluation_*.py` and
`test_evaluate_models_cli*.py`, with shared fixtures in `conftest.py` and
`eval_fixtures.py`. Model tests develop on their member branches;
`test_model_contract.py` and `test_evaluation_contract.py` still contain only
descriptions and TODOs.

- Member 1: records, labels, IDs, policy edge cases and fixed split integrity.
- Member 6 with Members 2–3: adapters, gold-free inputs, alignment and persistence.
- Member 5: hand-checkable metrics, O handling, zero division and projections.

Add tiny meaningful fixtures as behavior is implemented. Do not download full
datasets, weights or live API resources in unit tests. Initial structure/config
checks are documented separately in `docs/INITIAL_VERIFICATION.md`.
