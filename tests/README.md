# Contract test plans

These three Python files contain only descriptions and TODOs. No functional
tests have been implemented or passed; pytest currently has no tests to collect.

- Member 1: records, labels, IDs, policy edge cases and fixed split integrity.
- Member 6 with Members 2–3: adapters, gold-free inputs, alignment and persistence.
- Member 5: hand-checkable metrics, O handling, zero division and projections.

Add tiny meaningful fixtures as behavior is implemented. Do not download full
datasets, weights or live API resources in unit tests. Initial structure/config
checks are documented separately in `docs/INITIAL_VERIFICATION.md`.
