# Contract tests

Run `python -m pytest -q` from the repository root. Member 1's data contract and
release regression suites use tiny local fixtures and cover annotation policies,
alignment, masks, split integrity, checksums, extraction and immutable releases.
`test_hmm.py` and `test_hmm_cli.py` cover hand-calculated probabilities,
exhaustive Viterbi/END checks, OOV handling, all modes, gold-field invariance,
input failures, persistence and CLI manifests. These fixture fits are compatibility
tests, not research experiments. Remaining model/evaluation contract plans develop
on their member branches.

`test_hmm_workflow.py` checks strict selections and release checksum failures.
`test_hmm_experiments.py` checks validation-only freeze gates, tie-breaking,
saved-model inference and the independent metric counter.

- Member 1: records, labels, IDs, policy edge cases and fixed split integrity.
- Member 6 with Members 2–3: adapters, gold-free inputs, alignment and persistence.
- Member 5: hand-checkable metrics, O handling, zero division and projections.

Add tiny meaningful fixtures as behavior is implemented. Do not download full
datasets, weights or live API resources in unit tests. Initial structure/config
checks are documented separately in `docs/INITIAL_VERIFICATION.md`.
