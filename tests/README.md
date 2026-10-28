# Tests and contract plans

`test_hmm.py` and `test_hmm_cli.py` implement Member 2's synthetic unit/integration
tests. They cover hand-calculated estimation, exhaustive Viterbi checks, END,
OOV behavior, A/B/C alignment, gold-invariance, input failures, persistence and
CLI artifact/prediction manifests. Temporary test training is not a research
experiment and produces no committed datasets or model results.

```powershell
python -m pytest tests/test_hmm.py tests/test_hmm_cli.py -q
```

The three `test_*_contract.py` files remain descriptions/TODO plans for shared
integration as other components become available:

- Member 1: records, labels, IDs, policy edge cases and fixed split integrity.
- Member 6 with Members 2–3: adapters, gold-free inputs, alignment and persistence.
- Member 5: hand-checkable metrics, O handling, zero division and projections.

Add tiny meaningful fixtures as behavior is implemented. Do not download full
datasets, weights or live API resources in unit tests. Initial structure/config
checks are documented separately in `docs/INITIAL_VERIFICATION.md`.
