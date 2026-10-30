# Reproducibility & Environment Guide

Owner: Member 6, with all producing members. Member 2's HMM training/export
command is implemented and documented in [HMM notes](../experiments/HMM_NOTES.md).
Its full SponSpeech v2 study passed score recounts, independent full-data refits
and exact prediction reproduction; see [HMM results](../experiments/HMM_RESULTS.md)
and the [HMM runbook](../experiments/HMM_RUNBOOK.md). Other task commands remain
unfinished on this branch; the integrated workflow below is still a plan.

## Environment Specification
- **Python**: 3.11+
- **Setup**: `pip install -e ".[dev]"`
- **Hardware Target**: CPU

## Reproduction Steps

Member 6 will coordinate remaining flags and integration commands as adapters
arrive. Data preparation and HMM train/predict/search/verification/packaging
commands are implemented. The HMM study uses the pinned Member 5 evaluator in
a separate checkout. Do not infer other working command-line arguments from filenames.

HMM reproduction uses CPython 3.13.5 and the pinned PyYAML in
`requirements-hmm.txt`; all three reproduction levels below passed in an isolated
environment. Add only needed packages in reviewed implementation PRs and capture
a tested environment for each model. `requirements.txt` currently
installs the development extra; it is not a lock or a claim of a tested model
environment. Avoid adding GPU packages for the planned classical models.

## Reproduction levels

1. Recalculate metrics from an existing verified gold/prediction bundle.
2. Rerun inference using matching artifacts, feature code and input processing.
3. Retrain using matching data, configuration, seed and recorded environment.

Document which levels actually work; a saved estimator does not prove original
training provenance. Never update a checksum merely to conceal incompatible data.

## Member 6 integration acceptance

- [ ] Load one shared input sequence through each selected model adapter.
- [ ] Check mode, label order, feature/preprocessing versions and artifact metadata.
- [ ] Strip all gold/privileged metadata from prediction inputs.
- [ ] Reject incompatible or unfitted models with an actionable error.
- [ ] Export common prediction records and pass Member 5's alignment checks.
- [ ] Demonstrate approved punctuation/casing decoding; explain A/B casing limits.
- [ ] Reproduce one tiny end-to-end run in a clean environment using documented commands.
- [ ] Preserve measured evidence for the presentation and list remaining limits.
