# Reproducibility and integration runbook — planned

Owner: Member 6, with all producing members. No training/inference/evaluation
commands have been implemented. Environment setup in the README is available;
the six task scripts explicitly stop as unfinished.

## Future workflow and evidence

| Stage | Future entry point | Required handoff |
| --- | --- | --- |
| Prepare | `scripts/prepare_data.py` | Verified sources, policy version, data validation, immutable splits/checksums |
| Train HMM | `scripts/train_hmm.py` | Training config, seed, selected validation settings, model artifact |
| Train CRF | `scripts/train_crf.py` | Same data/mode, basic or improved injected feature version, artifact |
| Ablate | `scripts/run_ablations.py` | Controlled feature changes and full resolved configurations |
| Evaluate | `scripts/evaluate_models.py` | Gold release + verified predictions, frozen metric/projection settings |
| Demonstrate | `scripts/run_demo.py` | Compatible artifact, shared observable input and versioned text decoder |

Member 6 will define flags and exact example commands in a reviewed PR after
adapters exist. Do not infer working command-line arguments from filenames.

For every real run, capture code commit/dirty status, full command, resolved
configuration, source revisions, preprocessing and tokenizer versions, split
manifest/checksums, ordered labels, feature version, random seeds, Python/package
versions, OS/hardware/threads and measured duration. Use repository-relative
paths and distinct run IDs. Share artifact and prediction hashes with consumers.

Runtime dependencies are intentionally absent at scaffold time. Add only needed
packages in reviewed implementation PRs, verify Python compatibility, then
capture a tested environment lock for experiments. `requirements.txt` currently
installs the development extra; it is not a lock or a claim of a tested model
environment. Avoid adding GPU packages for the planned classical models.

## Three future reproduction levels

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
