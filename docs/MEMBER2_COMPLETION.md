# Member 2 completion audit

Study `hmm-sponspeech-v2-20261029` uses the approved `sponspeech-m1-v2`
release and `m1-v2` policy. Member 2's implementation, training, selection,
evaluation and reproduction are complete. Team distribution and human consumer
acknowledgments are recorded separately below.

## Requirements and evidence

| Requirement | Completed evidence |
| --- | --- |
| Count estimation, START/END, Viterbi and OOV | `HMM_NOTES.md`; hand-calculated/exhaustive tests, balanced counts and normalized probabilities |
| All A/B/C modes and common records | 4/8/12 states; strict schema, label order, IDs, lengths and release checks |
| Training uses approved data only | A/B 142,722 sequences / 4,137,833 tokens; C 141,745 / 4,112,054; counts independently matched to release |
| Validation selection | 27 predeclared candidates; alpha 0.01 for all modes, cutoff 1 for A/B and 2 for C |
| Test isolation | Global selection frozen at 13:30:02 UTC, before test access at 13:30:41 UTC; no validation refit or test-driven retuning |
| Saved models and aligned predictions | Three selected JSON models; full validation/test/test_amb predictions with manifests; 178,257 sequence predictions reproduced exactly |
| Shared evaluation | Pinned Member 5 scorer at `3044a58e777200bcac64e6e3364922404be359bf`; all 27 candidate primary scores and all blocks of nine HMM plus nine shortcut reports independently recounted |
| Reproduction | All three independent full-data refits produce identical model bytes; inference ignores gold fields |
| Actual runtime and limitations | Common 616,940-token input, one thread, three repetitions; hardware, scope, confusion counts, position effects and rare-label weaknesses reported |
| Commands, dependencies and provenance | Frozen plan/selection; command logs and data/config/code/model/prediction hashes; CPython 3.13.5 with PyYAML 6.0.3 |
| Meaningful checks | 214 repository tests pass; pinned evaluator passes 428 tests; synthetic full workflow passed before the real study |
| Portable handoff | Packager copies the complete verified study, selected-model index, IDs/masks, license, configuration and reproduction evidence; SHA-256 round-trip of every ZIP member |

Training implementation: `e4c31834799bb261e78218c67dd85e148ad8d493`.
Subsequent documentation/packaging changes preserve all six frozen implementation
file hashes. The study's independent `verification/verification.json` reports PASS.

## Handoff location and use

Local output: `artifacts/hmm/hmm-sponspeech-v2-20261029.zip`, with its
`.zip.sha256` sidecar. Inside, `MODEL_INDEX.json` identifies selected artifacts;
`SHA256SUMS` covers every other file. Raw dataset records are distributed separately
in the Member 1 v2 bundle. See the [runbook](../experiments/HMM_RUNBOOK.md) for
the exact packaging and reproduction commands and [results](../experiments/HMM_RESULTS.md)
for measured scores.

- Member 3 compares the CRF to the HMM within the same mode and primary IDs.
- Member 5 receives full predictions, reports, masks, counts and provenance for
  subsequent model comparisons and error analysis.
- Member 6 loads the selected JSON with `HMMModel.load_model(path)` and passes
  shared gold-free records to `predict`. Text reconstruction/demo integration
  remains their task; an O-only placeholder is not a trained HMM adapter.

The team lead receives the local bundle for the preferred shared Drive handoff.
Drive upload, individual consumer acknowledgments and the final integrated demo
remain team actions. No human Member 5 sign-off or public model publication is
claimed. CRF implementation, feature ablations and project-wide final acceptance
are outside Member 2's completed baseline deliverables.
