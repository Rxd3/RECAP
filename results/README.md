# Evaluation outputs

Owner: Member 5. [`scores/`](scores/README.md) holds the committed evaluator JSON behind every
number in the evaluation documents: HMM, CRF validation and test, threshold decoding, ablations,
filler cue, H1, leakage, runtime, per-class intervals, the B→A study and the settings sensitivity check. The score, compare and errors files record the SHA-256 of their
inputs (the τ files record the model hash, the runtime files the model file name), so any table can be
recomputed or checked from them. The `*.errors.json` files contain short SponSpeech excerpts (CC BY-NC 4.0,
attribution in [DATA_LICENSE.md](../docs/DATA_LICENSE.md)). `scores/crf_training_manifests/` holds Member 3's six CRF training
manifests (settings, data hashes, fit time, peak memory, environment). The scripts in
`scripts/reporting/` rebuild the result tables from these files ([REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md)).

Local outputs go to `predictions/` and `tables/`, which are ignored and not in Git; create them with
`mkdir -p results/predictions results/tables` first, because the evaluator never creates folders. Local prediction JSONL,
new score files and Markdown summaries written there are ignored; predictions are regenerated from the
shared models ([REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md)).

The reviewed results are tracked as documents:

- [EVALUATION_RESULTS.md](../docs/EVALUATION_RESULTS.md): HMM and shortcut baseline, validation, test
  and test_amb, with confusion matrices, error analysis and runtime;
- [EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md): CRF basic, improved and
  improved + threshold, with test results final (§1–4), validation selection (§5), feature
  ablations (§6), leakage (§8), test_amb (§9) and runtime (§10);
- [HMM_RESULTS.md](../experiments/HMM_RESULTS.md): the HMM study's own record.

Score a prediction file with `python scripts/evaluate_models.py score ... --out results/tables/<run_id>.json`,
which also writes `<run_id>.md`. The full commands and flags are in
[the evaluation protocol](../docs/EVALUATION_PROTOCOL.md#scored-sets-masks-views-and-the-shortcut-baseline)
and [REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md). Existing outputs are never overwritten.
