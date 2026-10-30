# Evaluation outputs

Owner: Member 5. `predictions/`, `tables/` and `figures/` hold only directory markers in Git.
Evaluator outputs (prediction JSONL, score and compare JSON, Markdown summaries) are written here
locally and are ignored. They are distributed through the team's shared Drive folder and the
Member 5 results bundle, with SHA-256 sums. They are not committed.

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
