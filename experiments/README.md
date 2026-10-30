# Experiment records

Notes and runbooks for every model. Run outputs (models, predictions, score JSON) are written
under the ignored `experiments/runs/<run_id>/` and are not committed. They are distributed through the
team's shared Drive folder, the HMM handoff ZIP and the Member 5 results bundle, with SHA-256 sums.

| File | Content |
| --- | --- |
| [HMM_NOTES.md](HMM_NOTES.md) | HMM states per mode, estimation, Viterbi decoding, unknown words. |
| [HMM_RUNBOOK.md](HMM_RUNBOOK.md) | Exact commands for the HMM search, final evaluation, verification and packaging. |
| [HMM_RESULTS.md](HMM_RESULTS.md) | The completed HMM study: 27 validation candidates, selected settings, scores and independent reproduction. |
| [CRF_NOTES.md](CRF_NOTES.md) | CRF model, CLI, artifact and test notes (Member 3). |
| [FEATURE_NOTES.md](FEATURE_NOTES.md) | Basic and improved CRF features, and the feature ablations as run on validation. |

Results:

- HMM and shortcut, validation and test: [EVALUATION_RESULTS.md](../docs/EVALUATION_RESULTS.md).
- CRF basic and improved, threshold decoding, leakage, test_amb and runtime:
  [EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md). Test results are final; the
  validation selection is in §5 and the feature ablations (validation) in §6.
- The six CRF runs, their settings and hashes: [CRF_EXPERIMENTS.md](../docs/CRF_EXPERIMENTS.md).

To rerun anything, use [REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md). Select settings on
validation only, freeze them, and predict test once. Keep negative findings. In an ablation, change
only the declared factor.
