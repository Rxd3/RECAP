# Evaluator outputs (Member 5)

Every number in [EVALUATION_RESULTS.md](../../docs/EVALUATION_RESULTS.md) and [EVALUATION_RESULTS_CRF.md](../../docs/EVALUATION_RESULTS_CRF.md) comes from one of these JSON files.

All files were written by `scripts/evaluate_models.py` (`score`, `compare`, `errors`), except two folders:
- `threshold/tau_<mode>.json` comes from `scripts/tune_threshold.py tune`;
- `runtime/` comes from the timing script.

**Format:**
- Scores are fractions with six decimals; the docs show them as percentages rounded half-up.
- Each file records the sha256 of its gold, ID-list, mask and prediction inputs. The predictions themselves are not committed: they can be regenerated from the shared models (see [REPRODUCIBILITY.md](../../docs/REPRODUCIBILITY.md)).

| Folder | Contents |
|---|---|
| `hmm/` | HMM and shortcut baseline scores on validation, test and test_amb; HMM error pairs |
| `crf_validation/` | Member 3's six CRFs on validation (model selection); paired comparisons against the basic CRF and the HMM; error pairs |
| `crf_test/` | The same six CRFs on test and test_amb, each predicted once; paired comparisons; error pairs |
| `threshold/` | τ grids tuned on validation; threshold-decoded test and test_amb scores; paired comparisons against the plain improved CRF |
| `ablations/mode_<M>/` | One-feature-group-removed ablations on validation (50k-sequence training sample) |
| `filler/mode_<M>/` | Improved CRF with and without the filler-word cue on validation (same sample) |
| `h1_joint/` | Mode A model vs the mode C model projected to punctuation, on mode A validation gold over the mode C IDs: paired comparisons (`*_jointC_vs_A.json`) and per-model scores (`*_A_on_C_ids.json`, `*_C_as_A_on_C_ids.json`) |
| `leakage/` | HMM and improved CRF on the official mode A test, split by leakage-control status and exclusion reason; native SponSpeech labels |
| `runtime/` | HMM and CRF prediction timing (median of 3 passes, same laptop) |
| `crf_training_manifests/` | Member 3's six training manifests: settings, data hashes, fit time, peak memory, environment |
