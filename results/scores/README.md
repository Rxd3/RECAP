# Evaluator outputs (Member 5)

Every number in [EVALUATION_RESULTS.md](../../docs/EVALUATION_RESULTS.md) and [EVALUATION_RESULTS_CRF.md](../../docs/EVALUATION_RESULTS_CRF.md) comes from one of these JSON files.

All files were written by `scripts/evaluate_models.py` (`score`, `compare`, `errors`), except two folders:
- `threshold/tau_<mode>.json` comes from `scripts/tune_threshold.py tune`;
- `runtime/` comes from the timing script.

**Format:**
- Scores are fractions with six decimals; the docs show them as percentages rounded half-up.
- The score, compare and errors files record the sha256 of their gold, ID-list, mask and prediction inputs. `threshold/tau_<mode>.json` records only `model_sha256`, and `runtime/*.json` only the model file name. The predictions themselves are not committed: they can be regenerated from the shared models (see [REPRODUCIBILITY.md](../../docs/REPRODUCIBILITY.md)).
- **Data excerpts:** the `*.errors.json` files contain short SponSpeech excerpts as error examples (gold and predicted text around each error). They are CC BY-NC 4.0, with the upstream attribution in [DATA_LICENSE.md](../../docs/DATA_LICENSE.md).

| Folder | Contents |
|---|---|
| `hmm/` | HMM and shortcut baseline scores on validation, test and test_amb; HMM error pairs |
| `crf_validation/` | Member 3's six CRFs on validation (model selection); paired comparisons against the basic CRF and the HMM; error pairs |
| `crf_test/` | The same six CRFs on test and test_amb, each predicted once; paired comparisons; error pairs |
| `threshold/` | τ grids tuned on validation; threshold-decoded test and test_amb scores; paired comparisons against the plain improved CRF |
| `ablations/mode_<M>/` | One-feature-group-removed ablations on validation (50k-sequence training sample) |
| `filler/mode_<M>/` | Improved CRF with and without the filler-word cue on validation (same sample) |
| `h1_joint/` | Mode A model vs the mode C model projected to punctuation, on mode A validation gold over the mode C IDs: paired comparisons (`*_jointC_vs_A.json`) and per-model scores (`*_A_on_C_ids.json`, `*_C_as_A_on_C_ids.json`) |
| `b_to_a/` | Mode B models projected to mode A (D5 mapping) vs their mode A twins, mode A validation gold, with per-class intervals |
| `per_class/` | Paired per-class F1 intervals (`compare --class`) behind hypotheses H2–H4 |
| `sensitivity/` | Mode A CRF basic and improved retrained on the 50k sample at c1 = c2 ∈ {0.01, 0.1, 1.0} and 100 or 300 iterations, scored on validation |
| `leakage/` | HMM and improved CRF on the official mode A test, split by leakage-control status and exclusion reason; native SponSpeech labels |
| `runtime/` | HMM and CRF prediction timing (median of 3 passes, same laptop) |
| `crf_training_manifests/` | Member 3's six training manifests: settings, data hashes, fit time, peak memory, environment |
