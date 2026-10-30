# Evaluator outputs (Member 5)

Every number in [EVALUATION_RESULTS.md](../../docs/EVALUATION_RESULTS.md) and [EVALUATION_RESULTS_CRF.md](../../docs/EVALUATION_RESULTS_CRF.md) comes from one of these JSON files.

All files were written by `scripts/evaluate_models.py` (`score`, `compare`, `errors`), except these:
- `threshold/tau_<mode>.json` and `threshold_prior/tau_gamma_<mode>.json` come from `scripts/tune_threshold.py tune`;
- `runtime/` comes from the timing script;
- `crf_training_manifests/*.manifest.json` come from `scripts/train_crf.py train` (Member 3);
- `asr_noise/wer<rate>/summary.json` comes from `scripts/simulate_asr_noise.py`;
- `gold_noise/*_B_noisy.tsv` (sample ID, exclusion rule) and `gold_noise/*_B_summary.json` (counts, including test_amb and `default_train`) come from `scripts/find_gold_noise.py`;
- `crf_convergence/convergence.json` comes from `scripts/reporting/crf_convergence.py`;
- `hmm/length_position_A_test.json` comes from `scripts/reporting/length_position.py`.

**Format:**
- Scores are fractions with six decimals; the docs show them as percentages rounded half-up.
- The score, compare and errors files record the sha256 of their gold, ID-list, mask and prediction inputs. The committed `threshold/tau_<mode>.json` and `threshold_prior/tau_gamma_<mode>.json` record only `model_sha256` (they predate the change that makes `tune_threshold.py tune` also record the split and the input sha256 values), and `runtime/*.json` only the model file name. The predictions themselves are not committed: they can be regenerated from the shared models (see [REPRODUCIBILITY.md](../../docs/REPRODUCIBILITY.md)).
- **Intentional copies:** `filler/mode_A/improved.json`, `filler/mode_B/improved.json`, `filler/mode_C/improved.json` and `factored/improved_C.json` are byte-identical to `ablations/mode_<M>/improved.json`, the shared 50k-sample baseline. They are kept so that each study folder stands alone.
- **Data excerpts:** the `*.errors.json` files contain short SponSpeech excerpts as error examples (gold and predicted text around each error). They are CC BY-NC 4.0, with the upstream attribution in [DATA_LICENSE.md](../../docs/DATA_LICENSE.md).

| Folder | Contents |
|---|---|
| `hmm/` | HMM and shortcut baseline scores on validation, test and test_amb; HMM error pairs; `length_position_A_test.json` (EVALUATION_RESULTS.md §6) |
| `crf_validation/` | Member 3's six CRFs on validation (model selection); paired comparisons against the basic CRF and the HMM; error pairs |
| `crf_test/` | The same six CRFs on test and test_amb, each predicted once; paired comparisons; error pairs |
| `threshold/` | τ grids tuned on validation; threshold-decoded test and test_amb scores; paired comparisons against the plain improved CRF |
| `threshold_prior/` | τ × γ prior-adjusted decoding: grids tuned on validation (`tau_gamma_<mode>.json`), test and test_amb scores, paired comparisons against the plain improved CRF |
| `gap/` | Gap classifier (the improved CRF's features without transitions) vs the improved CRF on validation, same 50k sample |
| `hidden_event_lm/` | Hidden-event LM: order × discount grid on validation, chosen setting on test and test_amb, paired comparisons against the HMM and the improved CRF |
| `question2/` | Improved CRF with vs without the question-lookback features on validation (50k sample), with QUESTION per-class intervals |
| `factored/` | Mode C joint improved CRF vs the factored pipeline (mode A punctuation + CASE casing) on validation (50k sample) |
| `ablations/mode_<M>/` | One-feature-group-removed ablations on validation (50k-sequence training sample) |
| `filler/mode_<M>/` | Improved CRF with and without the filler-word cue on validation (same sample) |
| `h1_joint/` | Mode A model vs the mode C model projected to punctuation, on mode A validation gold over the mode C IDs: paired comparisons (`*_jointC_vs_A.json`) and per-model scores (`*_A_on_C_ids.json`, `*_C_as_A_on_C_ids.json`) |
| `b_to_a/` | Mode B models projected to mode A (D5 mapping) vs their mode A twins, mode A validation gold, with per-class intervals |
| `per_class/` | Paired per-class F1 intervals (`compare --class`) behind hypotheses H2–H4 |
| `sensitivity/` | Mode A CRF basic and improved retrained on the 50k sample at c1 = c2 ∈ {0.01, 0.1, 1.0} and 100 or 300 iterations, scored on validation |
| `gold_noise/` | Mode B rescored without the known noisy records (speaker labels, subtitle cue numbers, two auto-caption videos), with the excluded ID lists and their rules and the finder's counts per split |
| `crf_convergence/` | Loss traces of the six delivered CRFs: iterations, final loss, relative drop over the last 10 iterations |
| `asr_noise/wer<rate>/` | Frozen HMM and CRFs on test inputs with simulated ASR-style word errors (5/10/15% WER), with the corruption summary |
| `leakage/` | HMM and improved CRF on the official mode A test, split by leakage-control status and exclusion reason; native SponSpeech labels |
| `runtime/` | HMM and CRF prediction timing (median of 3 passes, same laptop) |
| `crf_training_manifests/` | Member 3's six training manifests: settings, data hashes, fit time, peak memory, environment |
