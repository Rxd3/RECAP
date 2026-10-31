# Evaluator outputs (Member 5)

Every number in [EVALUATION_RESULTS.md](../../docs/EVALUATION_RESULTS.md) and [EVALUATION_RESULTS_CRF.md](../../docs/EVALUATION_RESULTS_CRF.md) comes from one of these JSON files.

All files were written by `scripts/evaluate_models.py` (`score`, `compare`, `errors`), except these:
- `threshold/tau_<mode>.json` and `threshold_prior/tau_gamma_<mode>.json` come from `scripts/tune_threshold.py tune`;
- `runtime/` comes from the timing script;
- `crf_training_manifests/*.manifest.json` come from `scripts/train_crf.py train` (Member 3);
- `asr_noise/wer<rate>/summary.json` and `asr_noise_v2/<edit_model>/wer<rate>/seed<seed>/summary.json` come from `scripts/simulate_asr_noise.py`, and `asr_noise_v2/summary.json` comes from `scripts/reporting/asr_noise_table.py`;
- `clusters/clusters_summary.json` was assembled from the six `clusters/compare_*.json` files during the word-cluster run (the K choice and its caveat; every number in it repeats a compare file);
- `gold_noise/label_audit_sheet.csv` is the team's filled audit sheet (100 primary-validation records; rows 51–100 checked by hand, rows 1–50 agent counts only), and `gold_noise/label_audit.json` holds the precision, recall and casing accuracy computed from it with 95% Wilson intervals;
- `gold_noise/*_B_noisy.tsv` (sample ID, exclusion rule) and `gold_noise/*_B_summary.json` (counts, including test_amb and `default_train`) come from `scripts/find_gold_noise.py`;
- `crf_convergence/convergence.json` comes from `scripts/reporting/crf_convergence.py`;
- `hmm/length_position_A_test.json` comes from `scripts/reporting/length_position.py`.

**Format:**
- Scores are fractions with six decimals; the docs show them as percentages rounded half-up.
- The score, compare and errors files record the sha256 of their gold, ID-list, mask and prediction inputs. `threshold/tau_<mode>.json` and `threshold_prior/tau_gamma_<mode>.json` record the model sha256, the split and the sha256 of every input (gold, ID list, masks, and the training data for the priors); they were regenerated with this provenance on 2026-10-30 and gave the same τ, γ and grid scores as before. `runtime/*.json` record only the model file name. The predictions themselves are not committed: they can be regenerated from the shared models (see [REPRODUCIBILITY.md](../../docs/REPRODUCIBILITY.md)).
- **Intentional copies:** `filler/mode_A/improved.json`, `filler/mode_B/improved.json`, `filler/mode_C/improved.json` and `factored/improved_C.json` are byte-identical to `ablations/mode_<M>/improved.json`, the shared 50k-sample baseline. They are kept so that each study folder stands alone.
- **Data excerpts:** the `*.errors.json` files contain short SponSpeech excerpts as error examples (gold and predicted text around each error), and `gold_noise/label_audit_sheet.csv` holds the upstream transcripts and our rendered labels of the 100 audited validation utterances. They are CC BY-NC 4.0, with the upstream attribution in [DATA_LICENSE.md](../../docs/DATA_LICENSE.md).

| Folder | Contents |
|---|---|
| `hmm/` | HMM and shortcut baseline scores on validation, test and test_amb; HMM error pairs; `length_position_A_test.json` (EVALUATION_RESULTS.md §6) |
| `crf_validation/` | Member 3's six CRFs on validation (model selection); paired comparisons against the basic CRF and the HMM; error pairs |
| `crf_test/` | The same six CRFs on test and test_amb, each predicted once; paired comparisons; error pairs |
| `threshold/` | τ grids tuned on validation; threshold-decoded test and test_amb scores; paired comparisons against the plain improved CRF |
| `threshold_prior/` | τ × γ prior-adjusted decoding: grids tuned on validation (`tau_gamma_<mode>.json`), test and test_amb scores, paired comparisons against the plain improved CRF (`tg_vs_improved_*`) and against the plain threshold (`tg_vs_tau_*`) |
| `gap/` | Gap classifier (the improved CRF's features without transitions) vs the improved CRF on validation, same 50k sample |
| `hidden_event_lm/` | Hidden-event LM: order × discount grid on validation, chosen setting on test and test_amb, paired comparisons against the HMM and the improved CRF |
| `question2/` | Improved CRF with vs without the question-lookback features on validation (50k sample), with QUESTION per-class intervals |
| `factored/` | Mode C joint improved CRF vs the factored pipeline (mode A punctuation + CASE casing) on validation (50k sample) |
| `neural/` | Member 2's fine-tuned BERT and FLAN-T5 (modes A and B): scores on validation, test and test_amb and paired comparisons against the improved CRF, τ × γ and the HMM, with per-class intervals |
| `clusters/` | Improved CRF with vs without the data-driven word-cluster features on validation (50k sample), K = 100 and 300 (`compare_<mode>_k<K>.json`), with the K choice and its caveat in `clusters_summary.json` |
| `ablations/mode_<M>/` | One-feature-group-removed ablations on validation (50k-sequence training sample) |
| `filler/mode_<M>/` | Improved CRF with and without the filler-word cue on validation (same sample) |
| `h1_joint/` | Mode A model vs the mode C model projected to punctuation, on mode A validation gold over the mode C IDs: paired comparisons (`*_jointC_vs_A.json`, with per-class COMMA/PERIOD/QUESTION blocks), the controlled check with the mode A HMM and CRF basic retrained on the mode C training IDs (`*_jointC_vs_A_sameids.json`), and per-model scores (`*_A_on_C_ids.json`, `*_C_as_A_on_C_ids.json`) |
| `b_to_a/` | Mode B models projected to mode A (D5 mapping) vs their mode A twins, mode A validation gold, with per-class intervals |
| `per_class/` | Paired per-class F1 intervals (`compare --class`) behind hypotheses H2–H4 |
| `sensitivity/` | Mode A CRF basic and improved retrained on the 50k sample at c1 = c2 ∈ {0.01, 0.1, 1.0} and 100 or 300 iterations, scored on validation |
| `gold_noise/` | Mode B rescored without the known noisy records (speaker labels, subtitle cue numbers, two auto-caption videos), with the excluded ID lists and their rules and the finder's counts per split |
| `crf_convergence/` | Loss traces of the six delivered CRFs: iterations, final loss, relative drop over the last 10 iterations |
| `asr_noise/wer<rate>/` | Frozen HMM and CRFs on test inputs with simulated ASR-style word errors (5/10/15% WER), with the corruption summary |
| `asr_noise_v2/` | Frozen HMM and CRFs on test inputs with `v2` simulated ASR word errors across seeds 2026–2028 at 5/10/15% WER, `v1` seed variance at 15% WER, and `summary.json` |
| `leakage/` | HMM and improved CRF on the official mode A test, split by leakage-control status and exclusion reason; native SponSpeech labels |
| `runtime/` | HMM and CRF prediction timing (median of 3 passes, same laptop) |
| `crf_training_manifests/` | Member 3's six training manifests: settings, data hashes, fit time, peak memory, environment |
