# Architecture and ownership

Every component is implemented: the data release (Member 1), the HMM workflow (Member 2), the CRF
and its features (Member 3), the evaluator (Member 5) and the demo pipeline. Results are in
[EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) (HMM and shortcut) and
[EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) (CRF, threshold decoding, leakage, runtime).

```text
verified source text
  -> data preprocessing and labels (Member 1)
  -> immutable JSONL release + shared split manifest
  -> HMM (Member 2) or CRF (Member 3)
                         ^ basic/improved feature extractor (Member 3)
  -> standard prediction JSONL
  -> common evaluation and error analysis (Member 5)
  -> integration/demo and documented evidence (Member 6)
```

One `punct_restore` package supports every mode. Do not create separate copies
of the pipeline for four-, eight-, and twelve-class experiments. Model and
feature configuration selects behavior; versioned contracts fix alignment.

| Area | Primary owner | Boundary |
| --- | --- | --- |
| `src/punct_restore/data/`, `scripts/prepare_data.py`, data documentation | Member 1 | Source text → records; no model training |
| `models/hmm.py`, `scripts/train_hmm.py` | Member 2 | Records → HMM artifacts/predictions |
| `models/crf.py`, `scripts/train_crf.py` | Member 3 | Records + injected features → CRF artifacts/predictions |
| `features/`, `scripts/run_ablations.py` | Member 4 (role); code by Member 3, ablation runner by Member 5 | Observable tokens → aligned feature dictionaries |
| `evaluation/`, `scripts/evaluate_models.py`, results | Member 5 | Gold + predictions → verified reports |
| Comparison methods and robustness checks: `models/gap_classifier.py`, `models/hidden_event_lm.py` (+ `ngram_lm.py`, `truecaser.py`), τ × γ in `models/threshold.py`, question lookback, `models/factored.py` and the CRF `CASE` mode, `data/asr_noise.py`, the gold-noise check, the demo loader; scripts `train_hidden_event_lm.py`, `factor_predictions.py`, `simulate_asr_noise.py` | Member 5 | Exploratory comparisons and checks next to the course baselines |
| `pipeline.py` (`load_model`, `ModelAdapter`, `RestorationPipeline`) and `scripts/run_demo.py` (the demo path), presentation | Member 6 | Text + saved HMM or hidden-event LM (`.json`), or CRF or gap classifier (`.joblib`, `.pkl`) → restored text |

The feature code in `crf_features.py`, both presets included, was written by Member 3; Member 4's
branch has no commits ([contributions](../contributions/Mohamed_STUDENT_ID_GITHUB.md)). The threshold
decoder and the filler cue were added by Member 5 during release integration. Shared documents and
configs need affected owners' review.

## Boundaries

- Data loading validates [DATA_CONTRACT.md](DATA_CONTRACT.md), records rejection
  counts and never silently changes sequence lengths.
- Both models follow [MODEL_CONTRACT.md](MODEL_CONTRACT.md). Training adapters
  derive the target vector from `label_mode`; observable model inputs contain
  tokens and identifiers only. Evaluation retains gold in a separate path.
- Feature functions accept tokens and declared settings, not whole gold records.
- The pipeline applies the same preprocessing/tokenization used during training,
  verifies artifact compatibility and delegates to model adapters.
- Evaluation joins by ID, checks order/length/release and applies a named,
  approved projection. It never repairs mismatches by truncating or zipping.
- Task scripts are thin entry points around package code, with repository-relative
  configuration and paths. Every script is implemented; see [scripts/README.md](../scripts/README.md).

Future neural adapters may implement the common contract; add complexity only
when an agreed experiment requires it. Separate dataset, artifact and prediction
versions preserve the ability to recalculate metrics without retraining.
