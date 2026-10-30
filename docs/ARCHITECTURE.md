# Architecture and ownership

Member 1's release pipeline and Member 2's full HMM workflow are implemented.
HMM training, selection, evaluation and independent reproduction are complete;
see [results](../experiments/HMM_RESULTS.md). The study calls Member 5's pinned
scorer from a separate checkout. Other components remain designs/placeholders
on this branch.

```text
verified source text
  -> data preprocessing and labels (Member 1)
  -> immutable JSONL release + shared split manifest
  -> HMM (Member 2) or CRF (Member 3)
                         ^ basic/improved feature extractor (Members 3–4)
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
| `features/`, `scripts/run_ablations.py` | Member 4 | Observable tokens → aligned feature dictionaries |
| `evaluation/`, `scripts/evaluate_models.py`, results | Member 5 | Gold + predictions → verified reports |
| `pipeline.py` (`load_model`, `ModelAdapter`, `RestorationPipeline`) and `scripts/run_demo.py` (the demo path), presentation | Member 6 | Text + saved HMM (`.json`) or CRF (`.joblib`) artifact → restored text |

Member 4 owns `crf_features.py`; Member 3 reviews its interface and contributes
the basic preset through a coordinated PR. Member 4 does not replace or fork
Member 3's model. Shared documents/configs need affected owners' review.

## Planned boundaries

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
- Task scripts will be thin entry points around package code. Configuration and
  paths are repository-relative. Preparation and HMM train/predict/experiment/
  verification/handoff scripts are implemented; CRF, ablations, evaluation on
  this branch and demo scripts remain unfinished.

Future neural adapters may implement the common contract; add complexity only
when an agreed experiment requires it. Separate dataset, artifact and prediction
versions preserve the ability to recalculate metrics without retraining.
