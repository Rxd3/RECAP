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

## Code layout

All code is in the `punct_restore` package. The one command line is `recap <command> [args]`, a console
script that `[project.scripts]` in `pyproject.toml` points at `punct_restore.cli:main`.

```text
src/punct_restore/
  cli/              __init__.py (the dispatcher and COMMANDS) and one module per
                    command (18), plus train_crf_inspect.py and train_crf_predict.py, the inspect and
                    predict subcommands of train-crf
  reporting/        7 reporting commands, plus _cli.py, build_tables.py and
                    asr_noise_{stats,report,findings}.py (the parts of the ASR noise report)
  data/             extraction, preprocessing, labeling, splits, validation, asr_noise, gold_noise;
                    release.py with release_{constants,stats,notes,checks,manifest,outputs}.py
  features/         crf_features.py with crf_lexicons.py and crf_cues.py; word_clusters.py
  models/           hmm.py with hmm_validation.py; hmm_workflow.py and
                    hmm_experiment_{protocol,commands,runtime,audit}.py (the HMM study);
                    crf.py with crf_{modes,records,io,training}.py; threshold.py; gap_classifier.py;
                    hidden_event_lm.py with hidden_event_lm_viterbi.py, ngram_lm.py, truecaser.py; factored.py
  evaluation/       inputs, views, metrics, uncertainty, comparison, error_analysis, baseline, manifest, report
  paths.py          REPO_ROOT, the checkout root, for the commands' default paths
  pipeline.py       demo inference (load_model, RestorationPipeline)
  prediction_io.py  gold-free prediction files and manifests
scripts/            25 three-line stubs (scripts/<name>.py, scripts/reporting/<name>.py), one per command
```

- **Dispatcher.** `punct_restore.cli.COMMANDS` maps each command name (the module name with `_` written as
  `-`) to its module. `recap`, `recap -h` and `recap --help` list the commands with the first line of each
  module's docstring and exit 0. An unknown name prints one line on stderr and exits 2. Otherwise the
  dispatcher returns the module's `main(argv)`, so argparse help and usage errors pass through.
- **Command modules** have a module docstring (the help summary) and `main(argv: list[str] | None) -> int`.
  `python -m punct_restore.cli.<module>` (or `punct_restore.reporting.<module>`) runs one directly.
- **Stubs.** Each file in `scripts/` imports `main` from its command module and exits with its return
  value, so the old `scripts/<name>.py` command lines keep working. Recorded commands in run manifests
  keep those paths.
- **Split modules.** The size split moved helpers out of long modules (for example `models/crf.py` into
  `crf_io`, `crf_modes`, `crf_records` and `crf_training`). The original module re-exports them, so old
  import names still work. `tests/test_code_limits.py` keeps every function in `src/` and `scripts/` at
  50 lines or less, nesting at 4 levels or less and every file at 400 lines or less.

## Ownership

| Area | Primary owner | Boundary |
| --- | --- | --- |
| `data/` (with the `release_*` modules), `cli/prepare_data.py` (`recap prepare-data`), data documentation | Member 1 | Source text → records; no model training |
| `models/hmm.py`, `models/hmm_validation.py`, `models/hmm_workflow.py`, `models/hmm_experiment_*.py`; `cli/train_hmm.py`, `cli/predict_hmm.py`, `cli/run_hmm_experiments.py`, `cli/verify_hmm_experiments.py`, `cli/package_hmm_handoff.py` | Member 2 | Records → HMM artifacts/predictions |
| `models/crf.py` with `crf_modes.py`, `crf_records.py`, `crf_io.py`, `crf_training.py`; `cli/train_crf.py` with `train_crf_inspect.py` and `train_crf_predict.py` (`recap train-crf`) | Member 3 | Records + injected features → CRF artifacts/predictions |
| `features/` (`crf_features.py`, `crf_lexicons.py`, `crf_cues.py`, `word_clusters.py`), `cli/run_ablations.py`, `cli/build_word_clusters.py` | Member 4 (role); code by Member 3, ablation runner by Member 5 | Observable tokens → aligned feature dictionaries |
| `evaluation/`, `cli/evaluate_models.py` (`recap evaluate-models`), `reporting/`, results | Member 5 | Gold + predictions → verified reports |
| Comparison methods and robustness checks: `models/gap_classifier.py`, `models/hidden_event_lm.py` (+ `hidden_event_lm_viterbi.py`, `ngram_lm.py`, `truecaser.py`), τ × γ in `models/threshold.py` (`cli/tune_threshold.py`), question lookback, `models/factored.py` and the CRF `CASE` mode, `data/asr_noise.py`, `data/gold_noise.py` (the gold-noise check), the demo loader; commands `recap train-hidden-event-lm`, `recap factor-predictions`, `recap simulate-asr-noise`, `recap find-gold-noise`, `recap project-joint`, `recap build-leakage-subsets`, `recap crf-convergence`, `recap length-position`, `recap asr-noise-table` (+ `reporting/asr_noise_stats.py`, `asr_noise_report.py`, `asr_noise_findings.py`) | Member 5 | Exploratory comparisons and checks next to the course baselines |
| `pipeline.py` (`load_model`, `ModelAdapter`, `RestorationPipeline`) and `cli/run_demo.py` (`recap run-demo`, the demo path), presentation | Member 6 | Text + saved HMM or hidden-event LM (`.json`), or CRF or gap classifier (`.joblib`, `.pkl`) → restored text |
| `cli/__init__.py` (dispatcher), `paths.py`, the `scripts/` stubs | Team (restructure, 2026-11-01) | Command routing only; no model or data logic |

The feature code in `crf_features.py`, both presets included, was written by Member 3; Member 4's
branch has no commits ([contributions](../contributions/Mohamed_2309015740_Mokamaleldin.md)). The threshold
decoder and the filler cue were added by Member 5 during release integration. Shared documents and
configs need affected owners' review.

## Boundaries

- Data loading validates [DATA_CONTRACT.md](DATA_CONTRACT.md), records rejection
  counts and never silently changes sequence lengths. Consumers read a release through the library
  API `punct_restore.data.load_records(path)` (raises `ValueError` on any contract violation, never
  repairs) and `punct_restore.data.read_ids(path)` (the sample IDs of an `.ids` list); see
  [data/README.md](../data/README.md).
- Both models follow [MODEL_CONTRACT.md](MODEL_CONTRACT.md). Training adapters
  derive the target vector from `label_mode`; observable model inputs contain
  tokens and identifiers only. Evaluation retains gold in a separate path.
- Feature functions accept tokens and declared settings, not whole gold records.
- The demo pipeline does not use the training tokenizer. `clean_tokens` in `pipeline.py` splits the raw
  text on whitespace (`str.split()`), lowercases each piece and strips `,.?!:;-"` from its edges. The
  release tokens come from `data.preprocessing.tokenize` (lowercased by `model_token`). It also keeps an
  abbreviation's final period in the word, and it keeps brackets, quotes and dashes in the gaps between
  words. Inputs such as `(hello)` or `mr.` can therefore give tokens that the release never contains. The
  pipeline checks artifact compatibility and delegates to model adapters.
- Evaluation joins by ID, checks order/length/release and applies a named,
  approved projection. It never repairs mismatches by truncating or zipping.
- Command modules are thin entry points around library code, with repository-relative
  configuration and paths (`paths.REPO_ROOT`). `scripts/` holds only the stubs; see
  [scripts/README.md](../scripts/README.md).

Member 2's fine-tuned BERT and FLAN-T5 taggers are an external reference, not part of this package. Their
code and predictions are held by Member 2. They were scored here with the official evaluator, and the
SHA-256s of their predictions are recorded in `results/scores/neural/`
([EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §15). Future neural adapters may implement the
common contract; add complexity only when an agreed experiment requires it. Separate dataset, artifact and prediction
versions preserve the ability to recalculate metrics without retraining.
