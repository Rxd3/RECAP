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
  cli/              __init__.py (the dispatcher and COMMANDS) and one module per command (18);
                    each parses arguments, reads and writes files and runs its command's
                    own steps, behind the error boundary
  reporting/        7 reporting commands, plus _cli.py, build_tables.py and
                    asr_noise_{stats,report,findings}.py (the parts of the ASR noise report)
  data/             extraction, preprocessing, labeling, splits, validation, asr_noise, gold_noise;
                    release.py with release_{constants,stats,notes,checks,manifest,outputs}.py
                    (release_outputs holds every writer; release_checks returns a ReleaseCheck)
  features/         crf_features.py with crf_lexicons.py and crf_cues.py; word_clusters.py
  models/           hmm.py with hmm_validation.py and hmm_records.py; hmm_workflow.py; the HMM study in
                    hmm_experiment_{identity,protocol,commands,tune,frozen,finalize,verify,runtime,audit}.py;
                    crf.py with crf_{modes,records,io,training,training_data,prediction,inspection}.py;
                    threshold.py; gap_classifier.py;
                    hidden_event_lm.py with hidden_event_lm_viterbi.py, ngram_lm.py, truecaser.py; factored.py
  evaluation/       inputs, views, metrics, uncertainty, comparison, error_analysis, baseline, manifest, report
  hashing.py        sha256_file and sha256_stream, the one streaming SHA-256
  paths.py          REPO_ROOT, the checkout root, for the commands' default paths
  pipeline.py       demo inference (load_model, RestorationPipeline)
  prediction_io.py  shared command I/O: gold-free prediction files and manifests, read_ids,
                    parse_json_object and parse_sample_record, write_new_file, refuse_existing, and the
                    error boundary (one_line_error, OS_REASONS, reconfigure_stdout)
  progress.py       Log and discard_log: library functions take a log callable and stay silent by default
scripts/            25 three-line stubs (scripts/<name>.py, scripts/reporting/<name>.py), one per command
```

- **Dispatcher.** `punct_restore.cli.COMMANDS` maps each command name (the module name with `_` written as
  `-`) to its module. `recap`, `recap -h` and `recap --help` list the commands with the first line of each
  module's docstring and exit 0. An unknown name prints `error: unknown command '<name>'; run recap --help
  for the list` on stderr and exits 2. Otherwise the dispatcher returns the module's `main(argv)`, so
  argparse help and usage errors (exit 2) pass through.
- **Command modules** have a module docstring (the help summary, with no team roles) and exactly
  `main(argv: list[str] | None = None) -> int` (`tests/test_main_signature.py`). They parse arguments,
  read and write files and run their command's own steps behind the error boundary. Logic that more than
  one command uses lives in the library modules. `cli/package_hmm_handoff.py` keeps its packaging steps
  because it copies its own source into the handoff. No cli module
  imports another cli module (`tests/test_repo_hygiene.py`). `python -m punct_restore.cli.<module>` (or
  `punct_restore.reporting.<module>`) runs one directly.
- **Errors.** A runtime error (a missing, unreadable or malformed file, a bad value) is one stderr line,
  `error: <file name>: <reason>`, and exit 1, built by `prediction_io.one_line_error`. It names files by
  file name only, never a folder, and prints no traceback. Argparse usage errors stay exit 2, and a code
  bug propagates. `tests/test_malformed_inputs.py` and `tests/test_malformed_inputs_hmm.py` check this
  for every command. An unapproved B → A projection in `configs/label_sets.yaml` is a `ValueError`, so
  `recap prepare-data` reports it the same way.
- **Library calls.** The record, release and HMM helpers return new values instead of changing their
  arguments (`tests/test_arguments_not_mutated.py`, `tests/test_data_release_structure.py`). Long
  parameter lists are frozen dataclasses built with keywords, for example
  `build_release(sources: ReleaseSources, *, log=discard_log)`; `CRFModel.__init__`, a scikit-learn-style
  estimator constructor, keeps its keyword-only parameters. Library functions report progress through a
  `log` callable: the default `progress.discard_log` says nothing, and the commands pass `print`.
- **Stubs.** Each file in `scripts/` imports `main` from its command module and exits with its return
  value, so the old `scripts/<name>.py` command lines keep working. Recorded commands in run manifests
  keep those paths.
- **Split modules.** The size split moved helpers out of long modules (for example `models/crf.py` into
  `crf_io`, `crf_modes`, `crf_records` and `crf_training`). Library import paths from
  `main` at `72ebfd1` still work: every public name in `punct_restore.data`, `punct_restore.evaluation`,
  `punct_restore.features`, `punct_restore.models`, `punct_restore.pipeline` and `punct_restore.prediction_io`
  stays importable from its old module, as re-exports or thin wrappers (`tests/test_library_names.py`,
  pinned to that commit's name list); private `_helper` names are not covered. The classes stored in saved models (`CRFModel`, `FeatureConfig`,
  `HMMModel`, `HiddenEventLM`) stay where they were, so saved models load. Moved helpers also live in their
  new modules, for example `FILLERS` in `features/crf_lexicons.py` and `CASING_VOCABULARY` in
  `models/crf_modes.py`. The internals of the cli modules (such as `cli.run_hmm_experiments.frozen`) are
  not part of that promise. `tests/test_code_limits.py`
  keeps every function in `src/` and `scripts/` at 50 lines or less (from its first decorator to its
  last line), nesting at 4 levels or less and every file at 400 lines or less; `tests/test_type_hints.py`
  requires every function in `src/` to annotate its parameters and return type.
- **HMM study provenance.** `models/hmm_experiment_identity.SOURCE_FILES` lists the study's whole
  `punct_restore` import closure (30 files, the command modules and their stubs included). A test checks it
  against the real imports, and `plan.json` records the SHA-256 of each file. `verified_frozen_selection`
  (`models/hmm_experiment_frozen.py`) is the gate that finalize, verify and package all pass: it refuses a
  study whose recorded hashes differ from the code on disk.

## Ownership

| Area | Primary owner | Boundary |
| --- | --- | --- |
| `data/` (with the `release_*` modules), `cli/prepare_data.py` (`recap prepare-data`), data documentation | Member 1 | Source text → records; no model training |
| `models/hmm.py`, `models/hmm_validation.py`, `models/hmm_records.py`, `models/hmm_workflow.py`, `models/hmm_experiment_*.py`; `cli/train_hmm.py`, `cli/predict_hmm.py`, `cli/run_hmm_experiments.py`, `cli/verify_hmm_experiments.py`, `cli/package_hmm_handoff.py` | Member 2 | Records → HMM artifacts/predictions |
| `models/crf.py` with `crf_modes.py`, `crf_records.py`, `crf_io.py`, `crf_training.py`, `crf_training_data.py`, `crf_prediction.py` and `crf_inspection.py`; `cli/train_crf.py` (`recap train-crf`) | Member 3 | Records + injected features → CRF artifacts/predictions |
| `features/` (`crf_features.py`, `crf_lexicons.py`, `crf_cues.py`, `word_clusters.py`), `cli/run_ablations.py`, `cli/build_word_clusters.py` | Member 4 (role); code by Member 3, ablation runner by Member 5 | Observable tokens → aligned feature dictionaries |
| `evaluation/`, `cli/evaluate_models.py` (`recap evaluate-models`), `reporting/`, results | Member 5 | Gold + predictions → verified reports |
| Comparison methods and robustness checks: `models/gap_classifier.py`, `models/hidden_event_lm.py` (+ `hidden_event_lm_viterbi.py`, `ngram_lm.py`, `truecaser.py`), τ × γ in `models/threshold.py` (`cli/tune_threshold.py`), question lookback, `models/factored.py` and the CRF `CASE` mode, `data/asr_noise.py`, `data/gold_noise.py` (the gold-noise check), the demo loader; commands `recap train-hidden-event-lm`, `recap factor-predictions`, `recap simulate-asr-noise`, `recap find-gold-noise`, `recap project-joint`, `recap build-leakage-subsets`, `recap crf-convergence`, `recap length-position`, `recap asr-noise-table` (+ `reporting/asr_noise_stats.py`, `asr_noise_report.py`, `asr_noise_findings.py`) | Member 5 | Exploratory comparisons and checks next to the course baselines |
| `pipeline.py` (`load_model`, `ModelAdapter`, `RestorationPipeline`) and `cli/run_demo.py` (`recap run-demo`, the demo path), presentation | Member 6 | Text + saved HMM or hidden-event LM (`.json`), or CRF or gap classifier (`.joblib`, `.pkl`) → restored text |
| `cli/__init__.py` (dispatcher), `paths.py`, `hashing.py`, `progress.py`, the shared helpers and error boundary in `prediction_io.py`, the `scripts/` stubs | Team (restructure, 2026-11-01) | Command routing and shared plumbing; no model or data logic |

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
- The demo pipeline does not use the training tokenizer. `RestorationPipeline.process_text` in
  `pipeline.py` splits the raw text on whitespace (`str.split()`). `clean_tokens` then lowercases each
  piece, strips `,.?!:;-"` from both ends, and drops pieces that end up empty; if none is left it raises
  `ValueError("empty input")`. The
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
