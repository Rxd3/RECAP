# RECAP

**Restoring English Casing and Punctuation**

COE025 Natural Language Processing, Group Project 2: sequence labeling.
Repository: [github.com/Rxd3/RECAP](https://github.com/Rxd3/RECAP) (public).
Code licence: [MIT](LICENSE). Data licence: SponSpeech, CC BY-NC 4.0 ([DATA_LICENSE.md](docs/DATA_LICENSE.md)).

RECAP restores punctuation and capitalization in lowercased, unpunctuated English text.
Every token gets a label for the punctuation mark that follows it and, in mode C, a casing
label for the token itself. The course asks for three models, which we build as the baselines and
compare on the same data and splits:

1. a hidden Markov model with Viterbi decoding;
2. a linear-chain CRF with basic features;
3. the same CRF with improved features.

What we added beyond these baselines is listed under [What is new](#what-is-new).

The data are the reference transcripts of SponSpeech, informal podcast-style English speech
from YouTube. RECAP uses the transcripts only. It does not use audio and was not evaluated
on speech-recognition (ASR) output.

## Task

Input (model view: lowercased tokens, no punctuation):

```text
hello everyone welcome to our presentation today we will discuss natural language processing
```

Target (illustrative, not a model prediction):

```text
Hello everyone, welcome to our presentation. Today we will discuss natural language processing.
```

| Mode | Target | Labels |
| --- | --- | --- |
| A | Standard punctuation | O, COMMA, PERIOD, QUESTION (4) |
| B | Extended punctuation | mode A plus EXCLAMATION, COLON, SEMICOLON, DASH (8) |
| C | Joint punctuation and casing | mode A × LOWER, FIRST_UPPER, ALL_UPPER (4 × 3 = 12) |

A mode C label reads `PERIOD|FIRST_UPPER`: the mark after this word, and the casing of this
word. Modes A and B predict no casing. [configs/label_sets.yaml](configs/label_sets.yaml) is
the source of truth for the vocabularies; [DATA_CONTRACT.md](docs/DATA_CONTRACT.md) defines the
record format and alignment.

## Models

| Architecture | Trained model | Settings | Source |
| --- | --- | --- | --- |
| First-order HMM, Viterbi decoding | HMM, one per mode | additive smoothing α = 0.01 in A/B/C; rare-word cutoff 1 (A, B) and 2 (C); chosen on validation from 9 candidates per mode | [HMM_RESULTS.md](experiments/HMM_RESULTS.md) |
| Linear-chain CRF (`sklearn-crfsuite`, L-BFGS) | CRF basic preset | ±1 window, affixes, token shape, utterance boundaries; c1 = c2 = 0.1, 100 iterations, fixed | [CRF_EXPERIMENTS.md](docs/CRF_EXPERIMENTS.md) §3, §6 |
| Linear-chain CRF (`sklearn-crfsuite`, L-BFGS) | CRF improved preset | basic plus ±2 window, word pairs, question, clause, casing and position cues; same settings | [CRF_EXPERIMENTS.md](docs/CRF_EXPERIMENTS.md) §3 |
| Linear-chain CRF + threshold decoding (ours) | CRF improved + τ | a token is non-O when 1 − P(O) > τ; τ = 0.40 (A), 0.30 (B), 0.35 (C), chosen on validation | [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §7 |

## What is new

**Course baselines** (what the assignment asks for, not our novelty): the HMM, the basic CRF and the
improved-feature CRF above, in modes A, B and C. Member 3 wrote both CRF feature presets.

**Our additions: data** (built from the SponSpeech transcripts; [DATASET_CARD.md](docs/DATASET_CARD.md)):

- **Extended label views of the same tokens.** Upstream SponSpeech scores three marks (full stop,
  comma, question mark). We add an 8-class extended view (mode B), a 3-way casing label with
  per-position masks for approximate casing, and a 12-class joint view (mode C).
- **One documented projection.** Mode A is the projection of mode B: EXCLAMATION and SEMICOLON
  become PERIOD, COLON and DASH become COMMA ([DECISIONS.md](docs/DECISIONS.md), D5).
- **Leak-free primary splits.** The official splits share source videos. Our primary
  validation lists drop every utterance whose video also appears in train, and our primary test
  lists drop every utterance whose video appears in train or validation, plus exact and near
  duplicates. In the official test file, 4,716 of 17,695 utterances
  (26.7%) come from such videos ([DATASET_PLAN.md](docs/DATASET_PLAN.md)).
- **A gold-noise audit.** Leftover speaker labels, subtitle cue numbers and auto-caption-style
  videos are listed, not relabelled ([EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md)).

**Our additions: analysis and methods:**

- **The end-of-utterance shortcut, measured.** Every utterance ends in PERIOD or QUESTION, so
  we report a rule-based shortcut baseline and a `non_final` view that drops each
  utterance's last token.
- **The cost of leakage, measured.** The frozen models also predicted the full official test, split
  by why each utterance was excluded ([EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §8).
- **Threshold decoding** of the CRF marginals (`scripts/tune_threshold.py`), with τ tuned on validation.
- **A filler-word cue** (`improved_plus_filler` in `scripts/run_ablations.py`), tested on validation.
- **Joint vs punctuation-only (H1).** Each mode C model, projected to punctuation, is compared with
  its mode A twin on mode A gold over the same 19,936 mode C validation IDs. The training sets
  differ: A uses 142,722 sequences and C uses 141,745.

**Evaluation rigor:** grouped bootstrap by video for every 95% interval, paired intervals for every
model comparison, and one-group-out feature ablations on validation only. Prior work and how RECAP
differs: [RELATED_WORK.md](docs/RELATED_WORK.md).

## Key findings

Every number names its split.
- **Primary metric:** macro F1 % over the non-O punctuation classes (A, B), or over all 12 joint classes on unmasked positions (C).
- **Intervals:** 95%, grouped bootstrap by video, 1,000 resamples.
- **Modes B and C are different tasks;** never rank one against the other.
- **Selection:** Validation chose the HMM smoothing and the threshold τ; the CRF settings (c1 = c2 = 0.1, 100 iterations) were fixed and not tuned; no test result influenced any choice. Each frozen model predicted the primary test lists once. The mode A improved CRF also predicted the full official test once more, for the leakage analysis.

**Course baselines on test, primary ID lists** ([EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §1;
the shortcut and threshold columns are our additions):

| Mode | Shortcut (ours) | HMM | CRF basic | CRF improved | CRF improved + threshold (ours) |
| --- | --- | --- | --- | --- | --- |
| A | 27.48 | 35.12 (32.80–37.31) | 43.65 (41.74–45.59) | 49.92 (47.91–52.01) | **50.44** (48.53–52.41) |
| B | 11.87 | 16.79 (15.28–18.49) | 20.25 (18.87–21.95) | 22.82 (21.42–24.42) | **23.23** (21.74–25.05) |
| C | 18.09 | 37.41 (34.56–40.71) | 43.85 (42.13–45.47) | 47.02 (44.97–49.00) | **47.72** (45.78–49.61) |

- **The improved CRF beats the HMM** by +14.80 (A), +6.03 (B) and +9.61 (C) points on test. It beats the basic CRF by +6.27, +2.57 and +3.17, and the basic CRF beats the HMM by +8.53, +3.46 and +6.43. All nine paired intervals exclude 0, and the ranking holds on test_amb.
- **The CRF gain is mostly COMMA and QUESTION** (mode A test F1, HMM → CRF improved): COMMA 26.28 → 52.52, QUESTION 11.75 → 26.23, PERIOD 67.34 → 71.02.
- **Casing** (mode C, masked casing F1, test): HMM 79.81, CRF improved 85.09.
- **Feature ablations of the improved preset** (validation, 50k-sequence training sample, [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §6):
  - Removing word pairs, the ±2 window or the question cues costs points in modes A and B.
  - Clause, casing and position cues have small effects of mixed sign.

**Our additions:**

- **The HMM mostly learns the end-of-utterance shortcut.**
  - The shortcut alone reaches PERIOD F1 66.68 in mode A, against the HMM's 67.34 (test).
  - Without each utterance's final token, mode A scores HMM 9.13, CRF basic 28.30 and CRF improved 36.40 (test).
- **Threshold decoding** adds +0.51 / +0.41 / +0.70 on test, mostly through comma recall. The B interval includes 0.
- **The filler-word cue has no detectable effect** on validation: every interval includes 0.
- **Leakage inflates lexical models.** On test utterances from videos in the training set, the improved CRF scores 67.09 against 49.50 on the leak-free test, while the HMM scores 33.54 against 34.74. See [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §8.
- **Joint vs punctuation-only (H1, validation, same 19,936 evaluation IDs):** every joint model gains
  on non-final positions (HMM +8.69, CRF basic +1.59, CRF improved +1.16; all intervals exclude 0), but
  PERIOD F1 does not rise consistently, so H1 as stated is not supported.

## Quickstart

Python 3.11 or newer, run from the repository root. Versions actually used are recorded in
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

The tests use small fixtures and download nothing. With a trained model (an HMM `.json` or a
CRF `.joblib`), restore one sentence:

```powershell
python scripts/run_demo.py --model-path <model.json or model.joblib> --text "hello everyone welcome to our presentation"
```

Every command, from rebuilding the data to scoring a model, is in
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Data access

- **Upstream:** SponSpeech (Liu & Beigi, 2024, [arXiv:2409.11241](https://arxiv.org/abs/2409.11241)),
  CC BY-NC 4.0. No data is committed to this repository.
- **Rebuild from source:** `scripts/prepare_data.py` with `extract` reads only about 97 MB of pinned
  byte ranges from the official archive; `build` creates release `sponspeech-m1-v2`. Steps
  and checksums: [data/README.md](data/README.md).
- **Team copy:** releases are shared in the team's Google Drive folder
  [RECAP -NLP2](https://drive.google.com/drive/folders/1ZKDNnRHXsLJf3UjOdKVgQ0bYpo12IQfH?usp=sharing),
  with SHA-256 sums ([data/README.md](data/README.md)).
- **Preprocessing:** normalization, label and casing rules, masks and split construction are in
  [DATASET_PLAN.md](docs/DATASET_PLAN.md) and [DATA_CONTRACT.md](docs/DATA_CONTRACT.md); the
  release checks are in [DATA_RELEASE_VALIDATION.md](docs/DATA_RELEASE_VALIDATION.md).

Train on `default_train/mode_<X>.ids`, select on `primary_eval/validation_mode_<X>.ids`, and
report `test` and `test_amb` separately (they share 25 videos).

## Folder and script map

| Path | What it holds |
| --- | --- |
| `artifacts/` | Ignored local model and data bundles (handoff ZIPs). Only `artifacts/README.md` is tracked. |
| `configs/` | `configs/label_sets.yaml` (label vocabularies), `configs/datasets.yaml` (source checklist), `configs/hmm_sponspeech_v2.yaml` (frozen HMM search). |
| `contributions/` | One file per team member: role, branch, pull request, commits and files. |
| `data/` | Ignored local data: `data/raw/`, `data/processed/`, `data/splits/`. `data/README.md` explains how to rebuild or download a release. |
| `docs/` | Contracts, dataset documentation, decisions, evaluation protocol and results; listed below. Index: `docs/README.md`. |
| `experiments/` | Model notes and runbooks: `experiments/HMM_NOTES.md`, `experiments/HMM_RUNBOOK.md`, `experiments/HMM_RESULTS.md`, `experiments/CRF_NOTES.md`, `experiments/FEATURE_NOTES.md`. Run outputs go to the ignored `experiments/runs/`. |
| `presentation/` | Presentation workspace (`presentation/README.md`); the outline is `docs/PRESENTATION_OUTLINE.md`. |
| `results/` | `results/scores/` holds the committed evaluator JSON behind every reported number (`results/scores/README.md`), including Member 3's six CRF training manifests in `results/scores/crf_training_manifests/`. Local outputs in `results/predictions/` and `results/tables/` (kept as empty folders, because the evaluator never creates folders) are ignored. |
| `scripts/` | Command-line entry points; see the table below and `scripts/README.md`. |
| `src/punct_restore/` | The Python package; every module is listed in the package table below. |
| `tests/` | The pytest suite for data, models, features, CLIs and evaluation; fixtures only. |

**Package** (`src/punct_restore/`; each package folder `src/punct_restore/`, `src/punct_restore/data/`,
`src/punct_restore/models/`, `src/punct_restore/features/` and `src/punct_restore/evaluation/` has a
package initializer, `<folder>/__init__.py`):

| Module | Role |
| --- | --- |
| `src/punct_restore/data/` | Member 1's data pipeline: `extraction` (raw transcripts from the official archive), `preprocessing` (loading, cleaning, tokenization), `labeling` (punctuation and casing labels), `splits` (split assignment and leakage audits), `release` (versioned release build and load), `validation` (checks against the data contract). |
| `src/punct_restore/models/hmm.py` | Supervised HMM with Viterbi decoding. |
| `src/punct_restore/models/hmm_workflow.py` | Validated file selection and provenance for the HMM commands. |
| `src/punct_restore/models/crf.py` | Linear-chain CRF: fit, predict, marginals, save and load. |
| `src/punct_restore/models/threshold.py` | Threshold decoding of CRF marginals (τ tuned on validation only). |
| `src/punct_restore/features/crf_features.py` | CRF features: basic and improved presets, ablation switches and the filler cue. |
| `src/punct_restore/evaluation/` | Member 5's evaluator: `inputs` (input checks), `views` (scored positions and report blocks), `metrics` (token-level P/R/F1), `uncertainty` (bootstrap grouped by video), `comparison` (paired comparison), `error_analysis` (error examples), `baseline` (shortcut baseline), `manifest` (prediction-manifest check), `report` (JSON and Markdown reports). |
| `src/punct_restore/prediction_io.py` | Gold-free prediction JSONL and manifest writing, shared by `scripts/train_crf.py` and `scripts/tune_threshold.py`. |
| `src/punct_restore/pipeline.py` | Demo inference: load a trained HMM or CRF, clean raw text, predict labels, rebuild text. |

Top-level files: `LICENSE` (MIT), `CONTRIBUTING.md` (daily Git checklist), `TEAM_TASKS.md`
(work board), `pyproject.toml` and `requirements.txt` (package and dependencies),
`requirements-hmm.txt` (the pinned HMM reproduction environment), `.gitignore`, `.gitattributes`.

**Scripts** (run from the repository root; each has `--help`):

| Script | Stage | What it does |
| --- | --- | --- |
| `scripts/prepare_data.py` | Data | `extract` recreates the raw transcripts from pinned archive ranges; `build` writes a validated, immutable release. |
| `scripts/train_hmm.py` | HMM | Trains the HMM on an explicit ID list; writes a JSON artifact, a run manifest and optional predictions. |
| `scripts/predict_hmm.py` | HMM | Predicts with a saved HMM artifact, without training. |
| `scripts/run_hmm_experiments.py` | HMM | `tune` runs the frozen validation search; `finalize` predicts and scores the selected models. |
| `scripts/verify_hmm_experiments.py` | HMM | Audits an HMM study: recounts scores, refits the winners and compares predictions. |
| `scripts/package_hmm_handoff.py` | HMM | Packages a verified HMM study as a checksummed ZIP. |
| `scripts/train_crf.py` | CRF | `train` fits a CRF (basic or improved features); `predict` writes predictions and a manifest; `inspect` prints artifact metadata. |
| `scripts/run_ablations.py` | CRF | Trains the feature-group ablations and writes their validation predictions (never reads test). |
| `scripts/tune_threshold.py` | CRF | Threshold decoding: `tune` picks τ on validation, `predict` applies the frozen τ. |
| `scripts/evaluate_models.py` | Evaluation | `score`, `compare` (paired), `errors` and `baseline` (the shortcut). |
| `scripts/run_demo.py` | Demo | Restores punctuation and casing for one text with a saved HMM or CRF. |
| `scripts/project_joint.py` | Study (H1) | Projects mode C predictions onto mode A punctuation labels, so a joint model can be compared with its mode A twin. |
| `scripts/build_leakage_subsets.py` | Study (leakage) | Splits an official evaluation file into kept and excluded ID lists by exclusion cause, for the leakage-cost analysis. |
| `scripts/reporting/build_m3_tables.py` | Reporting | Rebuilds the CRF result tables for one split (main, paired, views, per class, errors, confusion) from `results/scores/`. |
| `scripts/reporting/build_m3_final_sections.py` | Reporting | Rebuilds the threshold, leakage, test_amb and runtime sections from `results/scores/` and the training manifests. |
| `scripts/reporting/build_ablation_table.py` | Reporting | Rebuilds the feature-ablation tables from `results/scores/ablations/`. |
| `scripts/reporting/time_models.py` | Reporting | Times HMM or CRF prediction (and an HMM fit) with the protocol used in the runtime tables. |
| `scripts/reporting/build_tables.py` | Reporting | Helper module, not run directly: shared helpers for the table builders (half-up percentages, cells, JSON loading). |
| `scripts/reporting/_cli.py` | Reporting | Helper module, not run directly: shared one-line error handling for the reporting scripts. |

**Docs:**

| File | Content |
| --- | --- |
| `docs/PROJECT_SCOPE.md`, `docs/ARCHITECTURE.md` | Scope, components and ownership. |
| `docs/DATASET_CARD.md` | What SponSpeech ships, what RECAP adds, splits, labels, leakage, licence. |
| `docs/DATASET_PLAN.md` | Source, annotation policy, released counts and limitations (preprocessing). |
| `docs/DATA_CONTRACT.md` | Record schema, alignment and approved projections. |
| `docs/DATA_RELEASE_VALIDATION.md` | Checks and fingerprints of release `sponspeech-m1-v2`. |
| `docs/DATA_LICENSE.md` | SponSpeech licence and the upstream attribution text. |
| `docs/DECISIONS.md` | Data decisions D1–D6 and the resolved decisions D7–D12. |
| `docs/MODEL_CONTRACT.md` | Model inputs, prediction records and manifests. |
| `docs/EVALUATION_PROTOCOL.md` | Metrics, views, masks, bootstrap and known gold issues. |
| `docs/EVALUATION_RESULTS.md` | HMM and shortcut results (validation, test, test_amb). |
| `docs/EVALUATION_RESULTS_CRF.md` | CRF test results, validation selection, ablations, threshold decoding, filler cue, leakage, runtime. |
| `docs/CRF_EXPERIMENTS.md` | The six CRF runs: features, settings, hashes, commands. |
| `docs/RELATED_WORK.md` | Prior work, checked online, and how RECAP differs. |
| `docs/REPRODUCIBILITY.md` | Every command, with the environments actually used. |
| `docs/PRESENTATION_OUTLINE.md` | Four-slide outline with verified numbers. |

## Team

| Member | Role | Branch | Contribution file |
| --- | --- | --- | --- |
| 1 Mutasim | Dataset and preprocessing | `feat/m1-data-preprocessing` | `contributions/Mutasim_STUDENT_ID_Sam-Ali23.md` |
| 2 Raşit (lead) | HMM baseline | `feat/m2-hmm-baseline` | `contributions/Rasit_STUDENT_ID_Rxd3.md` |
| 3 Omar | CRF model and improved features | `feat/m3-crf-model` | `contributions/Omar_STUDENT_ID_Devinci101.md` |
| 4 Mohamed | Feature engineering | `feat/m4-feature-engineering` | `contributions/Mohamed_STUDENT_ID_GITHUB.md` |
| 5 Abd Raouf | Evaluation and error analysis | `feat/m5-evaluation` | `contributions/AbdRaouf_STUDENT_ID_afel6.md` |
| 6 Abdulla | Integration and presentation | `feat/m6-integration-presentation` | `contributions/Abdulla_STUDENT_ID_abdu3061.md` |

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch, commit and pull-request rules.

## Licence

Code and documentation: MIT ([LICENSE](LICENSE)). SponSpeech data and anything derived from
it: CC BY-NC 4.0, non-commercial use only, with the upstream attribution
([DATA_LICENSE.md](docs/DATA_LICENSE.md)).
