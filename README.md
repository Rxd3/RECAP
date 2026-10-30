# RECAP

**Restoring English Casing and Punctuation**

COE025 Natural Language Processing, Group Project 2: sequence labeling.
Repository: [github.com/Rxd3/RECAP](https://github.com/Rxd3/RECAP) (public).
Code licence: [MIT](LICENSE). Data licence: SponSpeech, CC BY-NC 4.0 ([DATA_LICENSE.md](docs/DATA_LICENSE.md)).

RECAP restores punctuation and capitalization in lowercased, unpunctuated English text.
Every token gets a label for the punctuation mark that follows it and, in mode C, a casing
label for the token itself. We compare three models on the same data and splits:

1. a hidden Markov model with Viterbi decoding (baseline);
2. a linear-chain CRF with basic features;
3. the same CRF with improved features, plus one-group-out feature ablations.

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

## What is new

**Data** (built by us from the SponSpeech transcripts; [DATASET_CARD.md](docs/DATASET_CARD.md)):

- **Three label views of the same tokens.** Upstream SponSpeech scores three marks (full stop,
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

**Task and method:**

- **Joint punctuation and casing** (mode C) in one sequence labeler, compared with
  punctuation-only training on identical sequences.
- **The end-of-utterance shortcut, measured.** Every utterance ends in PERIOD or QUESTION, so
  we report a rule-based shortcut baseline and a `non_final` view that drops each
  utterance's last token.
- **Positioned against prior work** in [RELATED_WORK.md](docs/RELATED_WORK.md): classical
  sequence labelers on spontaneous speech rather than transformers on TED talks.
- **Grouped bootstrap by video** for every 95% interval, and paired intervals for every model
  comparison.
- **Feature ablations** on validation only, one feature group at a time.
- **Two team extensions, tuned on validation only:**
  - threshold decoding of the CRF marginals (`scripts/tune_threshold.py`);
  - a filler-word cue (`improved_plus_filler` in `scripts/run_ablations.py`).
- **The cost of leakage, measured.** On official test utterances whose video is in the training set, the frozen improved CRF scores 67.09, against 49.50 on the leak-free test.

## Key findings

Every number names its split.
- **Primary metric:** macro F1 % over the non-O punctuation classes (A, B), or over all 12 joint classes on unmasked positions (C).
- **Intervals:** 95%, grouped bootstrap by video, 1,000 resamples.
- **Modes B and C are different tasks;** never rank one against the other.
- **Selection:** validation chose every setting, and each frozen model predicted the primary test lists once. The mode A improved CRF also predicted the full official test once more, for the leakage analysis.

**Test, primary ID lists** ([EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §1):

| Mode | Shortcut | HMM | CRF basic | CRF improved | CRF improved + threshold |
| --- | --- | --- | --- | --- | --- |
| A | 27.48 | 35.12 (32.80–37.31) | 43.65 (41.74–45.59) | 49.92 (47.91–52.01) | **50.44** (48.53–52.41) |
| B | 11.87 | 16.79 (15.28–18.49) | 20.25 (18.87–21.95) | 22.82 (21.42–24.42) | **23.23** (21.74–25.05) |
| C | 18.09 | 37.41 (34.56–40.71) | 43.85 (42.13–45.47) | 47.02 (44.97–49.00) | **47.72** (45.78–49.61) |

- **The improved CRF beats the HMM** by +14.80 (A), +6.03 (B) and +9.61 (C) points on test. It beats the basic CRF by +6.27, +2.57 and +3.17, and the basic CRF beats the HMM by +8.53, +3.46 and +6.43. All nine paired intervals exclude 0, and the ranking holds on test_amb.
- **The HMM mostly learns the end-of-utterance shortcut.**
  - The shortcut alone reaches PERIOD F1 66.68 in mode A, against the HMM's 67.34 (test).
  - Without each utterance's final token, mode A scores HMM 9.13, CRF basic 28.30 and CRF improved 36.40 (test).
- **The CRF gain is mostly COMMA and QUESTION** (mode A test F1, HMM → CRF improved): COMMA 26.28 → 52.52, QUESTION 11.75 → 26.23, PERIOD 67.34 → 71.02.
- **Threshold decoding** adds +0.51 / +0.41 / +0.70 on test, mostly through comma recall. The B interval includes 0.
- **The filler-word cue has no detectable effect** on validation: every interval includes 0.
- **Leakage inflates lexical models.** On test utterances from videos in the training set, the improved CRF scores 67.09 against 49.50 on the leak-free test, while the HMM scores 33.54 against 34.74. See [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §8.
- **Casing** (mode C, masked casing F1, test): HMM 79.81, CRF improved 85.09.
- **Ablations** (validation, 50k-sequence training sample, [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §6):
  - Removing word pairs, the ±2 window or the question cues costs points in modes A and B.
  - Clause, casing and position cues have small effects of mixed sign.

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
| `configs/` | `configs/label_sets.yaml` (label vocabularies), `configs/datasets.yaml` (source checklist), `configs/hmm_sponspeech_v2.yaml` (frozen HMM search), `configs/experiment.example.yaml` (run template). |
| `contributions/` | One file per team member: role, branch, pull request, commits and files. |
| `data/` | Ignored local data: `data/raw/`, `data/processed/`, `data/splits/`. `data/README.md` explains how to rebuild or download a release. |
| `docs/` | Contracts, dataset documentation, decisions, evaluation protocol and results; listed below. |
| `experiments/` | Model notes and runbooks: `experiments/HMM_NOTES.md`, `experiments/HMM_RUNBOOK.md`, `experiments/HMM_RESULTS.md`, `experiments/CRF_NOTES.md`, `experiments/FEATURE_NOTES.md`. Run outputs go to the ignored `experiments/runs/`. |
| `presentation/` | Presentation workspace; the outline is `docs/PRESENTATION_OUTLINE.md`. |
| `results/` | Ignored evaluator outputs: `results/predictions/`, `results/tables/`, `results/figures/`. |
| `scripts/` | Command-line entry points; see the table below and `scripts/README.md`. |
| `src/punct_restore/` | The package: `src/punct_restore/data/` (extraction, labeling, preprocessing, release, splits, validation), `src/punct_restore/models/` (`src/punct_restore/models/hmm.py`, `src/punct_restore/models/hmm_workflow.py`, `src/punct_restore/models/crf.py`), `src/punct_restore/features/crf_features.py`, `src/punct_restore/evaluation/` (metrics, views, bootstrap, comparison, errors, reports) and `src/punct_restore/pipeline.py` (model loading for the demo). |
| `tests/` | The pytest suite for data, models, features, CLIs and evaluation; fixtures only. |

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

**Docs:**

| File | Content |
| --- | --- |
| `docs/PROJECT_SCOPE.md`, `docs/ARCHITECTURE.md` | Scope, components and ownership. |
| `docs/DATASET_CARD.md` | What SponSpeech ships, what RECAP adds, splits, labels, leakage, licence. |
| `docs/DATASET_PLAN.md` | Source, annotation policy, released counts and limitations (preprocessing). |
| `docs/DATA_CONTRACT.md` | Record schema, alignment and approved projections. |
| `docs/DATA_RELEASE_VALIDATION.md` | Checks and fingerprints of release `sponspeech-m1-v2`. |
| `docs/DATA_LICENSE.md` | SponSpeech licence and the upstream attribution text. |
| `docs/DECISIONS.md` | Approved data decisions D1–D6 and the open questions D7–D12. |
| `docs/MODEL_CONTRACT.md` | Model inputs, prediction records and manifests. |
| `docs/EVALUATION_PROTOCOL.md` | Metrics, views, masks, bootstrap and known gold issues. |
| `docs/EVALUATION_RESULTS.md` | HMM and shortcut results (validation, test, test_amb). |
| `docs/EVALUATION_RESULTS_CRF.md` | CRF test results, validation selection, ablations, threshold decoding, filler cue, leakage, runtime. |
| `docs/CRF_EXPERIMENTS.md` | The six CRF runs: features, settings, hashes, commands. |
| `docs/RELATED_WORK.md` | Prior work, checked online, and how RECAP differs. |
| `docs/REPRODUCIBILITY.md` | Every command, with the environments actually used. |
| `docs/PRESENTATION_OUTLINE.md` | Four-slide outline with verified numbers. |
| `docs/GIT_WORKFLOW.md`, `docs/MEMBER2_COMPLETION.md`, `docs/INITIAL_VERIFICATION.md`, `docs/REFERENCE_REVIEW.md` | Git workflow, the HMM completion audit, and historical setup records. |

## Team

| Member | Role | Branch | Contribution file |
| --- | --- | --- | --- |
| 1 Mutasim | Dataset and preprocessing | `feat/m1-data-preprocessing` | `contributions/Mutasim_STUDENT_ID_Sam-Ali23.md` |
| 2 Raşit (lead) | HMM baseline | `feat/m2-hmm-baseline` | `contributions/Rasit_STUDENT_ID_Rxd3.md` |
| 3 Omar | CRF model and improved features | `feat/m3-crf-model` | `contributions/Omar_STUDENT_ID_Devinci101.md` |
| 4 Mohamed | Feature engineering | `feat/m4-feature-engineering` | `contributions/Mohamed_STUDENT_ID_GITHUB.md` |
| 5 Abd Raouf | Evaluation and error analysis | `feat/m5-evaluation` | `contributions/AbdRaouf_STUDENT_ID_afel6.md` |
| 6 Abdulla | Integration and presentation | `feat/m6-integration-presentation` | `contributions/Abdulla_STUDENT_ID_abdu3061.md` |

See [CONTRIBUTING.md](CONTRIBUTING.md) and [GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md) for the
branch and pull-request workflow.

## Licence

Code and documentation: MIT ([LICENSE](LICENSE)). SponSpeech data and anything derived from
it: CC BY-NC 4.0, non-commercial use only, with the upstream attribution
([DATA_LICENSE.md](docs/DATA_LICENSE.md)).
