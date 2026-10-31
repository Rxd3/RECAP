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

| Architecture | Implementation | Specific model | Settings |
| --- | --- | --- | --- |
| First-order HMM + Viterbi | own code, `src/punct_restore/models/hmm.py` | `hmm-sponspeech-v2-20261029` bundle, one model per mode | additive smoothing α = 0.01 (A, B, C); rare-word cutoff 1 (A, B) and 2 (C); chosen on validation from 9 candidates per mode |
| Linear-chain CRF | CRFsuite via `sklearn-crfsuite` 0.5.0, L-BFGS; `src/punct_restore/models/crf.py` | `crf_mode_<a\|b\|c>_basic_full_01` (basic preset: ±1 window, affixes, token shape, utterance boundaries) | c1 = c2 = 0.1, 100 iterations, fixed |
| Linear-chain CRF | same | `crf_mode_<a\|b\|c>_improved_full_01` (improved preset: adds ±2 window, word pairs, question, clause, casing and position cues) | c1 = c2 = 0.1, 100 iterations, fixed |
| + threshold decoding (ours) | `src/punct_restore/models/threshold.py` on the improved CRF's marginals | improved CRF + τ: a token is non-O when 1 − P(O) > τ | τ = 0.40 (A), 0.30 (B), 0.35 (C), chosen on validation |
| + prior-adjusted threshold decoding (ours, exploratory) | `src/punct_restore/models/threshold.py` with add-one training label priors (`tune_threshold.py tune --train --train-ids`) | improved CRF + τ × γ: after the τ gate, the non-O label maximises P(c) / prior(c)^γ | τ = 0.4, γ = 0.5 (A); τ = 0.45, γ = 0.5 (B); τ = 0.35, γ = 0.5 (C), chosen on validation |
| Hidden-event n-gram LM + Viterbi (ours, exploratory) | own code, `src/punct_restore/models/hidden_event_lm.py`; mode C adds a unigram truecaser | one model per mode | order 4 with discount 0.9 (A), 0.75 (B, C), chosen on validation from order ∈ {2, 3, 4} × discount ∈ {0.5, 0.75, 0.9} |

Sources: [HMM_RESULTS.md](experiments/HMM_RESULTS.md), [CRF_EXPERIMENTS.md](docs/CRF_EXPERIMENTS.md) §3, §6–7
and [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §7, §14.2–14.3. Where to download each model:
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## What is new

**Course baselines** (what the assignment asks for, not our novelty): the HMM, the basic CRF and the
improved-feature CRF above, in modes A, B and C. Member 3 wrote both CRF feature presets.

**Our additions: data** (built from the SponSpeech transcripts; [DATASET_CARD.md](docs/DATASET_CARD.md)):

- **Labels for the course's modes B and C, derived from SponSpeech.** The course specifies the 8-class
  mode B and the 12-class joint mode C; upstream SponSpeech scores only three marks (full stop, comma,
  question mark). We derived the mode B labels, a 3-way casing label with per-position masks for
  approximate casing, and the mode C joint labels from the SponSpeech transcripts.
- **One documented projection.** Mode A is the projection of mode B: EXCLAMATION and SEMICOLON
  become PERIOD, COLON and DASH become COMMA ([DECISIONS.md](docs/DECISIONS.md), D5).
- **Leak-free primary splits.** The official splits share source videos. Our primary
  validation lists drop every utterance whose video also appears in train, and our primary test
  lists drop every utterance whose video appears in train or validation, plus exact and near
  duplicates. In the official test file, 4,716 of 17,695 utterances
  (26.7%) come from such videos ([DATASET_PLAN.md](docs/DATASET_PLAN.md)).
- **A gold-noise audit** (`scripts/find_gold_noise.py`). Leftover speaker labels, subtitle cue residue and
  auto-caption-style videos are flagged, not relabelled. Speaker labels give about a third of the validation
  COLONs (144 of 425), so the COLON scores are an upper bound
  ([EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §12).

**Our additions: analysis and methods:**

- **The end-of-utterance shortcut, measured.** Every utterance ends in PERIOD or QUESTION, so
  we report a rule-based shortcut baseline and a `non_final` view that drops each
  utterance's last token.
- **The cost of leakage, measured.** The frozen models also predicted the full official test, split
  by why each utterance was excluded ([EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §8).
- **Threshold decoding** of the CRF marginals (`scripts/tune_threshold.py`), with τ tuned on validation.
- **Comparison methods and robustness checks** (exploratory, added on 2026-10-30 after the first test report; settings chosen on validation): a gap classifier (the CRF features without transitions), a
  Stolcke-style hidden-event language model, prior-adjusted threshold decoding (τ × γ), question-lookback
  features, data-driven word clusters (PPMI + SVD + k-means from the training text), a factored casing pipeline
  (punctuation CRF + casing-only CRF), and simulated ASR noise on the test
  inputs (results in [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md)).
- **A filler-word cue** (`improved_plus_filler` in `scripts/run_ablations.py`), tested on validation.
- **Word-cluster features** (`improved_plus_clusters` in `scripts/run_ablations.py`, with `--cluster-map` from
  `scripts/build_word_clusters.py`): clusters induced from `default_train` only (PPMI, SVD, k-means with k = 100 or 300),
  tested on validation.
- **Label granularity (B→A):** mode B models projected to the four mode A labels with the D5 mapping,
  compared with their mode A twins on mode A validation gold
  ([EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §11).
- **Per-class intervals** (`evaluate_models.py compare --class`): paired 95% intervals for single classes,
  used for hypotheses H2–H4 (EVALUATION_RESULTS_CRF.md, Team hypotheses).
- **A sensitivity check of the fixed CRF settings:** both presets retrained on the same 50k training
  sample as the ablations at c1 = c2 ∈ {0.01, 0.1, 1.0} and 100 or 300 iterations (EVALUATION_RESULTS_CRF.md §5).
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
- **Selection:** The course-baseline settings (HMM smoothing, CRF settings, the plain threshold τ) were fixed or chosen on validation before any test result. The comparison methods in sections 13–14 of [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) (simulated ASR noise, gap classifier, hidden-event LM, prior-adjusted decoding, question lookback, factored casing) were added on 2026-10-30, after the first test report, in response to a review that cited test results. They are exploratory; their settings were still chosen on validation only, and each test list was predicted once. The CRF settings were c1 = c2 = 0.1 and 100 iterations, not tuned. Each frozen model predicted the primary test lists once. The mode A improved CRF also predicted the full official test once more, for the leakage analysis.

**Course baselines on test, primary ID lists** ([EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §1;
the shortcut, hidden-event LM and threshold columns are our additions):

| Mode | Shortcut (ours) | HMM | Hidden-event LM (ours) | CRF basic | CRF improved | CRF improved + prior-adjusted threshold (ours) |
| --- | --- | --- | --- | --- | --- | --- |
| A | 27.48 | 35.12 (32.80–37.31) | 48.38 (46.21–50.62) | 43.65 (41.74–45.59) | 49.92 (47.91–52.01) | **51.34** (49.61–53.06) |
| B | 11.87 | 16.79 (15.28–18.49) | 22.51 (21.08–23.98) | 20.25 (18.87–21.95) | 22.82 (21.42–24.42) | **24.25** (23.06–25.54) |
| C | 18.09 | 37.41 (34.56–40.71) | 41.18 (39.42–42.70) | 43.85 (42.13–45.47) | 47.02 (44.97–49.00) | **49.77** (47.26–51.95) |

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

**Exploratory comparison methods** (added on 2026-10-30 after the first test report; settings chosen on validation; each test list predicted once):

- **Prior-adjusted threshold decoding** gives +1.42 / +1.43 / +2.74 over the improved CRF on test, all surviving a Holm correction. It lifts rare marks such as QUESTION (A 26.23 → 30.24; +4.00, 1.88 to 6.13) and EXCLAMATION (B 1.85 → 5.45; +3.60, 1.21 to 6.33), and beats the plain threshold by +0.91 / +1.03 / +2.04 on test, every interval excluding 0 (exploratory comparisons, not Holm-corrected). It costs PERIOD F1: −1.79 (−2.20 to −1.39) in A and −1.59 (−1.87 to −1.31) in B. In mode C, 1.45 of the +2.74 gain comes from two tiny classes, `QUESTION|ALL_UPPER` (9 test tokens) and `PERIOD|ALL_UPPER` (125). A plain threshold adds only +0.51 / +0.41 / +0.70. H2 (single τ) not supported; prior-adjusted decoding is an exploratory follow-up.
- **A classic hidden-event language model** (Stolcke-style, n-gram) comes close to the improved CRF on punctuation: −1.54 in A and a tie in B on test. It is 5.84 points behind in mode C, where its unigram truecaser is weak.
- **A gap classifier** (the same features without label-to-label transitions) loses 1.27 / 1.70 / 4.74 to the improved CRF on validation (50k sample). The gap is transitions plus training differences: the gap classifier is an L2 logistic regression with C = 1.0 and 200 iterations, while the CRF uses c1 = c2 = 0.1 and 100 iterations.
- **Question-lookback features** (a wh-word plus an auxiliary anywhere in the last 12 words, and tag questions) raise QUESTION F1 by 6–7 points in every mode on validation (+6.11 / +7.12 / +6.01, every interval excludes 0).
- **Punctuation and casing as separate models** beat the joint mode C model on the joint metric (+2.08, validation, 50k sample), because a dedicated casing CRF is better at casing. Joint training still helps punctuation inside the utterance.
- **Data-driven word clusters** (K = 300; validation, 50k sample) raise COMMA F1 by +0.44 in A and +0.25 in B and the mode C joint score by +0.90; the mode A macro score does not move ([EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §14.6).
- **Under simulated ASR noise** (15% word errors) the ranking holds: the improved CRF drops from 49.92 to 45.42 in mode A, still well above the HMM (35.16). With a more realistic, substitution-heavy edit mix over three seeds ([ASR_NOISE_V2.md](docs/ASR_NOISE_V2.md)), it drops to 43.46 (43.10–43.88) and the HMM to 33.82; the ranking holds for every seed.

**Other analyses:**

- **The filler-word cue** leaves macro F1 unchanged on validation, but raises comma F1 slightly: +0.21 (B) and +0.25 (C), with both intervals excluding 0.
- **Training on the 8-class labels barely changes the 4-class task.** Mode B models projected to mode A differ from their mode A twins by under 0.3 points on validation (HMM +0.10, CRF basic −0.24, CRF improved +0.25) ([EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §11).
- **Leakage inflates lexical models.** On test utterances from videos in the training set, the improved CRF scores 67.09 against 49.50 on the leak-free test, while the HMM scores 33.54 against 34.74. See [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §8.
- **Joint vs punctuation-only (H1, validation, same 19,936 evaluation IDs):** every joint model gains
  on non-final positions (HMM +8.69, CRF basic +1.59, CRF improved +1.16; all intervals exclude 0), but
  PERIOD F1 does not rise consistently, so H1 as stated is not supported. A controlled check retrains the
  mode A HMM and CRF basic on exactly the mode C training IDs and gives the same picture: non-final
  +8.68 (HMM) and +1.36 (CRF basic) ([EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md), H1).

## Quickstart

Python 3.11 or newer, run from the repository root. Versions actually used are recorded in
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

The tests use small fixtures and download nothing. With a trained model (an HMM or hidden-event LM
`.json`, or a CRF or gap-classifier `.joblib`), restore one sentence:

```powershell
python scripts/run_demo.py --model-path <model.json or model.joblib> --text "hello everyone welcome to our presentation"
```

Every command, from rebuilding the data to scoring a model, is in
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Data access

- **Upstream:** SponSpeech (Liu & Beigi, 2024, [arXiv:2409.11241](https://arxiv.org/abs/2409.11241)),
  CC BY-NC 4.0. No dataset files are committed apart from a 10-utterance sample (`data/sample/`) and small release metadata (`data/release_metadata/`); `results/scores/*.errors.json` contain short SponSpeech excerpts as error examples (CC BY-NC 4.0, attribution in [DATA_LICENSE.md](docs/DATA_LICENSE.md)).
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
| `data/` | Local data goes in the ignored folders raw/, processed/ and splits/ under `data/`, created when a release is built or unpacked. Committed: a 10-utterance sample in `data/sample/` and release metadata in `data/release_metadata/`. `data/README.md` explains how to rebuild or download a release. |
| `docs/` | Contracts, dataset documentation, decisions, evaluation protocol and results; listed below. Index: `docs/README.md`. |
| `experiments/` | Model notes and runbooks: `experiments/HMM_NOTES.md`, `experiments/HMM_RUNBOOK.md`, `experiments/HMM_RESULTS.md`, `experiments/CRF_NOTES.md`, `experiments/FEATURE_NOTES.md`. Local run outputs go to the ignored folder experiments/runs/ (create it with `mkdir -p`). |
| `presentation/` | Presentation workspace (`presentation/README.md`); the outline is `docs/PRESENTATION_OUTLINE.md`. |
| `results/` | `results/scores/` holds the committed evaluator JSON behind every reported number (`results/scores/README.md`), including Member 3's six CRF training manifests in `results/scores/crf_training_manifests/`. Local evaluator outputs go to the ignored folders results/predictions/ and results/tables/; create them with `mkdir -p` first, because the evaluator never creates folders. |
| `scripts/` | Command-line entry points; see the table below and `scripts/README.md`. |
| `src/punct_restore/` | The Python package; every module is listed in the package table below. |
| `tests/` | The pytest suite for data, models, features, CLIs and evaluation; fixtures only. Groups are listed in `tests/README.md`; the comparison-method and robustness tests are `tests/test_gap_classifier.py`, `tests/test_hidden_event_lm.py`, `tests/test_train_hidden_event_lm_cli.py`, `tests/test_factored.py`, `tests/test_factor_predictions.py`, `tests/test_crf_case_mode.py`, `tests/test_threshold_priors.py`, `tests/test_asr_noise.py`, `tests/test_simulate_asr_noise.py`, `tests/test_asr_noise_table.py`, `tests/test_asr_noise_findings.py` (with the fixture module `tests/asr_noise_table_fixtures.py`), `tests/test_prediction_io_helpers.py`, `tests/test_gold_noise.py`, `tests/test_find_gold_noise.py`, `tests/test_threshold_provenance.py`, `tests/test_crf_convergence.py`, `tests/test_length_position.py`, `tests/test_word_clusters.py`, `tests/test_build_word_clusters.py`, `tests/test_crf_word_clusters.py`. |

**Evaluator outputs** (`results/scores/`, one folder per study; details in `results/scores/README.md`):

| Folder | Content |
| --- | --- |
| `results/scores/ablations/` | One-feature-group-removed ablations of the improved CRF, per mode, on validation (50k training sample); the gap classifier, question-lookback and factored runs on the same sample are in `results/scores/gap/`, `results/scores/question2/` and `results/scores/factored/` |
| `results/scores/b_to_a/` | Mode B models projected to mode A (D5 mapping) vs their mode A twins, mode A validation gold |
| `results/scores/crf_convergence/` | CRF training loss traces of the six delivered models: iterations, final loss, drop over the last 10 iterations |
| `results/scores/crf_test/` | The six CRFs on test and test_amb, each predicted once; paired comparisons; error pairs |
| `results/scores/crf_training_manifests/` | Member 3's six CRF training manifests: settings, data hashes, fit time, peak memory, environment |
| `results/scores/crf_validation/` | The six CRFs on validation (model selection); paired comparisons; error pairs |
| `results/scores/filler/` | Improved CRF with and without the filler-word cue, per mode, on validation |
| `results/scores/gold_noise/` | Mode B rescored without the known noisy records; the flagged IDs and rules (`*_B_noisy.tsv`) and counts (`*_B_summary.json`, also for test_amb and `default_train`) from `scripts/find_gold_noise.py` |
| `results/scores/h1_joint/` | Joint mode C vs punctuation-only mode A on the mode C validation IDs (H1) |
| `results/scores/hmm/` | HMM and shortcut baseline on validation, test and test_amb; HMM error pairs; `length_position_A_test.json` (mode A micro F1 by length and QUESTION recall by position) |
| `results/scores/leakage/` | HMM and improved CRF on the official mode A test, split by exclusion cause |
| `results/scores/per_class/` | Paired per-class F1 intervals behind hypotheses H2–H4 |
| `results/scores/runtime/` | HMM and CRF prediction timing (median of 3 passes, same laptop) |
| `results/scores/sensitivity/` | Mode A CRF basic and improved at other c1/c2 and iteration settings, on validation |
| `results/scores/asr_noise/` | Frozen HMM and CRFs on test inputs with simulated ASR word errors (one folder per WER), with the corruption summary |
| `results/scores/asr_noise_v2/` | Frozen HMM and CRFs on test inputs with `v2` simulated ASR word errors across seeds 2026–2028 at WER 5%, 10% and 15%, `v1` seed variance at WER 15%, and `summary.json` |
| `results/scores/threshold/` | Plain threshold decoding (τ): validation grids, test and test_amb scores, paired comparisons against the improved CRF |
| `results/scores/threshold_prior/` | Prior-adjusted threshold decoding (τ × γ): validation grids, test and test_amb, paired comparisons |
| `results/scores/gap/` | Gap classifier vs the improved CRF on validation (the CRF without transitions) |
| `results/scores/hidden_event_lm/` | Hidden-event LM: validation grid, test and test_amb, comparisons against the HMM and the improved CRF |
| `results/scores/question2/` | Question-lookback features vs the improved CRF on validation |
| `results/scores/factored/` | Factored vs joint punctuation and casing, mode C validation |
| `results/scores/clusters/` | Word-cluster features vs the improved CRF on validation (50k sample), K = 100 and 300, with the K choice in `clusters_summary.json` |

**Package** (`src/punct_restore/`; each package folder `src/punct_restore/`, `src/punct_restore/data/`,
`src/punct_restore/models/`, `src/punct_restore/features/` and `src/punct_restore/evaluation/` has a
package initializer, `<folder>/__init__.py`):

| Module | Role |
| --- | --- |
| `src/punct_restore/data/` | Member 1's data pipeline: `extraction` (raw transcripts from the official archive), `preprocessing` (loading, cleaning, tokenization), `labeling` (punctuation and casing labels), `splits` (split assignment and leakage audits), `release` (versioned release build and load), `validation` (checks against the data contract), `asr_noise` (simulated ASR word errors with labels and masks kept aligned), `gold_noise` (rules that flag speaker labels, cue residue and auto-caption videos in the gold). |
| `src/punct_restore/models/hmm.py` | Supervised HMM with Viterbi decoding. |
| `src/punct_restore/models/hmm_workflow.py` | Validated file selection and provenance for the HMM commands. |
| `src/punct_restore/models/crf.py` | Linear-chain CRF: fit, predict, marginals, save and load. |
| `src/punct_restore/models/threshold.py` | Threshold decoding of CRF marginals (τ, and γ for the prior-adjusted variant, tuned on validation only). |
| `src/punct_restore/models/gap_classifier.py` | Gap classifier: the improved CRF features with a per-token logistic regression and no label transitions (`gap_improved`). |
| `src/punct_restore/models/hidden_event_lm.py` | Hidden-event n-gram language model (Stolcke style): Viterbi over word and punctuation-event tokens; mode C adds a unigram truecaser. |
| `src/punct_restore/models/ngram_lm.py` | N-gram counting and interpolated absolute-discount probabilities for the hidden-event LM. |
| `src/punct_restore/models/truecaser.py` | Unigram truecaser: the most frequent casing per word, with a sentence-start rule for unseen words. |
| `src/punct_restore/models/factored.py` | Factored casing pipeline: joins mode A punctuation and CASE predictions into mode C labels. |
| `src/punct_restore/features/crf_features.py` | CRF features: basic and improved presets, ablation switches, the filler cue and the opt-in word-cluster features (`use_word_clusters` / `word_clusters`: `cl0`, `cl-1`, `cl+1`, `cl0|cl+1`; the map is stored in the CRF bundle). |
| `src/punct_restore/features/word_clusters.py` | Word clusters induced from `default_train` only: lowercased words with count ≥ 5, ±2 co-occurrence matrix, PPMI, TruncatedSVD (about 100 dimensions), L2-normalised rows, k-means (k ∈ {100, 300}, fixed seed); unknown words get `unk`. The JSON map records the train-ids sha256, k, the seed and the vocabulary size. |
| `src/punct_restore/evaluation/` | Member 5's evaluator: `inputs` (input checks), `views` (scored positions and report blocks), `metrics` (token-level P/R/F1), `uncertainty` (bootstrap grouped by video), `comparison` (paired comparison), `error_analysis` (error examples), `baseline` (shortcut baseline), `manifest` (prediction-manifest check), `report` (JSON and Markdown reports). |
| `src/punct_restore/prediction_io.py` | Gold-free prediction JSONL and manifest writing, shared by `scripts/train_crf.py`, `scripts/tune_threshold.py` and `scripts/project_joint.py`. |
| `src/punct_restore/pipeline.py` | Demo inference: load a trained HMM, hidden-event LM, CRF or gap classifier (`load_model`), clean raw text, predict labels, rebuild text. |

Top-level files: `LICENSE` (MIT), `CONTRIBUTING.md` (daily Git checklist), `TEAM_TASKS.md`
(work board), `pyproject.toml` and `requirements.txt` (package and dependencies),
`requirements-lock.txt` (exact `pip freeze` of the tested environment, CPython 3.13.1),
`requirements-hmm.txt` (the pinned HMM reproduction environment), `.gitignore`, `.gitattributes`.

**Scripts** (run from the repository root; each has `--help` except the two reporting helper modules):

| Script | Stage | What it does |
| --- | --- | --- |
| `scripts/prepare_data.py` | Data | `extract` recreates the raw transcripts from pinned archive ranges; `build` writes a validated, immutable release. |
| `scripts/train_hmm.py` | HMM | Trains the HMM on an explicit ID list; writes a JSON artifact, a run manifest and optional predictions. |
| `scripts/predict_hmm.py` | HMM | Predicts with a saved HMM artifact, without training. |
| `scripts/run_hmm_experiments.py` | HMM | `tune` runs the frozen validation search; `finalize` predicts and scores the selected models. |
| `scripts/verify_hmm_experiments.py` | HMM | Audits an HMM study: recounts scores, refits the winners and compares predictions. |
| `scripts/package_hmm_handoff.py` | HMM | Packages a verified HMM study as a checksummed ZIP. |
| `scripts/train_crf.py` | CRF | `train` fits a CRF (basic or improved features); `predict` writes predictions, and the prediction manifest with `--manifest-out`; `inspect` prints artifact metadata. |
| `scripts/run_ablations.py` | CRF | Trains the feature-group ablations and writes their validation predictions (never reads test); `improved_plus_clusters` with `--cluster-map` adds the word-cluster features (the map's mode, release and train-ids sha256 must match). |
| `scripts/tune_threshold.py` | CRF | Threshold decoding: `tune` picks τ on validation, `predict` applies the frozen τ. |
| `scripts/evaluate_models.py` | Evaluation | `score`, `compare` (paired), `errors` and `baseline` (the shortcut). |
| `scripts/run_demo.py` | Demo | Restores punctuation and casing for one text with a saved HMM, hidden-event LM, CRF or gap classifier. |
| `scripts/project_joint.py` | Study (H1) | Projects mode C predictions onto mode A punctuation labels, so a joint model can be compared with its mode A twin. |
| `scripts/train_hidden_event_lm.py` | Comparison | `train` fits the hidden-event LM on an explicit ID list; `predict` writes predictions and, with `--manifest-out`, a manifest. |
| `scripts/factor_predictions.py` | Comparison | Combines mode A and CASE predictions into mode C predictions with a manifest (the factored casing pipeline). |
| `scripts/simulate_asr_noise.py` | Robustness | Writes copies of the validation or test inputs with simulated ASR word errors (`--edit-model {v1,v2}`) at a target WER, labels and masks kept aligned. |
| `scripts/build_leakage_subsets.py` | Study (leakage) | Splits an official evaluation file into kept and excluded ID lists by exclusion cause, for the leakage-cost analysis. |
| `scripts/find_gold_noise.py` | Study (gold noise) | Flags speaker labels, subtitle cue residue and auto-caption videos on the primary mode B lists; writes the flagged IDs and summaries. |
| `scripts/build_word_clusters.py` | CRF (features) | Builds a word-cluster map (`--mode {A,B,C} --k K --out MAP.json`) from `default_train` only; the map feeds `run_ablations.py --cluster-map`. |
| `scripts/reporting/asr_noise_table.py` | Reporting | Builds the ASR noise `v2` summary JSON and `docs/ASR_NOISE_V2.md` from `results/scores/`. |
| `scripts/reporting/asr_noise_stats.py`, `asr_noise_report.py`, `asr_noise_findings.py` | Reporting (modules) | Helpers of `asr_noise_table.py`: exact seed statistics and rounding, the report's Markdown, and the findings computed from the summary. Not run on their own. |
| `scripts/reporting/build_m3_tables.py` | Reporting | Rebuilds the CRF result tables for one split (main, paired, views, per class, errors, confusion) from `results/scores/`. |
| `scripts/reporting/build_m3_final_sections.py` | Reporting | Rebuilds the threshold, leakage, test_amb and runtime sections from `results/scores/` and the training manifests. |
| `scripts/reporting/build_ablation_table.py` | Reporting | Rebuilds the feature-ablation tables from `results/scores/ablations/`. |
| `scripts/reporting/time_models.py` | Reporting | Times HMM or CRF prediction (and an HMM fit) with the protocol used in the runtime tables. |
| `scripts/reporting/crf_convergence.py` | Reporting | Summarises the CRF training loss traces of trusted `.joblib` models (iterations, final loss, drop over the last 10 iterations). |
| `scripts/reporting/length_position.py` | Reporting | Mode A micro F1 by utterance length and QUESTION recall by position, per model (EVALUATION_RESULTS.md §6). |
| `scripts/reporting/build_tables.py` | Reporting | Helper module, not run directly: shared helpers for the table builders (half-up percentages, cells, JSON loading). |
| `scripts/reporting/_cli.py` | Reporting | Helper module, not run directly: shared one-line error handling for the reporting scripts. |

**Docs:**

| File | Content |
| --- | --- |
| `configs/README.md`, `data/README.md`, `data/sample/README.md`, `data/release_metadata/README.md`, `experiments/README.md`, `tests/README.md`, `scripts/README.md`, `results/README.md` | Folder guides: what each folder holds and how to use it. |
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
| `docs/HYPOTHESES.md` | The six team hypotheses, written before any CRF result existed (outcomes in the CRF results). |
| `docs/EVALUATION_RESULTS_CRF.md` | CRF test results, validation selection, ablations, threshold decoding, filler cue, leakage, runtime, label granularity, multiple comparisons. |
| `docs/ASR_NOISE_V2.md` | Simulated ASR noise `v2` edit model, multi-seed evaluation tables, `v1` vs `v2` comparison, findings and limits. |
| `docs/CRF_EXPERIMENTS.md` | The six CRF runs: features, settings, hashes, commands. |
| `docs/RELATED_WORK.md` | Prior work, checked online, and how RECAP differs. |
| `docs/REPRODUCIBILITY.md` | Every command, with the environments actually used. |
| `docs/PRESENTATION_OUTLINE.md` | Four-slide outline with verified numbers. |

## Team

| Member | Role | Branch | Contribution file |
| --- | --- | --- | --- |
| 1 Almutesim Alakili (Mutasim) | Dataset and preprocessing | `feat/m1-data-preprocessing` | `contributions/Mutasim_220911336_Sam-Ali23.md` |
| 2 Raşit (lead) | HMM baseline | `feat/m2-hmm-baseline` | `contributions/Rasit_STUDENT_ID_Rxd3.md` |
| 3 Omar Alzyod | CRF model and improved features | `feat/m3-crf-model` | `contributions/Omar_2409015412_Devinci101.md` |
| 4 Mohamed Kamaleldin Eliwa | Feature engineering | `feat/m4-feature-engineering` | `contributions/Mohamed_2309015740_Mokamaleldin.md` |
| 5 Abdlaraouf Alkabir | Evaluation and error analysis | `feat/m5-evaluation` | `contributions/Abdlaraouf_2309015482_afel6.md` |
| 6 Abdullah Gamal (Abdulla) | Integration and presentation | `feat/m6-integration-presentation` | `contributions/Abdulla_2309015819_abdu3061.md` |

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch, commit and pull-request rules.

## Licence

Code and documentation: MIT ([LICENSE](LICENSE)). SponSpeech data and anything derived from
it: CC BY-NC 4.0, non-commercial use only, with the upstream attribution
([DATA_LICENSE.md](docs/DATA_LICENSE.md)).
