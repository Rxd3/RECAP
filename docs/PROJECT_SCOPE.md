# Project scope

RECAP — Restoring English Casing and Punctuation.
COE025 Group Project 2: Extended Punctuation and Truecasing Restoration Using
Sequence Labeling. Six members; Python; one repository.

## Deliverable

Restore the punctuation mark after each English word and the casing of each word, on SponSpeech
transcripts, in three label modes: A (4 punctuation labels), B (8) and C (4 punctuation × 3 casing
= 12 joint labels). Compare the three models the course asks for on fixed, shared splits: an HMM
with Viterbi decoding, a linear-chain CRF with basic features, and the same CRF with improved
features.

All three are trained and evaluated. Results: [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) (HMM and
shortcut) and [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) (CRF, final test results).
The course-baseline settings (HMM smoothing, CRF settings, the plain threshold τ) were fixed or chosen on validation before any test result. The comparison methods in sections 13–14 of [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) (simulated ASR noise, gap classifier, hidden-event LM, prior-adjusted decoding, question lookback, factored casing) were added on 2026-10-30, after the first test report, in response to a review that cited test results. They are exploratory; their settings were still chosen on validation only, and each test list was predicted once. The team's additions beyond the course baselines are listed in the
[README](../README.md#what-is-new).

## Out of scope

- Audio, prosody and real speech-recognition (ASR) output: RECAP uses the transcripts only, and there is no
  real-ASR condition (a simulated ASR-noise check is in [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §13).
- Neural models (for example BiLSTM-CRF or transformers), GPU frameworks and pretrained-model
  downloads.
- An interactive application beyond the command-line demo (`scripts/run_demo.py`).

## Evidence rules

Every reported number comes from an evaluator output file under `results/scores/`. Validation is
used for selection; each frozen model predicts the test lists once. Limitations and negative
findings are reported with the results.
