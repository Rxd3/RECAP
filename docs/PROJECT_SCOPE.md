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
Validation chose the HMM smoothing and the threshold τ; the CRF settings (c1 = c2 = 0.1, 100 iterations) were fixed and not tuned; no test result influenced any choice. The team's additions beyond the course baselines are listed in the
[README](../README.md#what-is-new).

## Out of scope

- Audio, prosody and speech-recognition (ASR) output: RECAP uses the transcripts only, and no ASR
  condition was evaluated.
- Neural models (for example BiLSTM-CRF or transformers), GPU frameworks and pretrained-model
  downloads.
- An interactive application beyond the command-line demo (`scripts/run_demo.py`).

## Evidence rules

Every reported number comes from an evaluator output file under `results/scores/`. Validation is
used for selection; each frozen model predicts the test lists once. Limitations and negative
findings are reported with the results.
