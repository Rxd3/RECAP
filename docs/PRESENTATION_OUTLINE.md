# Presentation outline

Owner: Member 6 (Abdulla Jamal). Four slides, about 30 seconds each.

Every number comes from [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md), [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) or the evaluator JSON behind them (Member 5 results bundle).

**Rules for the slides:**
- **Main numbers are test.** The course-baseline settings (HMM smoothing, CRF settings, the plain threshold τ) were fixed or chosen on validation before any test result. The comparison methods in sections 13–14 of EVALUATION_RESULTS_CRF.md (simulated ASR noise, gap classifier, hidden-event LM, prior-adjusted decoding, question lookback, factored casing) were added on 2026-10-30, after the first test report, in response to a review that cited test results. They are exploratory; their settings were still chosen on validation only, and each test list was predicted once. Each frozen model predicted test once.
- **Always show the mode.**
- **Never rank mode B against mode C:** they are different tasks.

## Slide 1 — Task and data (30 s)

- Restore punctuation and casing in lowercased English transcripts, one label per token.
- Mode A: 4 punctuation labels. Mode B: 8. Mode C: 4 × 3 = 12 joint punctuation|casing labels.
- **Data:** SponSpeech transcripts of informal speech (CC BY-NC 4.0). We built:
  - the extended and joint label views;
  - casing masks;
  - leak-free primary splits with no shared videos.
- **Models:** HMM with Viterbi (baseline), linear-chain CRF with basic features, CRF with improved features. Our extensions: threshold decoding and a filler-word cue.

**Say:** "Same tokens, same splits, three models, three label sets."

## Slide 2 — Main result (test, primary macro F1 %, 95% CI) (30 s)

| Mode | Shortcut | HMM | Hidden-event LM | CRF basic | CRF improved | + threshold (τ, γ) |
|---|---|---|---|---|---|---|
| A (4 punct) | 27.48 | 35.12 (32.80–37.31) | 48.38 (46.21–50.62) | 43.65 (41.74–45.59) | 49.92 (47.91–52.01) | **51.34** (49.61–53.06) |
| B (8 punct) | 11.87 | 16.79 (15.28–18.49) | 22.51 (21.08–23.98) | 20.25 (18.87–21.95) | 22.82 (21.42–24.42) | **24.25** (23.06–25.54) |
| C (12 joint) | 18.09 | 37.41 (34.56–40.71) | 41.18 (39.42–42.70) | 43.85 (42.13–45.47) | 47.02 (44.97–49.00) | **49.77** (47.26–51.95) |

**Say:** "The improved CRF beats the HMM by 15 points in A, 6 in B and 10 in C, and beats the basic CRF in every mode. Every paired 95% interval for these model comparisons excludes zero, on test and on the second test set."

- **Exact gaps:**
  - improved − HMM: +14.80 / +6.03 / +9.61;
  - improved − basic: +6.27 / +2.57 / +3.17;
  - prior-adjusted threshold − improved: +1.42 / +1.43 / +2.74 (all survive a Holm correction);
  - hidden-event LM − improved CRF: −1.54 / −0.31 (tie) / −5.84.
- **Footnote:**
  - A/B primary = macro F1 over the non-O punctuation classes.
  - C primary = macro F1 over all 12 joint classes with the case_joint mask.
  - Intervals: grouped bootstrap by video, 1,000 resamples.

## Slide 3 — Is it just the end of the utterance? (test, mode A) (30 s)

Score without each utterance's final token:

| Model | Score |
|---|---|
| Shortcut | 0.00 |
| HMM | 9.13 |
| CRF basic | 28.30 |
| CRF improved | **36.40** |

F1 on non-final positions only:

| Class | HMM | CRF improved |
|---|---|---|
| COMMA | 26.28 | 52.52 |
| QUESTION | 0.20 | 20.96 |
| PERIOD | 0.91 | 35.70 |

**Say:** "The HMM mostly learns 'put a period at the end'. The CRF actually learns commas and questions inside the utterance."

## Slide 4 — What helps, leakage, and the limits (30 s)

Feature ablations on validation, one group removed at a time from the improved CRF. Each cell is the change in primary score, without − with (* = interval excludes 0):

| Removed | A | B | C (joint) |
|---|---|---|---|
| word pairs (n-grams) | −3.59 * | −1.63 * | −1.49 |
| ±2 window | −1.34 * | −0.76 * | −0.53 * |
| question cues | −1.35 * | −0.44 * | −0.40 * |

- **Leakage** (official mode A test):
  - The improved CRF scores 67.09 on utterances from videos in its training set, against 49.50 on the leak-free test (different videos; no interval on the difference).
  - The HMM: 33.54 vs 34.74.
- **Question-lookback features** (validation): +6.11 / +7.12 / +6.01 QUESTION F1 (A / B / C), every interval excludes 0.
- **Separate punctuation + casing models** beat the joint mode C model by +2.08 on the joint metric (validation).
- **Casing** (mode C, masked casing F1, test): HMM 79.81, CRF basic 84.23, CRF improved 85.09.
- **Most common error is a missed comma** (mode A test):
  - 63% of the HMM's 39,586 errors;
  - 45% of the improved CRF's 31,583 errors.

**Say:** "Word pairs, the wider window and the question cues help most. Without our leak-free split, the CRF would look 3 points better than it is. Limits:
- CRF settings were not tuned;
- in mode B, EXCLAMATION, SEMICOLON and DASH stay below 3 F1 and COLON below 8;
- the intervals cover sample variance only."

- **Footnote:** ablations train on a 50k-sequence sample, one fit per row. In mode C, the word-pair drop on the joint metric is not significant.

## Backup (only if asked)

- **Does predicting casing help punctuation?**
  - Setting: the same 19,936 validation sequences, non-final positions, mode A model vs mode C (joint) model.
  - HMM 8.64 vs 17.33, CRF basic 27.93 vs 29.52, CRF improved 37.00 vs 38.17. All three intervals exclude 0.
  - PERIOD F1 does not move consistently, so H1 as stated is not supported.
- **Are the CRF numbers reproducible?** Yes:
  - all six validation prediction files were regenerated byte for byte from the shared models;
  - every test file passes the evaluator's manifest check.
- **Runtime** (same laptop): HMM 46k–211k tokens/s; CRF basic 100k–112k; CRF improved 47k–50k.
