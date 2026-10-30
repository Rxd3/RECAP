# Presentation outline

Owner: Member 6 (Abdulla Jamal). Four slides, about 30 seconds each.

Every number comes from Member 5's verified slide sheet, which is built from
[EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) and [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md).
**All CRF numbers are validation.** Label them "validation" on the slide, do not call them final,
and swap in the test column when the CRF test predictions are scored. Always show the mode, and
never rank mode B against mode C: they are different tasks.

## Slide 1 — Task and data (30 s)

- Restore punctuation and casing in lowercased English transcripts, one label per token.
- Mode A: 4 punctuation labels. Mode B: 8. Mode C: 4 × 3 = 12 joint punctuation|casing labels.
- Data: SponSpeech transcripts of informal speech (CC BY-NC 4.0). We built the extended and joint
  label views, casing masks and leak-free primary splits with no shared videos.
- Models: HMM with Viterbi (baseline), linear-chain CRF with basic features, CRF with improved features.

**Say:** "Same tokens, same splits, three models, three label sets."

## Slide 2 — Main result (validation, primary macro F1 %, 95% CI) (30 s)

| Mode | Shortcut | HMM | CRF basic | CRF improved |
|---|---|---|---|---|
| A (4 punct) | 28.21 | 36.54 (34.80–38.01) | 44.54 (42.80–45.90) | **51.18** (49.60–52.49) |
| B (8 punct) | 12.14 | 18.47 (16.32–21.35) | 22.22 (19.79–24.98) | **25.10** (23.08–27.68) |
| C (12 joint) | 18.53 | 40.89 (38.72–42.93) | 47.72 (45.73–49.72) | **51.52** (49.14–53.56) |

**Say:** "The improved CRF beats the HMM by 15 points in A, 7 in B and 11 in C, and beats the basic
CRF in every mode. Every paired 95% interval excludes zero."

- Exact gaps, improved − HMM: +14.64 / +6.63 / +10.64. Improved − basic: +6.64 / +2.88 / +3.80.
- Footnote: A/B primary = macro F1 over the non-O punctuation classes; C primary = macro F1 over
  all 12 joint classes with the case_joint mask. Intervals: grouped bootstrap by video, 1,000 resamples.

## Slide 3 — Is it just the end of the utterance? (validation, mode A) (30 s)

Score without each utterance's final token:

| Model | Score |
|---|---|
| Shortcut | 0.00 |
| HMM | 8.63 |
| CRF basic | 27.92 |
| CRF improved | **36.99** |

F1 on non-final positions only:

| Class | HMM | CRF improved |
|---|---|---|
| COMMA | 24.67 | 51.71 |
| QUESTION | 0.35 | 24.11 |
| PERIOD | 0.88 | 35.16 |

**Say:** "The HMM mostly learns 'put a period at the end'. The CRF actually learns commas and
questions inside the utterance."

## Slide 4 — What helps, and the limits (validation) (30 s)

Feature ablations, one group removed at a time from the improved CRF (change in primary score,
without − with; * = interval excludes 0):

| Removed | A | B | C (joint) |
|---|---|---|---|
| word pairs (n-grams) | −3.59 * | −1.63 * | −1.49 |
| ±2 window | −1.34 * | −0.76 * | −0.53 * |
| question cues | −1.35 * | −0.44 * | −0.40 * |

- Casing (mode C, masked casing F1): HMM 84.31, CRF basic 86.62, CRF improved 87.32.
- Most common error for every model is a missed comma: 63% of the HMM's 59,145 errors and 45% of
  the improved CRF's 47,243 errors (mode A).

**Say:** "Word pairs, the wider window and the question cues help most. The main limits: these are
validation numbers until the CRF test predictions are scored, CRF settings were not tuned
(c1 = c2 = 0.1, 100 iterations), EXCLAMATION, SEMICOLON and DASH stay below 4 F1 in mode B for
every model, and the intervals cover sample variance only."

- Footnote: ablations train on a 50k-sequence sample, one fit per row. In mode C the word-pair drop
  on the joint metric is not significant.

## Backup (only if asked)

- **Does the CRF result hold on test?** Member 5's independent reference CRF (a different
  implementation, not Omar's model) scores on test: A 53.36, B 24.41, C 50.67, against the HMM's
  35.12, 16.79 and 37.41. It supports the direction (CRF > HMM on test), not Omar's exact numbers.
- **Does predicting casing help punctuation?** On the same 19,936 validation sequences, non-final
  positions, mode A vs mode C (joint) model: HMM 8.64 vs 17.33, CRF basic 27.93 vs 29.52, CRF
  improved 37.00 vs 38.17; all three intervals exclude 0. PERIOD F1 does not move consistently, so
  H1 as stated is not supported.
