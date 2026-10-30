# Evaluation results: CRF models (Member 3), validation

Member 5's evaluation of Member 3's six CRF models (modes A, B and C × basic and improved). It follows [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) and complements the HMM report in [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md).

> **Validation only.** Member 3 delivered validation predictions. Test and test_amb results will be added once the frozen models predict them. Until then, these numbers are for model selection and must not be quoted as final test results.

## Provenance

- **Models:**
  - Code: branch `feat/m3-crf-model`, commit `aa232be` (Member 3).
  - Engine: `sklearn-crfsuite` 0.5.0, L-BFGS.
  - Settings: c1 = c2 = 0.1, 100 iterations (fixed defaults, not tuned). Every fit hit the iteration cap.
  - Training data: the official `default_train` lists (A/B 142,722 sequences, C 141,745) and the official `train.jsonl` files; the manifests' input and ID hashes match the release.
- **Predictions:**
  - Handoff `recap_m3_handoff_eval.zip` (sha256 `0ceb81d6…`); all 12 entries of its `SHA256SUMS` verified.
  - Every primary validation ID present once, tokens aligned, every label in the mode vocabulary.
- **Reproduction:**
  - Member 5 retrained mode A basic from a clean `aa232be` checkout and predicted with the committed `scripts/train_crf.py predict`. The output is byte-identical to the delivered file (sha256 `cffd55c6…`).
  - The improved models were not re-run, because each needs about 14.5 GB of memory.
  - The delivered manifests record a code fingerprint that differs from `aa232be`, so the training working copy differed in at least one fingerprinted file. That difference does not change the mode A basic predictions; for the improved models it is an open question for Member 3.
- **Manifest check:** the delivered manifests are training manifests, so `score --manifest` cannot run (`missing field(s) ordered_labels, predictions_sha256, input`). Files were checked against `SHA256SUMS` instead.
- **Scoring:**
  - `scripts/evaluate_models.py` (this branch), on the primary validation lists, with the `case_joint` mask in mode C;
  - grouped bootstrap by video, 1,000 resamples, seed 2026;
  - 95% percentile intervals, not adjusted for multiple comparisons.
  - An independent from-scratch scorer gives identical values for every block.

**Metric:**
- A/B primary: macro F1 over the non-O punctuation classes.
- C primary: macro F1 over all 12 joint `PUNCT|CASE` classes on unmasked positions. Mode C micro P/R/F1 equals accuracy, because the micro average runs over all 12 classes.

## Main findings (validation)

1. **Both CRFs beat the HMM in every mode, and improved beats basic in every mode.** All nine paired intervals exclude 0.

   | Mode | Improved − basic | Improved − HMM |
   |---|---|---|
   | A | +6.64 | +14.64 |
   | B | +2.88 | +6.63 |
   | C | +3.80 | +10.64 |

2. **The CRFs learn punctuation inside the utterance.** Without the utterance-final token, mode A scores HMM 8.63, CRF basic 27.92, CRF improved 36.99.
3. **The gain is mostly COMMA and QUESTION** (mode A F1):

   | Class | HMM | CRF improved |
   |---|---|---|
   | COMMA | 24.67 | 51.71 |
   | QUESTION | 15.54 | 29.45 |
   | PERIOD | 69.42 | 72.39 |

   The CRFs trade some precision for much higher recall: mode A micro recall rises from 31.78 to 53.28.
4. **The most frequent error is still a missed comma** (COMMA → O): 42–45% of the improved model's punctuation errors, depending on the mode (mode C projected to punctuation).
5. **Mode B's rare classes stay low** for every model. Validation COLON support includes 26 false gold labels from leftover speaker tags (6.1%).
6. **Casing (mode C, masked):** CRF improved 87.32, and 84.53 without the first word. The HMM scores 84.31 and 80.67.
7. **Ablations (section 5):**
   - On the primary metric, removing word pairs, the ±2 window or the question cues costs points in modes A and B (all intervals exclude 0); in mode C, the window and question cues do.
   - Clause cues, casing cues and position buckets have small effects of mixed sign.

## Team hypotheses (validation)

| # | Hypothesis | Outcome | Evidence |
|---|---|---|---|
| H1 | Joint mode C gives higher PERIOD F1 than mode A | **Not supported for PERIOD**, but joint training helps inside the utterance | Paired comparison on mode A gold over the 19,936 mode C validation IDs: each mode A model against the same model type trained on mode C, projected to punctuation. PERIOD F1, A vs C: HMM 69.31 vs 68.33, CRF basic 71.57 vs 71.43, CRF improved 72.33 vs 72.63. The joint CRF improved gains +0.42 macro (0.05 to 0.82), mostly on QUESTION (+1.51). Every joint model gains on non-final positions, and all three intervals exclude 0: HMM +8.69, CRF basic +1.59, CRF improved +1.16. Overall, the joint HMM loses 0.62 (−1.20 to −0.03). |
| H2 | Threshold decoding raises rare-class recall | **Not tested** | No model implements threshold decoding. |
| H3 | Question cues raise QUESTION F1 | **Supported** | Removing the question cues lowers QUESTION F1 by 4.05 (A), 2.85 (B) and 4.93 (C, punctuation projection). Primary intervals exclude 0 in every mode (section 5). |
| H4 | Filler and discourse cues raise COMMA F1 | **Not supported** | Removing the clause cues (conjunctions and discourse markers, including filler-like words such as *well*, *actually*, *anyway*) changes COMMA F1 by +0.03 (A), +0.05 (B) and −0.20 (C). There is no dedicated filler feature (*um*, *uh*). |
| H5 | Much of PERIOD F1 comes from the utterance-final token | **Supported** | The shortcut alone reaches PERIOD F1 68.80 in mode A, against the HMM's 69.42. Without the final token, the HMM's primary score falls from 36.54 to 8.63. |
| H6 | The CRF beats the HMM in every mode | **Supported** | CRF basic − HMM and CRF improved − HMM: all six paired intervals exclude 0 (section 2). |

H3 and H4 come from the 50k-sample ablation models, the others from the delivered models. Per-class changes have no interval.

## 1. Main result (validation, primary ID lists)

Primary = macro F1 % with 95% CI (grouped bootstrap by video, 1,000 resamples, seed 2026). Micro P / R / F1 % over the same classes. A/B: non-O punctuation classes; C: all 12 joint classes on unmasked positions (case_joint mask applied).

**Mode A** (`punctuation/all`)

| Model | Macro F1 (95% CI) | Micro P / R / F1 | Accuracy |
|---|---|---|---|
| Shortcut | 28.21 (26.89–29.48) | 90.12 / 22.55 / 36.08 | 89.98 |
| HMM (M2) | 36.54 (34.80–38.01) | 77.74 / 31.78 / 45.12 | 90.44 |
| CRF basic (M3) | 44.54 (42.80–45.90) | 70.08 / 45.15 / 54.92 | 91.67 |
| CRF improved (M3) | 51.18 (49.60–52.49) | 68.38 / 53.28 / 59.89 | 92.37 |

**Mode B** (`punctuation/all`)

| Model | Macro F1 (95% CI) | Micro P / R / F1 | Accuracy |
|---|---|---|---|
| Shortcut | 12.14 (11.58–12.68) | 90.12 / 22.55 / 36.08 | 89.98 |
| HMM (M2) | 18.47 (16.32–21.35) | 68.14 / 31.56 / 43.14 | 89.89 |
| CRF basic (M3) | 22.22 (19.79–24.98) | 69.64 / 44.46 / 54.27 | 91.61 |
| CRF improved (M3) | 25.10 (23.08–27.68) | 67.84 / 52.40 / 59.13 | 92.29 |

**Mode C** (`joint/unmasked`)

| Model | Macro F1 (95% CI) | Micro P / R / F1 | Accuracy |
|---|---|---|---|
| Shortcut | 18.53 (18.05–18.98) | 83.67 / 83.67 / 83.67 | 83.67 |
| HMM (M2) | 40.89 (38.72–42.93) | 86.97 / 86.97 / 86.97 | 86.97 |
| CRF basic (M3) | 47.72 (45.73–49.72) | 88.50 / 88.50 / 88.50 | 88.50 |
| CRF improved (M3) | 51.52 (49.14–53.56) | 89.25 / 89.25 / 89.25 | 89.25 |

## 2. Paired differences (validation, points, 95% CI; * = interval excludes 0)

| Mode | Comparison | Primary | non_final |
|---|---|---|---|
| A | M3 improved − M3 basic | +6.64 (5.89 to 7.32) * | +9.07 (8.13 to 9.95) * |
| A | M3 basic − HMM | +8.00 (7.20 to 8.82) * | +19.29 (18.17 to 20.39) * |
| A | M3 improved − HMM | +14.64 (13.80 to 15.62) * | +28.36 (27.20 to 29.37) * |
| B | M3 improved − M3 basic | +2.88 (1.78 to 3.77) * | +3.95 (2.83 to 4.85) * |
| B | M3 basic − HMM | +3.75 (2.22 to 5.69) * | +8.66 (7.02 to 10.63) * |
| B | M3 improved − HMM | +6.63 (5.21 to 7.99) * | +12.61 (11.12 to 14.02) * |
| C | M3 improved − M3 basic | +3.80 (2.34 to 5.53) * | +8.65 (7.70 to 9.72) * |
| C | M3 basic − HMM | +6.83 (5.22 to 8.44) * | +12.19 (11.05 to 13.41) * |
| C | M3 improved − HMM | +10.64 (8.91 to 12.86) * | +20.84 (19.59 to 22.09) * |

## 3. Views and mode C sub-blocks (validation, macro F1 %, 95% CI)

`non_final` drops each utterance's last token. Mode C casing uses the case_joint mask; `casing/all` is unmasked, for reference only.

| Model | A non_final | B non_final | C non_final | C punctuation/all | C casing (masked) | C casing non-initial | C casing/all |
|---|---|---|---|---|---|---|---|
| Shortcut | 0.00 (0.00–0.00) | 0.00 (0.00–0.00) | 0.00 (0.00–0.00) | 28.18 (26.84–29.44) | 47.90 (47.15–48.61) | 32.03 (31.95–32.10) | 47.74 (46.99–48.46) |
| HMM (M2) | 8.63 (7.70–9.54) | 6.44 (4.34–9.20) | 17.33 (16.00–18.54) | 35.87 (34.11–37.28) | 84.31 (82.08–86.03) | 80.67 (78.45–82.31) | 84.23 (81.98–85.95) |
| CRF basic (M3) | 27.92 (26.31–29.22) | 15.10 (12.67–17.82) | 29.52 (27.89–30.85) | 44.89 (43.08–46.28) | 86.62 (84.79–88.11) | 83.62 (81.81–85.07) | 86.50 (84.64–87.97) |
| CRF improved (M3) | 36.99 (35.31–38.29) | 19.04 (16.99–21.54) | 38.17 (36.52–39.48) | 51.59 (50.03–52.84) | 87.32 (85.44–88.69) | 84.53 (82.81–85.84) | 87.20 (85.31–88.57) |

## 4. Per-class precision / recall / F1 % (validation)

Classes marked O are not in the A/B primary average; in mode C all 12 joint classes are.

**Mode A**

| Class | Support | Shortcut P / R / F1 | HMM (M2) P / R / F1 | CRF basic (M3) P / R / F1 | CRF improved (M3) P / R / F1 |
|---|---|---|---|---|---|
| O | 538,697 | 89.97 / 100.00 / 94.72 | 91.15 / 99.16 / 94.99 | 93.64 / 98.59 / 96.05 | 95.05 / 98.18 / 96.59 |
| COMMA | 44,279 | 0.00 / 0.00 / 0.00 | 56.96 / 15.74 / 24.67 | 60.34 / 32.91 / 42.59 | 61.58 / 44.56 / 51.71 |
| PERIOD | 32,392 | 92.70 / 54.70 / 68.80 | 91.03 / 56.10 / 69.42 | 79.21 / 65.41 / 71.65 | 76.55 / 68.65 / 72.39 |
| QUESTION | 3,403 | 36.96 / 10.08 / 15.84 | 57.20 / 8.99 / 15.54 | 57.45 / 11.67 / 19.39 | 53.35 / 20.34 / 29.45 |

**Mode B**

| Class | Support | Shortcut P / R / F1 | HMM (M2) P / R / F1 | CRF basic (M3) P / R / F1 | CRF improved (M3) P / R / F1 |
|---|---|---|---|---|---|
| O | 538,697 | 89.97 / 100.00 / 94.72 | 91.28 / 98.56 / 94.78 | 93.59 / 98.62 / 96.04 | 95.01 / 98.23 / 96.59 |
| COMMA | 42,346 | 0.00 / 0.00 / 0.00 | 57.27 / 15.83 / 24.81 | 59.69 / 33.14 / 42.62 | 60.61 / 44.82 / 51.53 |
| PERIOD | 32,132 | 92.70 / 55.14 / 69.15 | 91.07 / 56.53 / 69.76 | 78.91 / 65.74 / 71.73 | 76.57 / 69.09 / 72.64 |
| QUESTION | 3,403 | 36.96 / 10.08 / 15.84 | 58.29 / 8.99 / 15.58 | 56.69 / 11.46 / 19.07 | 53.86 / 21.13 / 30.35 |
| EXCLAMATION | 156 | 0.00 / 0.00 / 0.00 | 1.39 / 4.49 / 2.12 | 15.38 / 1.28 / 2.37 | 13.64 / 1.92 / 3.37 |
| COLON | 425 | 0.00 / 0.00 / 0.00 | 62.71 / 8.71 / 15.29 | 59.52 / 11.76 / 19.65 | 54.17 / 9.18 / 15.69 |
| SEMICOLON | 104 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| DASH | 1,508 | 0.00 / 0.00 / 0.00 | 1.18 / 3.32 / 1.74 | 1.89 / 0.07 / 0.13 | 16.83 / 1.13 / 2.11 |

**Mode C joint (masked)**

| Class | Support | Shortcut P / R / F1 | HMM (M2) P / R / F1 | CRF basic (M3) P / R / F1 | CRF improved (M3) P / R / F1 |
|---|---|---|---|---|---|
| O\|LOWER | 481,731 | 83.57 / 99.97 / 91.03 | 88.83 / 98.19 / 93.28 | 91.22 / 97.71 / 94.36 | 92.74 / 97.16 / 94.90 |
| O\|FIRST_UPPER | 53,395 | 86.44 / 32.24 / 46.97 | 84.26 / 68.36 / 75.48 | 84.84 / 72.81 / 78.36 | 83.93 / 75.02 / 79.23 |
| O\|ALL_UPPER | 1,399 | 0.00 / 0.00 / 0.00 | 73.51 / 66.05 / 69.58 | 81.59 / 71.27 / 76.08 | 80.62 / 72.27 / 76.22 |
| COMMA\|LOWER | 37,292 | 0.00 / 0.00 / 0.00 | 54.25 / 10.82 / 18.04 | 56.33 / 30.18 / 39.30 | 59.10 / 41.97 / 49.09 |
| COMMA\|FIRST_UPPER | 6,273 | 0.00 / 0.00 / 0.00 | 46.63 / 20.63 / 28.60 | 53.51 / 32.44 / 40.39 | 55.28 / 36.55 / 44.01 |
| COMMA\|ALL_UPPER | 383 | 0.00 / 0.00 / 0.00 | 52.75 / 12.53 / 20.25 | 52.50 / 32.90 / 40.45 | 63.18 / 33.16 / 43.49 |
| PERIOD\|LOWER | 29,349 | 86.10 / 55.58 / 67.55 | 73.53 / 63.64 / 68.23 | 75.40 / 67.21 / 71.07 | 73.86 / 70.72 / 72.26 |
| PERIOD\|FIRST_UPPER | 2,584 | 100.00 / 0.19 / 0.39 | 37.94 / 34.09 / 35.92 | 52.92 / 37.58 / 43.95 | 55.00 / 40.21 / 46.46 |
| PERIOD\|ALL_UPPER | 258 | 0.00 / 0.00 / 0.00 | 73.79 / 41.47 / 53.10 | 67.63 / 45.35 / 54.29 | 67.03 / 48.06 / 55.98 |
| QUESTION\|LOWER | 3,033 | 35.40 / 10.72 / 16.45 | 37.26 / 10.12 / 15.92 | 47.18 / 12.13 / 19.30 | 49.32 / 21.43 / 29.88 |
| QUESTION\|FIRST_UPPER | 334 | 0.00 / 0.00 / 0.00 | 16.67 / 0.90 / 1.70 | 37.04 / 2.99 / 5.54 | 41.03 / 4.79 / 8.58 |
| QUESTION\|ALL_UPPER | 18 | 0.00 / 0.00 / 0.00 | 100.00 / 5.56 / 10.53 | 33.33 / 5.56 / 9.52 | 50.00 / 11.11 / 18.18 |

**Mode C casing (masked)**

| Class | Support | Shortcut P / R / F1 | HMM (M2) P / R / F1 | CRF basic (M3) P / R / F1 | CRF improved (M3) P / R / F1 |
|---|---|---|---|---|---|
| LOWER | 551,405 | 92.47 / 99.96 / 96.07 | 96.46 / 99.04 / 97.73 | 97.01 / 99.04 / 98.01 | 97.24 / 98.86 / 98.04 |
| FIRST_UPPER | 62,586 | 98.60 / 31.39 / 47.62 | 88.96 / 68.84 / 77.62 | 89.62 / 73.72 / 80.90 | 88.18 / 75.81 / 81.53 |
| ALL_UPPER | 2,058 | 0.00 / 0.00 / 0.00 | 92.24 / 66.96 / 77.59 | 91.33 / 72.69 / 80.95 | 92.76 / 74.10 / 82.39 |

## 5. Feature ablations of the improved CRF (validation)

Member 5 retrained Member 3's improved preset from his committed code (`aa232be`) with one feature group switched off at a time. The settings match the delivered models (c1 = c2 = 0.1, 100 iterations). Unlike them, each ablation model trains on a fixed sample of 50,000 `default_train` sequences per mode, drawn with seed 2026; modes A and B share one sample. All models are scored on the primary validation lists. Test was not used.

| Group | `FeatureConfig` change | What it adds |
|---|---|---|
| window2 | `window_size=1` | all features at distance 2: words, affixes, question and clause cues (overlaps those groups) |
| ngrams | `use_token_ngrams=False` | word pairs `w-1_w0` and `w0_w+1` |
| question | `use_question_cues=False` | wh-words and auxiliary-subject inversion |
| clause | `use_clause_cues=False` | conjunctions and discourse markers |
| casing | `use_casing_cues=False` | lexical capitalisation triggers (I, days and months, titles) |
| position | `use_positional_buckets=False` | position bucket in the utterance (single, start, early, middle, late, end) |

**Findings:**
1. **Word pairs help most.** Without them, mode A drops 3.59 points, mode B 1.63, and mode C punctuation 3.74. The mode C joint drop of 1.49 has an interval that crosses 0, but the mode C punctuation non_final drop of 5.25 does not.
2. **The ±2 window and the question cues also help in every mode.** All three primary intervals exclude 0 for both.
3. **Clause cues, casing cues and position buckets have small effects of mixed sign.**
   - On the primary metric, every effect is under 1 point.
   - In mode B, removing clause cues (+0.17) or position buckets (+0.21) slightly *raises* the score.
   - In mode C, removing casing cues or position buckets slightly lowers punctuation: non_final −0.42 and −0.45, both intervals excluding 0.
4. **The casing cues have no measurable effect on casing.** Mode C masked casing changes by −0.08 (−0.29 to 0.11) without them.

The full-model scores here sit below the delivered full-data models (mode A 49.96 vs 51.18), because the training sample is a third of the data. Read only the differences.

Each row removes one feature group from the improved CRF. Difference = without − with, in points (95% paired CI, grouped bootstrap by video, 1,000 resamples, seed 2026; * = interval excludes 0). Negative means the group helps.

**Mode A**: full improved 49.96 (48.42–51.28) primary, 35.17 (33.52–36.47) non_final

| Without | Primary | Δ primary | non_final | Δ non_final |
|---|---|---|---|---|
| window2 | 48.62 (47.12–49.87) | -1.34 (-1.79 to -0.86) * | 33.27 (31.69–34.53) | -1.90 (-2.59 to -1.25) * |
| ngrams | 46.37 (44.80–47.59) | -3.59 (-4.07 to -3.13) * | 30.85 (29.22–32.08) | -4.32 (-5.03 to -3.64) * |
| question | 48.61 (46.96–49.95) | -1.35 (-1.83 to -0.99) * | 34.31 (32.50–35.74) | -0.86 (-1.39 to -0.45) * |
| clause | 49.89 (48.39–51.17) | -0.07 (-0.26 to 0.11) | 35.15 (33.44–36.48) | -0.03 (-0.35 to 0.24) |
| casing | 49.97 (48.43–51.24) | +0.01 (-0.23 to 0.27) | 34.96 (33.30–36.25) | -0.21 (-0.52 to 0.11) |
| position | 49.76 (48.30–51.06) | -0.20 (-0.43 to 0.05) | 34.99 (33.35–36.24) | -0.19 (-0.52 to 0.16) |

**Mode B**: full improved 23.25 (21.43–25.70) primary, 16.80 (14.91–19.27) non_final

| Without | Primary | Δ primary | non_final | Δ non_final |
|---|---|---|---|---|
| window2 | 22.49 (20.78–24.79) | -0.76 (-1.14 to -0.44) * | 15.86 (14.15–18.21) | -0.94 (-1.38 to -0.57) * |
| ngrams | 21.62 (19.93–23.85) | -1.63 (-2.04 to -1.19) * | 14.95 (13.21–17.16) | -1.85 (-2.29 to -1.41) * |
| question | 22.81 (21.00–25.18) | -0.44 (-0.73 to -0.17) * | 16.67 (14.77–19.09) | -0.13 (-0.44 to 0.15) |
| clause | 23.41 (21.59–25.84) | +0.17 (0.04 to 0.35) * | 16.96 (15.05–19.41) | +0.16 (-0.00 to 0.36) |
| casing | 23.20 (21.46–25.56) | -0.04 (-0.27 to 0.13) | 16.88 (15.07–19.28) | +0.08 (-0.16 to 0.28) |
| position | 23.46 (21.68–25.81) | +0.21 (0.05 to 0.36) * | 17.02 (15.13–19.40) | +0.23 (0.03 to 0.39) * |

**Mode C**: full improved 47.70 (45.92–49.31) primary, 36.57 (34.88–37.95) non_final

| Without | Primary | Δ primary | non_final | Δ non_final |
|---|---|---|---|---|
| window2 | 47.17 (45.37–48.87) | -0.53 (-1.00 to -0.03) * | 35.43 (33.78–36.79) | -1.14 (-1.72 to -0.52) * |
| ngrams | 46.22 (43.86–48.34) | -1.49 (-2.82 to 0.31) | 31.33 (29.79–32.62) | -5.25 (-5.94 to -4.49) * |
| question | 47.30 (45.58–48.85) | -0.40 (-0.72 to -0.04) * | 35.07 (33.11–36.70) | -1.50 (-2.21 to -0.97) * |
| clause | 47.42 (45.68–48.97) | -0.29 (-0.64 to 0.04) | 36.28 (34.56–37.73) | -0.29 (-0.61 to 0.02) |
| casing | 47.62 (45.82–49.26) | -0.08 (-0.38 to 0.25) | 36.15 (34.38–37.62) | -0.42 (-0.70 to -0.16) * |
| position | 48.53 (46.10–50.60) | +0.83 (-0.25 to 2.52) | 36.13 (34.49–37.45) | -0.45 (-0.71 to -0.16) * |

## Most frequent errors, M3 improved (validation, punctuation labels; mode C projected to punctuation)

| Mode | Errors | Top pairs (count, share) |
|---|---|---|
| A | 47,243 | COMMA→O 21,151 (45%); O→COMMA 7,874 (17%); PERIOD→O 5,721 (12%); PERIOD→COMMA 4,064 (9%); COMMA→PERIOD 3,261 (7%) |
| B | 47,679 | COMMA→O 20,086 (42%); O→COMMA 7,588 (16%); PERIOD→O 5,647 (12%); PERIOD→COMMA 3,889 (8%); COMMA→PERIOD 3,084 (6%) |
| C | 46,999 | COMMA→O 21,073 (45%); O→COMMA 7,596 (16%); PERIOD→O 5,385 (11%); PERIOD→COMMA 3,808 (8%); COMMA→PERIOD 3,731 (8%) |

## Confusion matrices, M3 CRFs (validation)

Mode C is shown as its punctuation projection and its masked casing projection; the 12 × 12 joint matrix is in each score JSON (`blocks.joint/unmasked.confusion`).

### Mode A

*CRF basic (M3), punctuation* (rows = gold, columns = predicted)

| gold \ pred | O | COMMA | PERIOD | QUESTION |
|---|---|---|---|---|
| O | 531,096 | 6,090 | 1,483 | 28 |
| COMMA | 27,317 | 14,572 | 2,324 | 66 |
| PERIOD | 7,822 | 3,183 | 21,187 | 200 |
| QUESTION | 946 | 306 | 1,754 | 397 |

*CRF improved (M3), punctuation* (rows = gold, columns = predicted)

| gold \ pred | O | COMMA | PERIOD | QUESTION |
|---|---|---|---|---|
| O | 528,866 | 7,874 | 1,856 | 101 |
| COMMA | 21,151 | 19,732 | 3,261 | 135 |
| PERIOD | 5,721 | 4,064 | 22,238 | 369 |
| QUESTION | 645 | 372 | 1,694 | 692 |

### Mode B

*CRF basic (M3), punctuation* (rows = gold, columns = predicted)

| gold \ pred | O | COMMA | PERIOD | QUESTION | EXCLAMATION | COLON | SEMICOLON | DASH |
|---|---|---|---|---|---|---|---|---|
| O | 531,275 | 5,879 | 1,485 | 31 | 1 | 15 | 1 | 10 |
| COMMA | 25,958 | 14,034 | 2,230 | 66 | 6 | 14 | 1 | 37 |
| PERIOD | 7,764 | 3,031 | 21,124 | 201 | 3 | 4 | 1 | 4 |
| QUESTION | 950 | 290 | 1,771 | 390 | 0 | 1 | 0 | 1 |
| EXCLAMATION | 87 | 31 | 36 | 0 | 2 | 0 | 0 | 0 |
| COLON | 247 | 86 | 42 | 0 | 0 | 50 | 0 | 0 |
| SEMICOLON | 77 | 19 | 8 | 0 | 0 | 0 | 0 | 0 |
| DASH | 1,292 | 140 | 74 | 0 | 1 | 0 | 0 | 1 |

*CRF improved (M3), punctuation* (rows = gold, columns = predicted)

| gold \ pred | O | COMMA | PERIOD | QUESTION | EXCLAMATION | COLON | SEMICOLON | DASH |
|---|---|---|---|---|---|---|---|---|
| O | 529,137 | 7,588 | 1,821 | 100 | 3 | 13 | 3 | 32 |
| COMMA | 20,086 | 18,978 | 3,084 | 138 | 9 | 16 | 1 | 34 |
| PERIOD | 5,647 | 3,889 | 22,199 | 371 | 7 | 2 | 3 | 14 |
| QUESTION | 645 | 362 | 1,672 | 719 | 0 | 1 | 0 | 4 |
| EXCLAMATION | 61 | 54 | 37 | 0 | 3 | 1 | 0 | 0 |
| COLON | 233 | 90 | 63 | 0 | 0 | 39 | 0 | 0 |
| SEMICOLON | 58 | 32 | 13 | 1 | 0 | 0 | 0 | 0 |
| DASH | 1,063 | 319 | 103 | 6 | 0 | 0 | 0 | 17 |

### Mode C

*CRF basic (M3), punctuation* (rows = gold, columns = predicted)

| gold \ pred | O | COMMA | PERIOD | QUESTION |
|---|---|---|---|---|
| O | 528,937 | 6,238 | 1,858 | 53 |
| COMMA | 26,608 | 14,510 | 2,965 | 81 |
| PERIOD | 7,287 | 3,139 | 21,628 | 239 |
| QUESTION | 864 | 280 | 1,816 | 437 |

*CRF basic (M3), casing (masked)* (rows = gold, columns = predicted)

| gold \ pred | LOWER | FIRST_UPPER | ALL_UPPER |
|---|---|---|---|
| LOWER | 546,090 | 5,221 | 94 |
| FIRST_UPPER | 16,397 | 46,141 | 48 |
| ALL_UPPER | 436 | 126 | 1,496 |

*CRF improved (M3), punctuation* (rows = gold, columns = predicted)

| gold \ pred | O | COMMA | PERIOD | QUESTION |
|---|---|---|---|---|
| O | 527,258 | 7,596 | 2,119 | 113 |
| COMMA | 21,073 | 19,222 | 3,731 | 138 |
| PERIOD | 5,385 | 3,808 | 22,724 | 376 |
| QUESTION | 604 | 352 | 1,704 | 737 |

*CRF improved (M3), casing (masked)* (rows = gold, columns = predicted)

| gold \ pred | LOWER | FIRST_UPPER | ALL_UPPER |
|---|---|---|---|
| LOWER | 545,107 | 6,224 | 74 |
| FIRST_UPPER | 15,096 | 47,445 | 45 |
| ALL_UPPER | 396 | 137 | 1,525 |

## Limitations

- **Validation only.** These are model-comparison numbers on the primary validation lists. The final claims need the test and test_amb predictions from the same frozen models, scored once.
- **Settings were not tuned.** Every model uses c1 = c2 = 0.1 and 100 L-BFGS iterations, and every fit stopped at the iteration cap. The gaps may move with tuning. Member 5's reference CRF uses different code and settings (c1 = 1.0, c2 = 0.01, 200 iterations) and scored 4.19 points higher on mode A improved. Across the six presets, though, the differences have mixed signs.
- **Intervals cover sample variance only.** Each setting has one fit; the intervals do not cover training randomness or the number of comparisons.
- **Improved models were not reproduced.** Only mode A basic was retrained and matched byte for byte; the delivered code fingerprint differs from `aa232be`.
- **Ablations use a 50k-sequence sample.** Their absolute scores sit below the full-data models; read only the differences, which come from one fit each.
- **Rare classes:**
  - SEMICOLON scores 0 for every model.
  - EXCLAMATION and DASH stay under 4 F1.
  - `QUESTION|ALL_UPPER` has 18 validation tokens.
  - Read their per-class scores with their support.
