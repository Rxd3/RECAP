# Evaluation results: CRF models (Member 3) and Member 5 extensions

This is Member 5's evaluation of Member 3's six CRF models (modes A, B and C × basic and improved feature presets) and of Member 5's two extensions: threshold decoding, and a filler-word cue. It follows [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md). The HMM baseline is reported in [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md).

**Final results are on the primary test lists.**
- Validation chose the threshold τ (section 7). Member 3's CRF settings (c1 = c2 = 0.1, 100 iterations) were fixed and not tuned.
- No test result influenced any choice. Section 5 has the validation tables.
- Each frozen model predicted the primary test and test_amb lists exactly once.
- Every number comes from an evaluator JSON file in [`results/scores/`](../results/scores/README.md).

## Provenance

- **Models:**
  - Code: branch `feat/m3-crf-model`, commit `aa232be` (Member 3). The training manifests record a code fingerprint that differs from a clean `aa232be` checkout, so the models were trained from a working copy with uncommitted changes. Two checks show the shipped models match the committed code: the byte-identical reproductions below, and mode A basic retrained from a clean `aa232be` with byte-identical output.
  - Engine: `sklearn-crfsuite` 0.5.0 and `python-crfsuite` 0.9.12, L-BFGS.
  - Settings: c1 = c2 = 0.1, 100 iterations, fixed and not tuned. Every fit reached the iteration cap.
  - Training data: the official `default_train` lists (A/B 142,722 sequences, C 141,745).
  - Training machine, from the training manifests: AMD64 Family 23, 12 logical CPUs.
  - Fit time and peak memory: 335–875 s and about 7 GB for basic; 859–1,388 s and about 14.5 GB for improved. The manifests record Python "3.15.2", quoted as recorded.
- **Artifacts:**
  - The six `.joblib` files were shared on the team Drive with a `SHA256SUMS` file, and all six checksums match.
  - Before loading, Member 5 listed every import the pickles make. They are only `sklearn_crfsuite.estimator.CRF`, `sklearn_crfsuite._fileresource.FileResource` and `pycrfsuite._logparser.TrainLogParser`.
- **Reproduction:**
  - Member 5 regenerated all six validation prediction files from the delivered models with `scripts/train_crf.py predict`. Each is **byte-identical** to the file Member 3 delivered. The sha256 prefixes are:

    | Model | Basic | Improved |
    |---|---|---|
    | A | `cffd55c6` | `218a036e` |
    | B | `dcbd3656` | `8a84db7b` |
    | C | `8b17c871` | `7bb0d788` |

  - Mode A basic had also been retrained from a clean `aa232be` checkout, again with byte-identical output.
- **Test protocol:**
  - Each model predicted `test` and `test_amb` once with `train_crf.py predict --manifest-out`.
  - Every prediction file passes `evaluate_models.py score --manifest`.
  - Threshold decoding (section 7) picked one τ per mode on validation, then predicted test once.
- **Scoring:**
  - `scripts/evaluate_models.py` on the primary lists, with the `case_joint` mask in mode C.
  - Grouped bootstrap by video: 1,000 resamples, seed 2026, 95% percentile intervals, not adjusted for multiple comparisons.
  - Validation numbers were also confirmed by an independent from-scratch scorer.

**Metric:**
- A/B primary: macro F1 over the non-O punctuation classes.
- C primary: macro F1 over all 12 joint `PUNCT|CASE` classes on unmasked positions.
- Mode C micro P/R/F1 equals accuracy, because the micro average runs over all 12 classes.
- Why this metric: see [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md).

## Main findings (test)

1. **Both CRFs beat the HMM in every mode, and improved beats basic in every mode.** All nine paired intervals exclude 0.

   | Mode | HMM | CRF basic | CRF improved | Improved − basic | Improved − HMM |
   |---|---|---|---|---|---|
   | A | 35.12 | 43.65 | 49.92 | +6.27 | +14.80 |
   | B | 16.79 | 20.25 | 22.82 | +2.57 | +6.03 |
   | C | 37.41 | 43.85 | 47.02 | +3.17 | +9.61 |

2. **The CRFs learn punctuation inside the utterance.** Without each utterance's final token, mode A scores HMM 9.13, CRF basic 28.30 and CRF improved 36.40.
3. **The gain is mostly COMMA and QUESTION** (mode A test F1):

   | Class | HMM | CRF improved |
   |---|---|---|
   | COMMA | 26.28 | 52.52 |
   | QUESTION | 11.75 | 26.23 |
   | PERIOD | 67.34 | 71.02 |

4. **Threshold decoding adds a small, consistent gain:** +0.51 (A), +0.41 (B), +0.70 (C).
   - The A and C intervals exclude 0; B's crosses 0.
   - On test_amb the gain is +0.88 / +0.57 / +1.28, and all three intervals exclude 0.
   - In modes A and C most of it comes from comma recall. In mode B, comma and the rare classes contribute about equally.
5. **Leaked test utterances flatter the CRF, but not the HMM** (section 8).
   - On the 2,706 official test utterances whose video is also in the training set, the improved CRF scores 67.09 against 49.50 on the leak-free test.
   - The HMM scores 33.54 against 34.74.
   - The leakage-controlled split is what keeps the CRF's score honest.
6. **Casing (mode C, masked):** CRF improved 85.09, and 82.49 without the first word. The HMM scores 79.81 and 76.42.
7. **The most frequent error is still a missed comma** (COMMA → O): 42.9–44.9% of the improved model's punctuation errors on test.
8. **The ranking holds on test_amb**, the second held-out set:
   - Improved − HMM: +16.47 / +7.27 / +10.82.
   - Improved − basic: +6.61 / +3.15 / +3.58.
   - All six intervals exclude 0.

## Team hypotheses

The team wrote H1–H6 in its execution plan on 2026-10-29, before any CRF result existed; the original text is in [HYPOTHESES.md](HYPOTHESES.md). Each one is reported whatever its outcome.

| # | Hypothesis | Outcome | Evidence |
|---|---|---|---|
| H1 | Joint mode C gives higher PERIOD F1 than mode A | **Not supported for PERIOD** (mixed: higher only for CRF improved), but joint training helps inside the utterance | Paired comparison on mode A gold over the 19,936 mode C validation IDs: each mode A model against the same model type trained on mode C, projected to punctuation. PERIOD F1, A vs C: HMM 69.31 vs 68.33, CRF basic 71.57 vs 71.43, CRF improved 72.33 vs 72.63. The joint CRF improved gains +0.42 macro (0.05 to 0.82), mostly on QUESTION (+1.51). Every joint model gains on non-final positions, and all three intervals exclude 0: HMM +8.69, CRF basic +1.59, CRF improved +1.16. Overall, the joint HMM loses 0.62 (−1.20 to −0.03). |
| H2 | Threshold decoding raises rare-class recall | **Partly supported** (test) | Recall rises for almost every non-O class, but mostly for COMMA (mode A 45.22 → 52.63; mode B 45.36 → 57.80). The rare classes gain under 1 point: QUESTION A +0.46; B COLON +0.60, DASH +0.38, SEMICOLON +0.83, EXCLAMATION 0. Macro F1 gains +0.51 / +0.41 / +0.70 (sections 1–2; the τ grid is in section 7). |
| H3 | Question cues raise QUESTION F1 | **Supported** (validation ablation) | Removing the question cues lowers QUESTION F1 by 4.05 (A), 2.85 (B) and 4.93 (C, punctuation projection). Primary intervals exclude 0 in every mode (section 6). |
| H4 | Filler and discourse cues raise COMMA F1 | **Not supported** (validation) | Adding a filler cue (*um*, *uh*, *er*, *erm*, *like*, *you know*, *i mean*, repeated words) changes primary macro F1 by +0.04 (A), +0.04 (B) and −0.28 (C joint), and COMMA F1 by +0.14, +0.21 and +0.18 (C, punctuation projection). All primary intervals include 0. Removing the clause cues (conjunctions and discourse markers) changes COMMA F1 by +0.03 (A), +0.05 (B) and −0.20 (C, punctuation projection). |
| H5 | Much of PERIOD F1 comes from the utterance-final token | **Supported** (test) | The shortcut alone reaches PERIOD F1 66.68 in mode A, against the HMM's 67.34. Without the final token, the HMM's primary score falls from 35.12 to 9.13. |
| H6 | The CRF beats the HMM in every mode | **Supported** (test and test_amb) | CRF basic − HMM and CRF improved − HMM: all six paired intervals exclude 0 on test (section 2) and on test_amb (section 9). |

## 1. Main result (test, primary test ID lists, each model predicted once)

Primary = macro F1 % with 95% CI (grouped bootstrap by video, 1,000 resamples, seed 2026). Micro P / R / F1 % over the same classes. A/B: non-O punctuation classes; C: all 12 joint classes on unmasked positions (case_joint mask applied).

**Mode A** (`punctuation/all`)

| Model | Macro F1 (95% CI) | Micro P / R / F1 | Accuracy |
|---|---|---|---|
| Shortcut | 27.48 (25.55–29.38) | 89.17 / 20.79 / 33.72 | 89.73 |
| HMM (M2) | 35.12 (32.80–37.31) | 76.78 / 30.87 / 44.03 | 90.29 |
| CRF basic (M3) | 43.65 (41.74–45.59) | 69.66 / 44.41 / 54.24 | 91.53 |
| CRF improved (M3) | 49.92 (47.91–52.01) | 68.13 / 52.71 / 59.43 | 92.25 |
| CRF improved + threshold (M5) | 50.44 (48.53–52.41) | 62.97 / 57.73 / 60.24 | 92.00 |

**Mode B** (`punctuation/all`)

| Model | Macro F1 (95% CI) | Micro P / R / F1 | Accuracy |
|---|---|---|---|
| Shortcut | 11.87 (11.02–12.68) | 89.17 / 20.79 / 33.72 | 89.73 |
| HMM (M2) | 16.79 (15.28–18.49) | 67.58 / 30.68 / 42.20 | 89.76 |
| CRF basic (M3) | 20.25 (18.87–21.95) | 69.18 / 43.76 / 53.61 | 91.47 |
| CRF improved (M3) | 22.82 (21.42–24.42) | 67.47 / 51.86 / 58.64 | 92.18 |
| CRF improved + threshold (M5) | 23.23 (21.74–25.05) | 58.14 / 59.93 / 59.02 | 91.39 |

**Mode C** (`joint/unmasked`)

| Model | Macro F1 (95% CI) | Micro P / R / F1 | Accuracy |
|---|---|---|---|
| Shortcut | 18.09 (17.34–18.76) | 83.30 / 83.30 / 83.30 | 83.30 |
| HMM (M2) | 37.41 (34.56–40.71) | 86.80 / 86.80 / 86.80 | 86.80 |
| CRF basic (M3) | 43.85 (42.13–45.47) | 88.40 / 88.40 / 88.40 | 88.40 |
| CRF improved (M3) | 47.02 (44.97–49.00) | 89.15 / 89.15 / 89.15 | 89.15 |
| CRF improved + threshold (M5) | 47.72 (45.78–49.61) | 88.84 / 88.84 / 88.84 | 88.84 |

## 2. Paired differences (test, points, 95% CI; * = interval excludes 0)

| Mode | Comparison | Primary | non_final |
|---|---|---|---|
| A | M3 improved − M3 basic | +6.27 (5.54 to 6.96) * | +8.10 (7.13 to 9.02) * |
| A | M3 basic − HMM | +8.53 (7.55 to 9.54) * | +19.17 (17.73 to 20.58) * |
| A | M3 improved − HMM | +14.80 (13.49 to 16.12) * | +27.27 (25.49 to 29.02) * |
| A | threshold − M3 improved | +0.51 (0.22 to 0.80) * | +1.18 (0.72 to 1.61) * |
| B | M3 improved − M3 basic | +2.57 (1.97 to 3.22) * | +3.23 (2.57 to 3.91) * |
| B | M3 basic − HMM | +3.46 (2.83 to 4.08) * | +8.09 (7.32 to 8.85) * |
| B | M3 improved − HMM | +6.03 (5.21 to 6.87) * | +11.33 (10.41 to 12.22) * |
| B | threshold − M3 improved | +0.41 (-0.05 to 1.10) | +0.94 (0.43 to 1.68) * |
| C | M3 improved − M3 basic | +3.17 (1.85 to 4.26) * | +7.58 (6.44 to 8.68) * |
| C | M3 basic − HMM | +6.43 (3.22 to 9.00) * | +11.77 (10.67 to 12.87) * |
| C | M3 improved − HMM | +9.61 (6.20 to 12.38) * | +19.35 (17.84 to 20.83) * |
| C | threshold − M3 improved | +0.70 (0.19 to 1.24) * | +1.46 (1.01 to 1.92) * |

## 3. Views and mode C sub-blocks (test, macro F1 %, 95% CI)

`non_final` drops each utterance's last token. Mode C casing uses the case_joint mask; `casing/all` is unmasked, for reference only.

| Model | A non_final | B non_final | C non_final | C punctuation/all | C casing (masked) | C casing non-initial | C casing/all |
|---|---|---|---|---|---|---|---|
| Shortcut | 0.00 (0.00–0.00) | 0.00 (0.00–0.00) | 0.00 (0.00–0.00) | 27.44 (25.35–29.34) | 47.09 (46.10–48.04) | 32.00 (31.90–32.11) | 46.96 (45.98–47.91) |
| HMM (M2) | 9.13 (7.97–10.18) | 5.51 (4.18–6.93) | 17.42 (15.93–19.11) | 34.38 (32.28–36.51) | 79.81 (77.03–82.46) | 76.42 (73.54–79.24) | 79.72 (76.96–82.37) |
| CRF basic (M3) | 28.30 (26.26–30.28) | 13.60 (12.21–15.20) | 29.19 (27.53–31.03) | 43.67 (41.66–45.74) | 84.23 (82.20–86.13) | 81.42 (79.28–83.46) | 84.20 (82.18–86.10) |
| CRF improved (M3) | 36.40 (34.14–38.46) | 16.83 (15.38–18.39) | 36.77 (34.87–38.84) | 49.80 (47.67–51.97) | 85.09 (83.02–87.10) | 82.49 (80.30–84.60) | 85.00 (82.88–87.01) |
| CRF improved + threshold (M5) | 37.58 (35.49–39.49) | 17.77 (16.18–19.61) | 38.23 (36.33–40.23) | 50.46 (48.55–52.41) | 85.32 (83.28–87.23) | 82.72 (80.52–84.79) | 85.24 (83.20–87.17) |

## 4. Per-class precision / recall / F1 % (test)

Classes marked O are not in the A/B primary average; in mode C all 12 joint classes are.

**Mode A**

| Class | Support | Shortcut P / R / F1 | HMM (M2) P / R / F1 | CRF basic (M3) P / R / F1 | CRF improved (M3) P / R / F1 | CRF improved + threshold (M5) P / R / F1 |
|---|---|---|---|---|---|---|
| O | 354,902 | 89.75 / 100.00 / 94.60 | 91.03 / 99.14 / 94.91 | 93.50 / 98.55 / 95.96 | 94.94 / 98.14 / 96.52 | 95.92 / 97.10 / 96.51 |
| COMMA | 30,103 | 0.00 / 0.00 / 0.00 | 58.94 / 16.91 / 26.28 | 61.32 / 33.24 / 43.11 | 62.64 / 45.22 / 52.52 | 57.38 / 52.63 / 54.90 |
| PERIOD | 20,570 | 91.80 / 52.35 / 66.68 | 89.84 / 53.85 / 67.34 | 77.89 / 64.39 / 70.50 | 75.00 / 67.43 / 71.02 | 71.09 / 69.45 / 70.26 |
| QUESTION | 2,175 | 36.95 / 10.02 / 15.77 | 52.17 / 6.62 / 11.75 | 59.41 / 10.16 / 17.35 | 56.28 / 17.10 / 26.23 | 51.14 / 17.56 / 26.15 |

**Mode B**

| Class | Support | Shortcut P / R / F1 | HMM (M2) P / R / F1 | CRF basic (M3) P / R / F1 | CRF improved (M3) P / R / F1 | CRF improved + threshold (M5) P / R / F1 |
|---|---|---|---|---|---|---|
| O | 354,902 | 89.75 / 100.00 / 94.60 | 91.15 / 98.56 / 94.71 | 93.46 / 98.58 / 95.95 | 94.91 / 98.18 / 96.52 | 96.51 / 96.07 / 96.29 |
| COMMA | 29,153 | 0.00 / 0.00 / 0.00 | 60.11 / 17.04 / 26.56 | 61.01 / 33.36 / 43.14 | 62.09 / 45.36 / 52.42 | 52.35 / 57.80 / 54.94 |
| PERIOD | 20,246 | 91.80 / 53.19 / 67.35 | 89.85 / 54.68 / 67.99 | 77.34 / 65.06 / 70.67 | 74.44 / 68.15 / 71.16 | 68.01 / 71.19 / 69.57 |
| QUESTION | 2,175 | 36.95 / 10.02 / 15.77 | 55.38 / 6.62 / 11.83 | 57.95 / 9.89 / 16.89 | 55.24 / 17.20 / 26.23 | 46.83 / 17.98 / 25.98 |
| EXCLAMATION | 204 | 0.00 / 0.00 / 0.00 | 2.01 / 3.43 / 2.54 | 20.00 / 1.47 / 2.74 | 16.67 / 0.98 / 1.85 | 9.52 / 0.98 / 1.78 |
| COLON | 166 | 0.00 / 0.00 / 0.00 | 16.67 / 4.22 / 6.73 | 13.46 / 4.22 / 6.42 | 21.05 / 4.82 / 7.84 | 14.52 / 5.42 / 7.89 |
| SEMICOLON | 120 | 0.00 / 0.00 / 0.00 | 1.23 / 0.83 / 1.00 | 50.00 / 0.83 / 1.64 | 0.00 / 0.00 / 0.00 | 14.29 / 0.83 / 1.57 |
| DASH | 784 | 0.00 / 0.00 / 0.00 | 0.60 / 2.04 / 0.93 | 6.67 / 0.13 / 0.25 | 1.69 / 0.13 / 0.24 | 2.45 / 0.51 / 0.84 |

**Mode C joint (masked)**

| Class | Support | Shortcut P / R / F1 | HMM (M2) P / R / F1 | CRF basic (M3) P / R / F1 | CRF improved (M3) P / R / F1 | CRF improved + threshold (M5) P / R / F1 |
|---|---|---|---|---|---|---|
| O\|LOWER | 317,972 | 83.21 / 99.97 / 90.83 | 88.82 / 98.09 / 93.23 | 91.16 / 97.64 / 94.29 | 92.69 / 97.12 / 94.85 | 93.87 / 95.89 / 94.87 |
| O\|FIRST_UPPER | 35,294 | 86.67 / 30.11 / 44.70 | 83.57 / 68.95 / 75.56 | 84.70 / 73.52 / 78.71 | 83.86 / 75.58 / 79.51 | 86.23 / 73.95 / 79.62 |
| O\|ALL_UPPER | 657 | 0.00 / 0.00 / 0.00 | 66.40 / 49.62 / 56.79 | 73.94 / 60.88 / 66.78 | 78.00 / 64.23 / 70.45 | 79.81 / 62.56 / 70.14 |
| COMMA\|LOWER | 25,450 | 0.00 / 0.00 / 0.00 | 56.41 / 11.93 / 19.70 | 57.30 / 30.75 / 40.02 | 59.98 / 42.58 / 49.80 | 52.96 / 52.24 / 52.59 |
| COMMA\|FIRST_UPPER | 4,357 | 0.00 / 0.00 / 0.00 | 46.80 / 20.50 / 28.51 | 56.25 / 33.05 / 41.64 | 57.08 / 36.65 / 44.64 | 52.63 / 42.71 / 47.16 |
| COMMA\|ALL_UPPER | 171 | 0.00 / 0.00 / 0.00 | 26.83 / 6.43 / 10.38 | 40.00 / 23.39 / 29.52 | 44.33 / 25.15 / 32.09 | 41.54 / 31.58 / 35.88 |
| PERIOD\|LOWER | 18,708 | 85.01 / 53.00 / 65.29 | 71.17 / 62.03 / 66.29 | 73.63 / 66.13 / 69.68 | 71.74 / 69.32 / 70.51 | 67.35 / 71.78 / 69.50 |
| PERIOD\|FIRST_UPPER | 1,637 | 50.00 / 0.06 / 0.12 | 34.35 / 34.94 / 34.65 | 53.75 / 38.55 / 44.90 | 56.56 / 40.56 / 47.24 | 52.33 / 41.78 / 46.47 |
| PERIOD\|ALL_UPPER | 125 | 0.00 / 0.00 / 0.00 | 53.66 / 17.60 / 26.51 | 54.84 / 27.20 / 36.36 | 57.35 / 31.20 / 40.41 | 53.95 / 32.80 / 40.80 |
| QUESTION\|LOWER | 2,004 | 35.85 / 10.43 / 16.16 | 37.82 / 7.44 / 12.43 | 53.71 / 10.83 / 18.02 | 51.58 / 17.96 / 26.65 | 46.38 / 18.51 / 26.46 |
| QUESTION\|FIRST_UPPER | 148 | 0.00 / 0.00 / 0.00 | 28.57 / 2.70 / 4.94 | 41.67 / 3.38 / 6.25 | 28.00 / 4.73 / 8.09 | 30.77 / 5.41 / 9.20 |
| QUESTION\|ALL_UPPER | 9 | 0.00 / 0.00 / 0.00 | 100.00 / 11.11 / 20.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |

**Mode C casing (masked)**

| Class | Support | Shortcut P / R / F1 | HMM (M2) P / R / F1 | CRF basic (M3) P / R / F1 | CRF improved (M3) P / R / F1 | CRF improved + threshold (M5) P / R / F1 |
|---|---|---|---|---|---|---|
| LOWER | 364,134 | 92.33 / 99.98 / 96.00 | 96.52 / 98.93 / 97.71 | 97.08 / 99.03 / 98.04 | 97.29 / 98.86 / 98.07 | 97.24 / 98.99 / 98.11 |
| FIRST_UPPER | 41,436 | 99.08 / 29.32 / 45.26 | 88.17 / 69.59 / 77.79 | 89.67 / 74.41 / 81.33 | 88.34 / 76.34 / 81.91 | 89.50 / 75.90 / 82.14 |
| ALL_UPPER | 962 | 0.00 / 0.00 / 0.00 | 85.54 / 51.04 / 63.93 | 86.67 / 63.51 / 73.31 | 88.95 / 65.28 / 75.30 | 88.35 / 66.22 / 75.70 |

## 5. Validation (model selection)

### Main result (validation, primary ID lists)

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

### Paired differences (validation, points, 95% CI; * = interval excludes 0)

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

## 6. Feature ablations of the improved CRF (validation)

Member 5 retrained Member 3's improved preset with one feature group switched off at a time, using `scripts/run_ablations.py`. The feature code is Member 3's from `aa232be`, unchanged on `release/final` apart from the filler switch, which is off by default. The settings match the delivered models (c1 = c2 = 0.1, 100 iterations). Unlike them, each ablation model trains on a fixed sample of 50,000 `default_train` sequences per mode, drawn with seed 2026; modes A and B share one sample. All models are scored on the primary validation lists. Test was not used.

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
2. **The ±2 window and the question cues also help in every mode.**
   - All three primary intervals exclude 0 for both.
   - The mode C effects (−0.53 and −0.40) do not survive a Holm correction (see Multiple comparisons).
3. **Clause cues, casing cues and position buckets have small effects of mixed sign.**
   - On the primary metric, every effect is under 1 point.
   - In mode B, removing clause cues (+0.17) or position buckets (+0.21) slightly *raises* the score; neither survives a Holm correction.
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

### Filler cue (Member 5 extension, H4)

`FeatureConfig.improved(use_filler_cues=True)` adds filler features on top of the improved preset: *um*, *uh*, *er*, *erm*, *like*, the bigrams *you know* and *i mean*, and an immediately repeated word. It was trained on the same sample with the same seed and settings (`scripts/run_ablations.py --variants improved,improved_plus_filler`). The improved-preset predictions are byte-identical to the ablation run above.

Difference = with filler − without, in points (validation, 95% paired CI; * = interval excludes 0):

| Mode | Primary | non_final | COMMA F1 |
|---|---|---|---|
| A | +0.04 (-0.19 to 0.27) | +0.10 (-0.20 to 0.39) | +0.14 |
| B | +0.04 (-0.23 to 0.29) | +0.19 (-0.07 to 0.45) | +0.21 |
| C | -0.28 (-0.66 to 0.11) | +0.13 (-0.15 to 0.42) | +0.18 |

No interval excludes 0, so the filler cue has no detectable effect. The upper bounds (+0.27, +0.29 and +0.11 primary) do not rule out a small benefit.

## 7. Threshold decoding (Member 5 extension)

For each token, the improved CRF's marginals give P(O); in mode C, P(O) sums every `O|*` label. If 1 − P(O) > τ the token gets its most probable non-O label, otherwise its most probable O label (`src/punct_restore/models/threshold.py`). τ was tuned per mode on the primary validation list (`scripts/tune_threshold.py tune`, grid 0.05–0.95), then test was predicted once with the chosen τ: A 0.4, B 0.3, C 0.35. Test effects are in sections 1–2.

Validation primary macro F1 % per τ (bold = chosen; standard Viterbi decoding scores A 51.18, B 25.10, C 51.52):

| τ | A | B | C |
|---|---|---|---|
| 0.05 | 45.10 | 22.53 | 47.28 |
| 0.10 | 48.33 | 24.10 | 50.06 |
| 0.15 | 49.97 | 24.61 | 51.21 |
| 0.20 | 50.78 | 24.99 | 51.89 |
| 0.25 | 51.29 | 25.24 | 52.17 |
| 0.30 | 51.55 | **25.41** | 52.34 |
| 0.35 | 51.72 | 25.30 | **52.44** |
| 0.40 | **51.80** | 25.30 | 52.32 |
| 0.45 | 51.71 | 25.19 | 52.22 |
| 0.50 | 51.46 | 25.11 | 52.12 |
| 0.55 | 51.14 | 25.01 | 51.73 |
| 0.60 | 50.73 | 24.73 | 51.48 |
| 0.65 | 50.31 | 24.45 | 51.13 |
| 0.70 | 49.68 | 24.14 | 50.79 |
| 0.75 | 48.85 | 23.80 | 50.19 |
| 0.80 | 47.74 | 23.26 | 49.42 |
| 0.85 | 46.25 | 22.53 | 48.36 |
| 0.90 | 44.04 | 21.31 | 46.54 |
| 0.95 | 39.77 | 19.14 | 42.60 |

## 8. Leakage: official vs leakage-controlled test (mode A)

Supplementary. The frozen improved CRF and the HMM predicted all 17,695 official test utterances once (the CRF's predictions on the primary list are identical to section 1's). The excluded utterances are the official test minus `data/splits/sponspeech-m1-v2/leakage_controlled/test.ids`, split by why they were excluded (`leakage_controlled/test_excluded.tsv`). "Training set" means a video with an utterance in `default_train`, which both models were trained on. Macro F1 % (95% CI; the quality group is one video, so its interval is degenerate).

| Scored set | Gold labels | Utterances | HMM | CRF improved |
|---|---|---|---|---|
| Official test, all utterances | canonical | 17,695 | 34.98 (33.01–36.83) | 52.84 (50.88–54.81) |
| Leakage-controlled test (kept) | canonical | 12,668 | 34.74 (32.46–36.90) | 49.50 (47.56–51.52) |
| Excluded: video also in the training set | canonical | 2,706 | 33.54 (29.94–36.84) | 67.09 (64.42–70.09) |
| Excluded: other leakage (video shared with validation only, or duplicate text) | canonical | 2,130 | 37.50 (27.91–42.21) | 53.97 (49.57–57.49) |
| Excluded: quality filter (low comma density or TTS source) | canonical | 191 | 27.93 (27.93–27.93) | 43.12 (43.12–43.12) |
| Official test, all utterances | SponSpeech native | 17,695 | 35.10 (33.15–36.95) | 52.75 (50.79–54.72) |

On utterances from videos it was trained on, the improved CRF scores about 18 points higher than on the leak-free set; the HMM does not gain at all. Reporting the official test as-is would overstate the CRF by about 3 points.

## 9. test_amb (second held-out set)

Primary macro F1 % (95% CI) on the primary test_amb lists. test_amb shares 25 videos with test, so the two are reported separately and never pooled.

| Model | A | B | C |
|---|---|---|---|
| Shortcut | 23.90 (22.26–25.56) | 10.31 (9.61–11.01) | 16.86 (16.23–17.49) |
| HMM (M2) | 31.50 (29.36–33.66) | 15.32 (13.85–16.72) | 36.34 (34.92–37.53) |
| CRF basic (M3) | 41.35 (39.51–43.23) | 19.44 (18.05–20.90) | 43.57 (41.59–45.23) |
| CRF improved (M3) | 47.96 (45.91–50.02) | 22.59 (21.06–24.06) | 47.16 (45.12–48.89) |
| CRF improved + threshold (M5) | 48.85 (46.88–50.81) | 23.16 (21.83–24.36) | 48.44 (46.52–50.08) |

| Paired difference (test_amb) | A | B | C |
|---|---|---|---|
| M3 improved − M3 basic | +6.61 (5.88 to 7.38) * | +3.15 (2.45 to 3.90) * | +3.58 (2.44 to 4.56) * |
| M3 basic − HMM | +9.85 (8.92 to 10.92) * | +4.12 (3.41 to 5.03) * | +7.23 (6.18 to 8.50) * |
| M3 improved − HMM | +16.47 (15.41 to 17.72) * | +7.27 (6.37 to 8.35) * | +10.82 (9.54 to 12.15) * |
| threshold − M3 improved | +0.88 (0.59 to 1.20) * | +0.57 (0.05 to 1.20) * | +1.28 (0.76 to 1.87) * |

## 10. Runtime

Prediction speed on one laptop (Intel64 Family 6 Model 158, Python 3.13.1, one process, no other jobs): median of 3 passes over the primary validation set after a 200-record warm-up, model loading excluded (the same procedure and machine as the HMM rows in EVALUATION_RESULTS.md §7). Fit time is from each model's training manifest, measured on Member 3's machine (AMD64, 12 logical CPUs), so it is not comparable with the HMM's fit time.

| Model | Mode | Fit s (M3 machine) | Predict s | Tokens/s | ms/utterance |
|---|---|---|---|---|---|
| CRF basic | A | 334.5 | 5.54 | 111,775 | 0.276 |
| CRF basic | B | 510.8 | 5.82 | 106,291 | 0.290 |
| CRF basic | C | 874.9 | 6.17 | 99,917 | 0.310 |
| CRF improved | A | 859.1 | 12.41 | 49,855 | 0.619 |
| CRF improved | B | 1,084.3 | 12.72 | 48,647 | 0.635 |
| CRF improved | C | 1,388.3 | 13.23 | 46,629 | 0.664 |

The improved preset predicts at about half the basic preset's speed: under 0.32 ms per utterance for basic, under 0.67 ms for improved.

## Most frequent errors, M3 improved (test, punctuation labels; mode C projected to punctuation)

| Mode | Errors | Top pairs (count, share) |
|---|---|---|
| A | 31,583 | COMMA→O 14,177 (45%); O→COMMA 5,286 (17%); PERIOD→O 3,920 (12%); PERIOD→COMMA 2,607 (8%); COMMA→PERIOD 2,249 (7%) |
| B | 31,891 | COMMA→O 13,674 (43%); O→COMMA 5,108 (16%); PERIOD→O 3,818 (12%); PERIOD→COMMA 2,440 (8%); COMMA→PERIOD 2,150 (7%) |
| C | 31,614 | COMMA→O 14,127 (45%); O→COMMA 5,167 (16%); PERIOD→O 3,724 (12%); COMMA→PERIOD 2,654 (8%); PERIOD→COMMA 2,423 (8%) |

## Confusion matrices, M3 CRFs (test)

Mode C is shown as its punctuation projection and its masked casing projection; the 12 × 12 joint matrix is in each score JSON (`blocks.joint/unmasked.confusion`).

### Mode A

*CRF basic (M3), punctuation* (rows = gold, columns = predicted)

| gold \ pred | O | COMMA | PERIOD | QUESTION |
|---|---|---|---|---|
| O | 349,754 | 4,138 | 990 | 20 |
| COMMA | 18,465 | 10,006 | 1,601 | 31 |
| PERIOD | 5,228 | 1,997 | 13,245 | 100 |
| QUESTION | 609 | 177 | 1,168 | 221 |

*CRF improved (M3), punctuation* (rows = gold, columns = predicted)

| gold \ pred | O | COMMA | PERIOD | QUESTION |
|---|---|---|---|---|
| O | 348,312 | 5,286 | 1,252 | 52 |
| COMMA | 14,177 | 13,612 | 2,249 | 65 |
| PERIOD | 3,920 | 2,607 | 13,871 | 172 |
| QUESTION | 455 | 226 | 1,122 | 372 |

### Mode B

*CRF basic (M3), punctuation* (rows = gold, columns = predicted)

| gold \ pred | O | COMMA | PERIOD | QUESTION | EXCLAMATION | COLON | SEMICOLON | DASH |
|---|---|---|---|---|---|---|---|---|
| O | 349,856 | 3,991 | 998 | 22 | 2 | 26 | 0 | 7 |
| COMMA | 17,857 | 9,726 | 1,518 | 31 | 6 | 11 | 1 | 3 |
| PERIOD | 5,112 | 1,848 | 13,172 | 101 | 4 | 5 | 0 | 4 |
| QUESTION | 614 | 160 | 1,185 | 215 | 0 | 1 | 0 | 0 |
| EXCLAMATION | 97 | 59 | 43 | 1 | 3 | 1 | 0 | 0 |
| COLON | 125 | 23 | 10 | 1 | 0 | 7 | 0 | 0 |
| SEMICOLON | 72 | 22 | 25 | 0 | 0 | 0 | 1 | 0 |
| DASH | 590 | 112 | 80 | 0 | 0 | 1 | 0 | 1 |

*CRF improved (M3), punctuation* (rows = gold, columns = predicted)

| gold \ pred | O | COMMA | PERIOD | QUESTION | EXCLAMATION | COLON | SEMICOLON | DASH |
|---|---|---|---|---|---|---|---|---|
| O | 348,452 | 5,108 | 1,246 | 55 | 4 | 15 | 0 | 22 |
| COMMA | 13,674 | 13,224 | 2,150 | 64 | 4 | 7 | 0 | 30 |
| PERIOD | 3,818 | 2,440 | 13,798 | 177 | 1 | 4 | 2 | 6 |
| QUESTION | 456 | 218 | 1,125 | 374 | 0 | 2 | 0 | 0 |
| EXCLAMATION | 80 | 52 | 66 | 3 | 2 | 1 | 0 | 0 |
| COLON | 97 | 42 | 19 | 0 | 0 | 8 | 0 | 0 |
| SEMICOLON | 57 | 37 | 26 | 0 | 0 | 0 | 0 | 0 |
| DASH | 494 | 178 | 105 | 4 | 1 | 1 | 0 | 1 |

### Mode C

*CRF basic (M3), punctuation* (rows = gold, columns = predicted)

| gold \ pred | O | COMMA | PERIOD | QUESTION |
|---|---|---|---|---|
| O | 348,636 | 4,259 | 1,325 | 37 |
| COMMA | 18,032 | 9,976 | 2,018 | 34 |
| PERIOD | 4,881 | 1,980 | 13,549 | 116 |
| QUESTION | 585 | 154 | 1,198 | 231 |

*CRF basic (M3), casing (masked)* (rows = gold, columns = predicted)

| gold \ pred | LOWER | FIRST_UPPER | ALL_UPPER |
|---|---|---|---|
| LOWER | 360,589 | 3,478 | 67 |
| FIRST_UPPER | 10,578 | 30,831 | 27 |
| ALL_UPPER | 278 | 73 | 611 |

*CRF improved (M3), punctuation* (rows = gold, columns = predicted)

| gold \ pred | O | COMMA | PERIOD | QUESTION |
|---|---|---|---|---|
| O | 347,613 | 5,167 | 1,394 | 83 |
| COMMA | 14,127 | 13,208 | 2,654 | 71 |
| PERIOD | 3,724 | 2,423 | 14,192 | 187 |
| QUESTION | 440 | 211 | 1,133 | 384 |

*CRF improved (M3), casing (masked)* (rows = gold, columns = predicted)

| gold \ pred | LOWER | FIRST_UPPER | ALL_UPPER |
|---|---|---|---|
| LOWER | 359,979 | 4,107 | 48 |
| FIRST_UPPER | 9,773 | 31,633 | 30 |
| ALL_UPPER | 267 | 67 | 628 |

## Multiple comparisons (Holm)

The intervals in this document are not adjusted for the number of comparisons. As a check, two-sided bootstrap p-values were taken from each paired comparison's share of resamples in which B beats A (resolution 0.001), and Holm's step-down correction was applied at α = 0.05 within two families.

- **Test comparisons (12):** improved − basic, basic − HMM and improved − HMM in each mode, plus threshold − improved.
  - All nine model comparisons survive.
  - Threshold decoding survives in modes A (p ≤ 0.002) and C (p ≤ 0.004), but not B (p ≤ 0.100).
- **Ablations (18):** six feature groups in three modes, validation.
  - Survive: removing word pairs (A, B), the ±2 window (A, B) or the question cues (A, B).
  - Do not survive: everything in mode C (word pairs p ≤ 0.096, window p ≤ 0.042, question p ≤ 0.032) and the small mode B effects of the clause cues and position buckets.

The p-values are recomputed from the committed compare JSONs in `results/scores/crf_test/`, `results/scores/threshold/` and `results/scores/ablations/`.

## Limitations

- **CRF settings were not tuned.** Every model uses c1 = c2 = 0.1 and 100 L-BFGS iterations, and every fit stopped at the iteration cap. Only τ was tuned, so the gaps may move with other settings.
- **Intervals cover sample variance only.** Each setting has one fit. The intervals are not adjusted for the number of comparisons; the Holm check above shows which conclusions survive.
- **Ablation and filler models use a 50k-sequence sample.** Read only their differences; each comes from one fit.
- **Rare classes stay low.** In mode B the improved CRF scores under 8 F1 on COLON, DASH, EXCLAMATION and SEMICOLON on test. `QUESTION|ALL_UPPER` has 9 test tokens.
- **Gold noise.** Leftover speaker labels give false COLON labels; see EVALUATION_RESULTS.md.
- **Training environment.** Fit times, memory and the Python version come from Member 3's training manifests and were not re-measured.
- **test and test_amb share 25 videos.** They are reported separately and never pooled.
- **Section 8 is supplementary.** It scores the frozen models on additional official-test utterances; nothing was selected on it.

## Reproduce

Run every command from the repository root. `<rel>` is `sponspeech-m1-v2`. The shared models are listed in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

- **Test predictions, once per model:** `python scripts/train_crf.py predict --model crf_mode_<m>_<preset>_full_01.joblib --input data/processed/<rel>/mode_<M>/test.jsonl --ids data/splits/<rel>/primary_eval/test_mode_<M>.ids --output PRED.jsonl --manifest-out PRED.manifest.json`. The same command on validation reproduces Member 3's delivered files byte for byte.
- **Scores and paired comparisons:** `python scripts/evaluate_models.py score|compare|errors ... --bootstrap 1000 --seed 2026`. Mode C needs `--masks data/processed/<rel>/masks/case_joint/<split>.jsonl`.
- **Threshold decoding (§7):** `python scripts/tune_threshold.py tune ...` on validation, then `predict --tau-file ...` once on test and test_amb.
- **Ablations and filler cue (§6):** `python scripts/run_ablations.py --mode <M> --out DIR [--variants ...]`, then score as above.
- **Joint vs punctuation-only (H1):** `python scripts/project_joint.py --predictions C_PRED.jsonl --gold data/processed/<rel>/mode_A/validation.jsonl --out A_PRED.jsonl --manifest-out A_PRED.manifest.json`, then `compare` on mode A gold with `--ids data/splits/<rel>/primary_eval/validation_mode_C.ids`.
- **Leakage subsets (§8):**
  - `python scripts/build_leakage_subsets.py --mode A --split test --out DIR` writes the `all`, `kept`, `excluded_train_video`, `excluded_other_leak` and `excluded_quality` ID lists, with counts 17,695 / 12,668 / 2,706 / 2,130 / 191.
  - Predict the whole official test once: `python scripts/train_crf.py predict --model crf_mode_a_improved_full_01.joblib --input data/processed/<rel>/mode_A/test.jsonl --ids DIR/all.ids --output ALL.jsonl --manifest-out ALL.manifest.json`. Its primary-list predictions are identical to the test run above.
  - Score once per list: `python scripts/evaluate_models.py score --gold data/processed/<rel>/mode_A/test.jsonl --ids DIR/<list>.ids --predictions ALL.jsonl --bootstrap 1000 --out OUT.json`. For the native rows, use `--gold data/processed/<rel>/benchmark_native/mode_A/test.jsonl`.
- **Tables in this document:** `python scripts/reporting/build_m3_tables.py --split test --tau --out OUT.md` (§1–4, errors, confusion matrices).
  - `python scripts/reporting/build_m3_final_sections.py --manifests results/scores/crf_training_manifests --out OUT.md` gives §7–10.
  - `python scripts/reporting/build_ablation_table.py --out OUT.md` gives the §6 tables.
  - All three read the committed JSON in `results/scores/` and reproduce the tables above line for line.
- **Runtime (§10):** `python scripts/reporting/time_models.py --kind crf --model M --mode <M> --out OUT.json`.
