# Evaluation results: CRF models (Member 3) and Member 5 extensions

This is Member 5's evaluation of Member 3's six CRF models (modes A, B and C × basic and improved feature presets) and of Member 5's additions: threshold decoding and a filler-word cue (sections 6–7), the gold-noise and simulated-ASR checks (sections 12–13), and the exploratory comparison methods of section 14. It follows [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md). The HMM baseline is reported in [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md).

**Final results are on the primary test lists.**
- Validation chose the threshold τ (section 7). Member 3's CRF settings (c1 = c2 = 0.1, 100 iterations) were fixed and not tuned.
- The course-baseline settings (HMM smoothing, CRF settings, the plain threshold τ) were fixed or chosen on validation before any CRF test result; the HMM settings were frozen before any test access. Section 5 has the validation tables.
- The comparison methods in sections 13–14 (simulated ASR noise, gap classifier, hidden-event LM, prior-adjusted decoding, question lookback, factored casing) were added on 2026-10-30, after the first test report, in response to a review that cited test results. They are exploratory; their settings were still chosen on validation only, and each test list was predicted once.
- Each frozen model predicted the primary test and test_amb lists exactly once. For the supplementary leakage analysis (section 8), the mode A improved model also predicted the full official test once.
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
  - During the work, validation scores were cross-checked against scikit-learn's `f1_score`; that check is not committed.

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
   - Most of it comes from comma recall: COMMA F1 rises by +2.38 to +2.79 in every mode. The rare-class changes are individually within noise.
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
9. **Team comparison methods** (section 14; exploratory, added after the first test report):
   - Prior-adjusted threshold decoding is the best decoder on test: +1.42 / +1.43 / +2.74 over the improved CRF and +0.91 / +1.03 / +2.04 over the plain threshold, every interval excluding 0. It trades PERIOD (−1.79 in A) for QUESTION (+4.00 in A), and half of the mode C gain comes from two tiny ALL_UPPER classes.
   - A classic hidden-event LM nearly matches the CRF on punctuation (−1.54 in A, a tie in B).
   - The CRF's transitions, together with its different training settings, are worth 1.3–4.7 points over a per-token classifier (validation, 50k sample).
   - Question-lookback features add +6.11 / +7.12 / +6.01 QUESTION F1 (validation, 50k sample).
   - Separate punctuation and casing models beat the joint mode C model on the joint metric (+2.08; validation, 50k sample).

## Team hypotheses

The team wrote H1–H6 in its execution plan on 2026-10-29, before any CRF result existed; the original text is in [HYPOTHESES.md](HYPOTHESES.md). Each one is reported whatever its outcome.

| # | Hypothesis | Outcome | Evidence |
|---|---|---|---|
| H1 | Joint mode C gives higher PERIOD F1 than mode A | **Not supported for PERIOD** (mixed: higher only for CRF improved), but joint training helps inside the utterance | Paired comparison on mode A gold over the 19,936 mode C validation IDs: each mode A model against the same model type trained on mode C, projected to punctuation. PERIOD F1, A vs C, with the paired per-class difference C − A: HMM 69.31 vs 68.33, −0.98 (−1.95 to 0.04); CRF basic 71.57 vs 71.43, −0.15 (−0.44 to 0.17); CRF improved 72.33 vs 72.63, +0.31 (0.02 to 0.61). The joint CRF improved gains +0.42 macro (0.05 to 0.82), mostly on QUESTION (+1.51). Every joint model gains on non-final positions, and all three intervals exclude 0: HMM +8.69, CRF basic +1.59, CRF improved +1.16. Overall, the joint HMM loses 0.62 (−1.20 to −0.03). **Controlled check:** the mode A model trains on 142,722 sequences and the mode C model on 141,745. Retraining the mode A HMM and CRF basic on exactly the mode C training IDs gives the same picture: non-final +8.68 (7.79 to 9.63) for the HMM and +1.36 (0.89 to 1.86) for the CRF basic. All positions: −0.62 and +0.24 (−0.09 to 0.58). PERIOD: −0.98 (−1.95 to 0.04) and −0.07 (−0.36 to 0.23). The training-set difference moves the all-position, non-final and PERIOD results by at most about 0.2 points, but the CRF basic's QUESTION difference by 0.40 (+1.39 → +0.99; both intervals exclude 0) (`results/scores/h1_joint/*_sameids.json`). The CRF improved, the only model whose PERIOD favours the joint model, was not retrained on the mode C IDs: its full fit peaks at about 14 GiB of memory on Member 3's machine (`memory_diagnostics.peak_private_commit_mb` 14,406.39, in MiB, in `crf_mode_a_improved_full_01.manifest.json`), close to all of the evaluation laptop's 16 GB of RAM, so its +0.31 is within reach of that confound. |
| H2 | Threshold decoding raises rare-class recall | **Not supported** (test; single τ, as written). Prior adjustment is an exploratory follow-up | Recall rises for almost every non-O class, but mostly for COMMA (mode A 45.22 → 52.63; mode B 45.36 → 57.80). Per-class F1 changes, threshold − improved, with paired 95% intervals: COMMA +2.38 (1.89 to 2.88) in A, +2.52 (1.88 to 3.22) in B, and `COMMA|LOWER` +2.79 (2.15 to 3.46) in C. PERIOD falls overall (A −0.76, B −1.59) but rises on non-final positions (+1.43, +1.60). Every rare-class interval includes 0 (QUESTION, COLON, SEMICOLON, DASH, EXCLAMATION). Macro F1 gains +0.51 / +0.41 / +0.70 (sections 1–2; the τ grid is in section 7). **Exploratory follow-up, prior adjustment** (γ = 0.5, section 14.3, added after the first test report), per-class F1 − improved with paired 95% intervals: QUESTION +4.00 (1.88 to 6.13) in A; QUESTION +4.60 (2.63 to 6.50), EXCLAMATION +3.60 (1.21 to 6.33) and DASH +1.70 (0.16 to 3.23) in B, while SEMICOLON and COLON intervals include 0. PERIOD pays for it: −1.79 (−2.20 to −1.39) in A and −1.59 (−1.87 to −1.31) in B. |
| H3 | Question cues raise QUESTION F1 | **Supported** (validation ablation) | Removing the question cues changes QUESTION F1 by −4.05 (−5.37 to −2.99) in A and −2.85 (−4.45 to −1.58) in B, and `QUESTION|LOWER` F1 by −5.39 (−6.81 to −4.10) in C. These are paired per-class intervals (`compare --class`). Primary intervals exclude 0 in every mode (section 6). Question-lookback features (section 14.4) add +6.11 / +7.12 / +6.01 QUESTION F1 on validation. |
| H4 | Filler and discourse cues raise COMMA F1 | **Partly supported** for fillers, **not supported** for discourse markers (validation) | Adding the filler cue (*um*, *uh*, *er*, *erm*, *like*, *you know*, *i mean*, repeated words) raises COMMA F1 by +0.14 (−0.00 to 0.28) in A and +0.21 (0.06 to 0.37) in B, and `COMMA|LOWER` F1 by +0.25 (0.05 to 0.45) in C. Its primary macro F1 does not change (+0.04 / +0.04 / −0.28, all intervals include 0). Removing the clause cues (conjunctions and discourse markers) changes COMMA F1 by +0.03 (−0.08 to 0.14) in A and +0.05 (−0.08 to 0.18) in B, and `COMMA|LOWER` F1 by −0.18 (−0.34 to −0.03) in C. |
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

### Sensitivity to the CRF settings (validation, mode A)

Member 3's settings were fixed, not tuned. To check whether the conclusions depend on them, both presets were retrained on the same 50,000-sequence sample as the ablations (`train_crf.py train --ids sample.ids --c1 --c2 --max-iterations`). They were scored on the primary validation list only; test was not used. Primary macro F1 %, and improved − basic with paired 95% CI (* = interval excludes 0):

| c1 = c2 | Iterations | CRF basic | CRF improved | Improved − basic |
|---|---|---|---|---|
| 0.01 | 100 | 42.30 | 46.67 | +4.37 (3.68 to 5.07) * |
| 0.1 | 100 (delivered setting) | 43.58 | 49.96 | +6.38 (5.71 to 7.07) * |
| 1.0 | 100 | 44.56 | 50.36 | +5.79 (5.13 to 6.44) * |
| 0.01 | 300 | 41.76 | 47.63 | +5.87 (5.23 to 6.50) * |
| 0.1 | 300 | 43.60 | 50.14 | +6.53 (5.90 to 7.18) * |
| 1.0 | 300 | 44.64 | 50.28 | +5.65 (5.06 to 6.25) * |

The improved preset beats the basic one at every setting, by +4.37 to +6.53 points. The delivered setting (0.1, 100 iterations) is within 0.40 points of the best improved score on this sample, so leaving the settings untuned costs little. Files: `results/scores/sensitivity/`.

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

No primary interval excludes 0, so the filler cue does not move macro F1. The class it targets does move: COMMA F1 rises by +0.14 (−0.00 to 0.28) in A and +0.21 (0.06 to 0.37) in B, and `COMMA|LOWER` F1 by +0.25 (0.05 to 0.45) in C (paired per-class intervals, `results/scores/per_class/`). The gain is small but consistent.

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

## 11. Label granularity: trained on 8 classes, scored on 4 (validation)

Each mode B model was projected to mode A with the approved D5 mapping (`scripts/project_joint.py --from B`). It was compared with its mode A twin on mode A validation gold, over the same 20,040 primary IDs: B projection − A model, points, paired 95% CI.

| Model | Mode A model | Mode B projected | Difference | non_final difference |
|---|---|---|---|---|
| HMM | 36.54 | 36.64 | +0.10 (−0.00 to 0.22) | +0.48 (0.34 to 0.65) |
| CRF basic | 44.54 | 44.30 | −0.24 (−0.44 to −0.04) | −0.15 (−0.40 to 0.12) |
| CRF improved | 51.18 | 51.44 | +0.25 (0.05 to 0.47) | +0.39 (0.09 to 0.72) |

Training on the finer 8-class labels barely changes 4-class punctuation: the differences are under 0.3 points and go both ways. Per class, the improved CRF trained on mode B gains on QUESTION (+0.90, 0.36 to 1.52) and loses a little on COMMA (−0.25, −0.40 to −0.10). The extended labels change the basic task by at most a quarter of a point either way; the CRF basic loss of 0.24 has an interval that excludes 0. Files: `results/scores/b_to_a/`.

## 12. Gold-noise check (mode B)

The protocol lists four known sources of noise in the gold labels. `scripts/find_gold_noise.py` flags the three that a rule can detect (leftover speaker labels, subtitle cue residue and the two auto-caption-style videos) on the primary mode B lists (rules in `src/punct_restore/data/gold_noise.py`); the fourth, errors in about 1% of source transcripts, is an unmeasured estimate with no rule. The frozen release was not changed. The counts are an upper estimate: a few titles and headings at label positions also match the speaker-label rule (for example "Spooky Success:" in validation).

| Rule | Validation | Test | test_amb |
|---|---|---|---|
| Leftover speaker label: a name before ':' at the utterance start, after a sentence end or after a turn dash; after ', ' only a name that is a label in at least 3 records of the same video; a numbered role ('SPEAKER 2:') at any position | 134 | 22 | 63 |
| Subtitle cue residue (block number and timestamp in the original text) | 1 | 8 | 0 |
| Auto-caption-style video (`732fql_jhJQ`, `JNxcznsrRb8`) | 135 | 46 | 0 |
| Records excluded (any rule) | 270 | 76 | 63 |
| Speaker-label COLONs / all gold COLONs | 144 / 425 (33.88%) | 23 / 166 (13.86%) | 76 / 273 (27.84%) |

In `default_train`, 319 of 3,385 COLON labels (9.42%) sit on speaker labels, so the models also learn the pattern. The flagged IDs with their rules are in `results/scores/gold_noise/{validation,test,test_amb}_B_noisy.tsv`, and the counts in `*_B_summary.json`.

Mode B was rescored without every flagged record. Macro F1 % (COLON F1), on the primary list and then without the noisy records:

| Model | Validation | Without noise | Test | Without noise | test_amb | Without noise |
|---|---|---|---|---|---|---|
| HMM | 18.47 (15.29) | 16.43 (0.75) | 16.79 (6.73) | 15.97 (1.12) | 15.32 (11.41) | 15.01 (9.05) |
| CRF basic | 22.22 (19.65) | 19.70 (1.42) | 20.25 (6.42) | 19.51 (1.07) | 19.44 (9.76) | 18.80 (5.08) |
| CRF improved | 25.10 (15.69) | 23.27 (2.14) | 22.82 (7.84) | 22.08 (2.30) | 22.59 (9.88) | 22.07 (6.01) |
| CRF improved + τ | — | — | 23.23 (7.89) | 22.45 (2.04) | 23.16 (11.83) | 22.38 (6.18) |

COLON support falls from validation 425 → 245; test 166 → 143; test_amb 273 → 192.

Findings:
- **On validation and test, most of the models' COLON skill comes from speaker labels.** Without the flagged records, COLON F1 falls to 0.75–2.30 there. On test_amb it falls less, from 9.76–11.83 to 5.08–9.05, so each model keeps 52–79% of its COLON F1.
- **Macro F1 falls by 0.31–2.53 points**, and the model ranking is the same in every list (HMM < CRF basic < CRF improved < CRF improved + τ).
- **The mode B scores elsewhere in this report include this noise.** Read their COLON F1 as an upper bound; the other extended marks (EXCLAMATION, SEMICOLON, DASH) remain hard to predict from text alone.

## 13. Simulated ASR noise (test)

This check is **simulated** and **exploratory** (added after the first test report; nothing was tuned on it); RECAP has no real ASR condition. `scripts/simulate_asr_noise.py` corrupts the primary test inputs at a word error rate of 5, 10 and 15%, with seed 2026. The edits are taken round-robin by type:
- deletion of a function word, whose punctuation label moves to the previous word if that word is O, and is dropped otherwise;
- homophone substitution (their/there, your/you're, its/it's, to/too/two, then/than);
- an adjacent swap of two non-final words, counted as two errors.

Every mode gets the same edits, and the mode C casing masks are remapped through them. Each frozen model then predicted each noisy test set once. Nothing was retuned. Primary macro F1 %; the 15% column shows the 95% CI:

| Mode | Model | Clean (0%) | 5% | 10% | 15% |
|---|---|---|---|---|---|
| A | HMM | 35.12 | 35.12 | 35.16 | 35.16 (32.83–37.35) |
| A | CRF basic | 43.65 | 42.94 | 42.29 | 41.68 (39.81–43.66) |
| A | CRF improved | 49.92 | 48.13 | 46.64 | 45.42 (43.56–47.41) |
| B | HMM | 16.79 | 16.79 | 16.85 | 16.85 (15.36–18.50) |
| B | CRF basic | 20.25 | 19.93 | 19.65 | 19.28 (17.94–20.98) |
| B | CRF improved | 22.82 | 22.03 | 21.44 | 20.87 (19.48–22.37) |
| C | HMM | 37.41 | 37.26 | 37.19 | 37.09 (34.25–40.41) |
| C | CRF basic | 43.85 | 43.08 | 42.36 | 41.83 (39.97–43.47) |
| C | CRF improved | 47.02 | 45.97 | 45.00 | 44.12 (42.13–45.98) |

What the table shows:
- **Under v1, the HMM does not move.** It mostly predicts the utterance-final mark, which word errors inside the utterance do not touch.
- **In mode A, the improved CRF loses about 1.5 points per 5% of word errors (49.92 → 45.42), and the basic CRF about 0.65 (43.65 → 41.68).** The improved CRF's word-pair features depend on the exact words, so it drops faster.
- **The ranking holds at every noise level:** CRF improved > CRF basic > HMM in every mode, and at 15% the intervals of the improved CRF and the HMM still do not overlap.

Files: `results/scores/asr_noise/wer<rate>/`, including `summary.json` with the achieved WER and the edit counts.

### 13.1 A more realistic edit mix, with seed variance (v2)

[ASR_NOISE_V2.md](ASR_NOISE_V2.md) reruns this check with a substitution-dominant edit model (`--edit-model v2`): 45.00% substitution, 30.00% deletion, 15.00% insertion and 10.00% swaps, against v1's 50% swaps. Substitutions are edit-distance-1 words from the `default_train` vocabulary plus the homophones, and insertions are fillers and short function words. Seeds 2026, 2027, 2028; nothing was retuned. Primary macro F1 %, clean → mean at 15% WER (min–max over the seeds):

| Mode | HMM | CRF basic | CRF improved |
|---|---|---|---|
| A | 35.12 → 33.82 (33.75–33.87) | 43.65 → 39.30 (39.13–39.51) | 49.92 → 43.46 (43.10–43.88) |
| B | 16.79 → 15.94 (15.84–15.99) | 20.25 → 17.97 (17.72–18.26) | 22.82 → 19.73 (19.50–20.02) |
| C | 37.41 → 30.43 (30.14–30.92) | 43.85 → 33.23 (33.10–33.46) | 47.02 → 35.93 (35.43–36.84) |

What changes from v1:
- **The HMM now degrades too:** −1.31 in A, −0.86 in B and −6.98 in C. Half of v1's budget was swaps of non-final words, which leave the utterance-final word in place.
- **The improved CRF still drops most in every mode:** −6.47 / −3.09 / −11.09 in A / B / C.
- **The ranking CRF improved > CRF basic > HMM holds for every seed, WER and mode.**
- **Seed variance is small:** the min–max spread over the three seeds is 0.02–1.60 points per cell.

Files: `results/scores/asr_noise_v2/` (`summary.json` and one folder per edit model, WER and seed).

## 14. Team comparison methods

**Exploratory.** These methods were added on 2026-10-30, after the first test report, in response to a review that cited test results. Their settings were still chosen on validation only, and each test list was predicted once. The team's hypotheses H1–H6 are judged on the course models; these results are follow-ups.

### 14.1 Gap classifier: the CRF without its transitions (validation)

`GapClassifier` is a multinomial logistic regression on exactly the improved CRF's per-token features, with each token predicted independently. It was trained on the same 50k sample with the same features as the ablation CRF (`run_ablations.py --variants gap_improved`). It also differs in training: scikit-learn logistic regression with an L2 penalty, C = 1.0 and up to 200 lbfgs iterations, against the CRF's elastic net c1 = c2 = 0.1 and 100 iterations. The difference is gap − CRF, in points with paired 95% CI (* = interval excludes 0):

| Mode | CRF improved | Gap classifier | Difference | non_final difference |
|---|---|---|---|---|
| A | 49.96 | 48.69 | −1.27 (−1.93 to −0.66) * | −2.88 (−3.70 to −2.08) * |
| B | 23.25 | 21.54 | −1.70 (−3.77 to −0.17) * | −1.49 (−3.58 to 0.14) |
| C | 47.70 | 42.96 | −4.74 (−6.06 to −3.60) * | −3.20 (−4.06 to −2.42) * |

Transitions plus these training differences are worth 1.3–1.7 points in A and B, and 4.7 points in mode C. The large mode C gap fits the transitions carrying the capital after a sentence end, the mechanism behind H1, but this comparison cannot separate the two causes.

### 14.2 Hidden-event language model (classic comparison method)

This is a Stolcke-style hidden-event n-gram LM, `HiddenEventLM`:
- Punctuation marks are hidden event tokens between words.
- Exact Viterbi picks the most likely events under an interpolated absolute-discounting n-gram model.
- In mode C a unigram truecaser adds casing, so mode C is a factored pipeline.

It was trained on all of `default_train`. Order ∈ {2, 3, 4} and discount ∈ {0.5, 0.75, 0.9} were chosen on validation; the best settings are below, and the full grid is in `results/scores/hidden_event_lm/`. Test was predicted once with the chosen setting. Differences are paired 95% CIs:

| Mode | Chosen setting | Validation | Test (95% CI) | Test − HMM | Test − CRF improved |
|---|---|---|---|---|---|
| A | order 4, discount 0.9 | 50.41 | 48.38 (46.21–50.62) | +13.26 (12.12 to 14.44) * | −1.54 (−2.28 to −0.73) * |
| B | order 4, discount 0.75 | 25.07 | 22.51 (21.08–23.98) | +5.71 (4.98 to 6.57) * | −0.31 (−1.05 to 0.50) |
| C | order 4, discount 0.75 | 46.31 | 41.18 (39.42–42.70) | +3.77 (0.55 to 6.23) * | −5.84 (−7.19 to −4.45) * |

Results on test:
- The LM beats the HMM in every mode.
- In mode A it is 1.54 points behind the improved CRF, and in mode B the two are statistically tied.
- In mode C it is 5.8 points behind, because the unigram truecaser cannot use context.

test_amb shows the same picture (`results/scores/hidden_event_lm/`).

### 14.3 Prior-adjusted threshold decoding (τ × γ)

This extends section 7. After the gate 1 − P(O) > τ fires, the label is `argmax_c P(c|x) / P_train(c)^γ`, which divides each class's probability by its training frequency raised to γ. τ (0.05 to 0.95 in steps of 0.05, as in section 7) and γ ∈ {0, 0.25, 0.5, 0.75, 1} were chosen jointly on validation, with γ = 0 reproducing section 7; the test lists were then predicted once. Differences are paired 95% CIs, against the plain improved CRF and against the plain threshold of section 7:

| Mode | Chosen τ, γ | Validation | Test | Test − improved | Test − plain τ | test_amb − improved | test_amb − plain τ |
|---|---|---|---|---|---|---|---|
| A | 0.4, 0.5 | 53.50 | 51.34 | +1.42 (0.68 to 2.16) * | +0.91 (0.24 to 1.57) * | +2.08 (1.45 to 2.66) * | +1.19 (0.65 to 1.72) * |
| B | 0.45, 0.5 | 26.52 | 24.25 | +1.43 (0.52 to 2.22) * | +1.03 (0.10 to 1.80) * | +1.55 (0.40 to 2.63) * | +0.98 (0.11 to 1.75) * |
| C | 0.35, 0.5 | 53.16 | 49.77 | +2.74 (1.07 to 4.68) * | +2.04 (0.52 to 3.87) * | +1.15 (0.39 to 1.92) * | −0.13 (−0.76 to 0.46) |

Per class on test, prior adjustment − improved (paired 95% CI, `compare --class`):
- **A:** QUESTION +4.00 (1.88 to 6.13) *; COMMA +2.04 (1.62 to 2.46) *; PERIOD −1.79 (−2.20 to −1.39) *.
- **B:** QUESTION +4.60 (2.63 to 6.50) *; EXCLAMATION +3.60 (1.21 to 6.33) *; DASH +1.70 (0.16 to 3.23) *; SEMICOLON +1.28 (0.00 to 2.70); COLON −0.13 (−4.96 to 3.98); COMMA +0.57 (0.25 to 0.90) *; PERIOD −1.59 (−1.87 to −1.31) *.
- **C:** `QUESTION|FIRST_UPPER` +8.18 (2.28 to 13.78) *; `QUESTION|LOWER` +2.73 (0.56 to 4.79) *; `COMMA|LOWER` +2.35 (1.75 to 2.99) *; `PERIOD|LOWER` −2.09 (−2.58 to −1.61) *.
- **The mode C gain leans on two tiny classes.** `QUESTION|ALL_UPPER` (9 test tokens) and `PERIOD|ALL_UPPER` (125) change by +10.53 (0.00 to 30.78) and +6.89 (−0.80 to 13.72). With each of the 12 classes weighted equally, they account for 1.45 of the +2.74 gain.

Against the plain threshold, the gain comes from the question mark and, in B, the exclamation mark, at a cost on the frequent marks: QUESTION +4.09 (2.13 to 6.04) * and PERIOD −1.03 (−1.18 to −0.86) * in A; QUESTION +4.84 (3.02 to 6.66) *, EXCLAMATION +3.67 (1.32 to 6.45) * and COMMA −1.95 (−2.47 to −1.49) * in B.

Prior adjustment is the best decoding of the improved CRF on test in every mode, and on test_amb in A and B; on test_amb mode C it ties the plain threshold. The per-class and plain-τ comparisons are exploratory and are not in the Holm families.

### 14.4 Question-lookback and tag-question features (validation)

This extension is off by default (`FeatureConfig.improved(use_question_lookback=True)`). It adds two features: a wh-word followed by an auxiliary anywhere in the previous 12 tokens, and tag-question cues ("…, right?", "…n't it?"). It was trained on the same 50k sample as the ablations (`--variants improved_plus_question2`). The difference is with − without, in points with paired 95% CI:

| Mode | Improved | + question2 | Difference | non_final difference | QUESTION F1 difference |
|---|---|---|---|---|---|
| A | 49.96 | 52.00 | +2.04 (1.59 to 2.56) * | +1.22 (0.64 to 1.87) * | `QUESTION` +6.11 (4.86 to 7.58) * |
| B | 23.25 | 24.19 | +0.94 (0.74 to 1.18) * | +0.68 (0.42 to 0.99) * | `QUESTION` +7.12 (5.76 to 8.67) * |
| C | 47.70 | 49.77 | +2.07 (0.23 to 3.87) * | +1.35 (0.60 to 2.11) * | `QUESTION\|LOWER` +6.01 (4.30 to 7.83) * |

Question lookback is the largest single feature gain we found: +6.11 / +7.12 / +6.01 QUESTION F1 in modes A / B / C, with every interval excluding 0. It was tested at the 50k-sample scale on validation only, and it is not in the delivered models.

### 14.5 Factored vs joint punctuation and casing (validation, mode C)

This is decision D9. The joint mode C improved CRF is compared with a factored pipeline: a mode A improved CRF for punctuation plus a casing-only improved CRF (helper mode `CASE`), combined by `scripts/factor_predictions.py`. All three CRFs train on the same 50k mode C sample with the same features (`run_ablations.py --mode C --variants improved,case_improved,factored`). The difference is factored − joint, paired 95% CI:

| Block | Joint | Factored | Difference |
|---|---|---|---|
| 12-class joint (primary) | 47.70 | 49.78 | +2.08 (1.22 to 2.94) * |
| punctuation, all positions | 50.55 | 50.40 | −0.15 (−0.52 to 0.25) |
| punctuation, non_final | 36.57 | 35.75 | −0.83 (−1.36 to −0.28) * |
| casing (masked) | 85.33 | 86.82 | +1.49 (0.78 to 2.17) * |
| casing, non-initial | 82.35 | 83.84 | +1.49 (0.79 to 2.19) * |

Findings:
- **The factored pipeline wins on the joint metric (+2.08)**, because a dedicated casing CRF is better at casing (+1.49).
- **Punctuation over all positions is tied.**
- **The joint model keeps an advantage on punctuation inside the utterance (−0.83 for the factored pipeline)**, consistent with H1.

Joint training helps punctuation inside the utterance; a separate model helps casing.

## Multiple comparisons (Holm)

The intervals in this document are not adjusted for the number of comparisons. As a check, two-sided bootstrap p-values were taken from each paired comparison's share of resamples in which B beats A (resolution 0.001), and Holm's step-down correction was applied at α = 0.05 within two families.

- **Test comparisons (21):** improved − basic, basic − HMM and improved − HMM in each mode; threshold − improved; prior-adjusted threshold − improved; and the hidden-event LM against the HMM and the improved CRF.
  - All nine model comparisons and all three prior-adjusted threshold gains survive.
  - The plain threshold survives in A and C, but not B (p ≤ 0.100).
  - The hidden-event LM's advantage over the HMM survives in A and B, but not C (p ≤ 0.028). Its deficit to the improved CRF survives in A and C; in B the two are tied (p ≤ 0.486).
- **Ablations (18):** six feature groups in three modes, validation.
  - Survive: removing word pairs (A, B), the ±2 window (A, B) or the question cues (A, B).
  - Do not survive: everything in mode C (word pairs p ≤ 0.096, window p ≤ 0.042, question p ≤ 0.032) and the small mode B effects of the clause cues and position buckets.

The p-values are recomputed from the committed compare JSONs in `results/scores/crf_test/`, `results/scores/threshold/`, `results/scores/threshold_prior/`, `results/scores/hidden_event_lm/` and `results/scores/ablations/`.

## Limitations

- **CRF settings were not tuned.** Every model uses c1 = c2 = 0.1 and 100 L-BFGS iterations, and every fit stopped at the iteration cap. The loss was still falling at the cap: by 0.12–1.52% over the last 10 iterations, more for the improved models (0.86 / 0.63 / 1.52% in A / B / C) than for the basic ones (0.12 / 0.21 / 0.53%), so the iteration cap works against the improved − basic gain (`results/scores/crf_convergence/convergence.json`). Only the decoding was tuned: τ (section 7), and τ with γ (section 14.3). The validation sensitivity check in section 5 shows the improved − basic gap holds (+4.37 to +6.53) across six settings, but the absolute scores could move a little with tuning.
- **Intervals cover sample variance only.** Each setting has one fit. The intervals are not adjusted for the number of comparisons; the Holm check above shows which conclusions survive.
- **Ablation and filler models use a 50k-sequence sample.** Read only their differences; each comes from one fit.
- **Rare classes stay low.** In mode B the improved CRF scores under 8 F1 on COLON, DASH, EXCLAMATION and SEMICOLON on test. `QUESTION|ALL_UPPER` has 9 test tokens.
- **Gold noise.** Leftover speaker labels give false COLON labels: about a third of the validation COLONs and 9.42% of the training COLONs (section 12). Mode B COLON scores are an upper bound.
- **Training environment.** Fit times, memory and the Python version come from Member 3's training manifests and were not re-measured.
- **test and test_amb share 25 videos.** They are reported separately and never pooled.
- **The comparison methods of sections 13–14 are post hoc.** They were added after the first test report, in response to a review that cited test results. Their settings were chosen on validation and each test list was predicted once, but they are exploratory, and the hypotheses H1–H6 are judged on the course models.
- **Some comparison methods were run on validation only, at the 50k-sample scale:** the gap classifier, the question-lookback features and the factored pipeline (section 14). Their differences are controlled; their absolute scores sit below the full-data models.
- **Section 8 is supplementary.** It scores the frozen models on additional official-test utterances; nothing was selected on it.

## Reproduce

Run every command from the repository root. `<rel>` is `sponspeech-m1-v2`. The shared models are listed in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

- **Test predictions, once per model:** `python scripts/train_crf.py predict --model crf_mode_<m>_<preset>_full_01.joblib --input data/processed/<rel>/mode_<M>/test.jsonl --ids data/splits/<rel>/primary_eval/test_mode_<M>.ids --output PRED.jsonl --manifest-out PRED.manifest.json`. The same command on validation reproduces Member 3's delivered files byte for byte.
- **Scores and paired comparisons:** `python scripts/evaluate_models.py score|compare ... --bootstrap 1000 --seed 2026`, and `errors ... --per-pair 3` for the error pairs. Mode C needs `--masks data/processed/<rel>/masks/case_joint/<split>.jsonl`.
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
