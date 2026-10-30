# Evaluation results: HMM baseline (Member 5)

This is the Member 5 deliverable for the HMM baseline (PR #4) and the shortcut baseline: confusion matrices, rare-label and class-imbalance analysis, error pairs, sequence-level errors and a runtime measurement. It follows [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md). The CRF models (Member 3) are evaluated in [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) (test, test_amb, validation selection, ablations and extensions).

## Provenance

Every number below comes from an evaluator JSON file in [`results/scores/`](../results/scores/README.md).

**Data:** `sponspeech-m1-v2`:
- processed `SHA256SUMS` fingerprint `19ed874f…`;
- splits `40ea6371…`;
- primary ID lists only;
- case_joint masks for mode C.

**Models:**
- **HMM:** Raşit's tuned bundle `hmm-sponspeech-v2-20261029` (PR #4):
  - selected settings A/B smoothing 0.01, cutoff 1; C smoothing 0.01, cutoff 2;
  - settings chosen on validation.
- **Shortcut baseline:** `evaluate_models.py baseline`. It predicts O everywhere except the last token, which gets QUESTION after a wh-word or auxiliary first word and PERIOD otherwise. In mode C it puts FIRST_UPPER on the first word.

**Scoring:**
- `scripts/evaluate_models.py` at `9155a7e`;
- CIs are a grouped bootstrap over videos (`metadata.group_id`), 1,000 resamples, seed 2026, 95% percentile intervals, not adjusted for multiple comparisons;
- the HMM's numbers equal those in its own bundle.

**Raw outputs** stay under the ignored `results/` path, per `results/README.md`, and are shared as a bundle. Every table below was generated from the evaluator's score, compare and errors JSON files. No number was typed by hand.

**Metric:**

| Mode | Primary metric |
|---|---|
| A | macro F1 over COMMA, PERIOD, QUESTION (O excluded) |
| B | macro F1 over the 7 non-O punctuation classes |
| C | macro F1 over all 12 joint `PUNCT|CASE` classes, including the 3 `O|case` classes, on unmasked positions |

Test is the result. Validation selected the settings. `test_amb` shares 25 videos with test, so never pool the two.

## Main findings

1. **The HMM beats the shortcut in every mode** (test, paired differences):

   | Mode | HMM − shortcut |
   |---|---|
   | A | +7.64 (5.41 to 9.81) |
   | B | +4.92 (3.26 to 6.63) |
   | C | +19.32 (16.55 to 22.38) |

2. **Most of the HMM's punctuation score sits on the utterance-final token.** Without that token (`non_final`), it scores A 9.13, B 5.51, C 17.42.
   - It recalls 12.12% of utterance-final questions, but only 0.10% of questions inside an utterance.
3. **The HMM under-predicts commas.**
   - Mode A test: COMMA recall is 16.91% (precision 58.94%).
   - Missed commas (COMMA → O) are 63% of its validation errors.
4. **Mode B's rare classes are not learned.**
   - EXCLAMATION, COLON, SEMICOLON and DASH make up 2.4% of non-O test labels, and each scores below 7 F1.
   - The HMM predicts DASH 2,674 times against 784 gold. Almost all of these predictions fall on words unseen in training: smoothing decides the label for unknown words.
5. **Errors grow with utterance length:** micro F1 falls from 68.48 (1–10 tokens) to 31.22 (more than 50 tokens), because longer utterances hold more inner punctuation.
6. **Casing (mode C)** is much easier than punctuation: casing F1 79.81 unmasked, 76.42 without the first word.

## 1. Primary metric (macro F1 %, 95% CI) and micro F1 on test

**Mode A** (`punctuation/all`)

| Model | Validation | Test | Test_amb | Test micro F1 |
|---|---|---|---|---|
| shortcut | 28.21 (26.89–29.48) | 27.48 (25.55–29.38) | 23.90 (22.26–25.56) | 33.72 |
| HMM | 36.54 (34.80–38.01) | 35.12 (32.80–37.31) | 31.50 (29.36–33.66) | 44.03 |

**Mode B** (`punctuation/all`)

| Model | Validation | Test | Test_amb | Test micro F1 |
|---|---|---|---|---|
| shortcut | 12.14 (11.58–12.68) | 11.87 (11.02–12.68) | 10.31 (9.61–11.01) | 33.72 |
| HMM | 18.47 (16.32–21.35) | 16.79 (15.28–18.49) | 15.32 (13.85–16.72) | 42.20 |

**Mode C** (`joint/unmasked`)

| Model | Validation | Test | Test_amb | Test micro F1 |
|---|---|---|---|---|
| shortcut | 18.53 (18.05–18.98) | 18.09 (17.34–18.76) | 16.86 (16.23–17.49) | 83.30 |
| HMM | 40.89 (38.72–42.93) | 37.41 (34.56–40.71) | 36.34 (34.92–37.53) | 86.80 |

## 2. Shortcut reliance (test macro F1 %)

`non_final` drops each utterance's last token; `casing/unmasked_non_initial` drops position 0.

| Model | A non_final | B non_final | C non_final | C casing unmasked | C casing non-initial |
|---|---|---|---|---|---|
| shortcut | 0.00 (0.00–0.00) | 0.00 (0.00–0.00) | 0.00 (0.00–0.00) | 47.09 (46.10–48.04) | 32.00 (31.90–32.11) |
| HMM | 9.13 (7.97–10.18) | 5.51 (4.18–6.93) | 17.42 (15.93–19.11) | 79.81 (77.03–82.46) | 76.42 (73.54–79.24) |

## 3. Per-class results and class imbalance (test, `punctuation/all`)

Share = the class's share of all non-O gold labels. P/R/F1 in %; pred = predicted count.

**Mode A**

| Class | Support | Share | shortcut P/R/F1 (pred) | HMM P/R/F1 (pred) |
|---|---|---|---|---|
| COMMA | 30,103 | 57.0% | 0.00/0.00/0.00 (0) | 58.94/16.91/26.28 (8,638) |
| PERIOD | 20,570 | 38.9% | 91.80/52.35/66.68 (11,731) | 89.84/53.85/67.34 (12,330) |
| QUESTION | 2,175 | 4.1% | 36.95/10.02/15.77 (590) | 52.17/6.62/11.75 (276) |

**Mode B**

| Class | Support | Share | shortcut P/R/F1 (pred) | HMM P/R/F1 (pred) |
|---|---|---|---|---|
| COMMA | 29,153 | 55.2% | 0.00/0.00/0.00 (0) | 60.11/17.04/26.56 (8,267) |
| PERIOD | 20,246 | 38.3% | 91.80/53.19/67.35 (11,731) | 89.85/54.68/67.99 (12,320) |
| QUESTION | 2,175 | 4.1% | 36.95/10.02/15.77 (590) | 55.38/6.62/11.83 (260) |
| EXCLAMATION | 204 | 0.4% | 0.00/0.00/0.00 (0) | 2.01/3.43/2.54 (348) |
| COLON | 166 | 0.3% | 0.00/0.00/0.00 (0) | 16.67/4.22/6.73 (42) |
| SEMICOLON | 120 | 0.2% | 0.00/0.00/0.00 (0) | 1.23/0.83/1.00 (81) |
| DASH | 784 | 1.5% | 0.00/0.00/0.00 (0) | 0.60/2.04/0.93 (2,674) |

**Mode C**

| Class | Support | Share | shortcut P/R/F1 (pred) | HMM P/R/F1 (pred) |
|---|---|---|---|---|
| COMMA | 30,060 | 57.0% | 0.00/0.00/0.00 (0) | 60.99/14.91/23.96 (7,347) |
| PERIOD | 20,526 | 38.9% | 91.81/52.28/66.63 (11,690) | 71.06/62.57/66.55 (18,074) |
| QUESTION | 2,168 | 4.1% | 36.86/9.96/15.69 (586) | 39.66/7.52/12.64 (411) |

## 4. Confusion matrices (test)

Mode C is shown as its punctuation and casing projections; the 12 × 12 joint matrix is in each score JSON (`blocks.joint/unmasked.confusion`).

### Mode A

*shortcut, punctuation* (rows = gold, columns = predicted)

| gold \ pred | O | COMMA | PERIOD | QUESTION |
|---|---|---|---|---|
| O | 354,902 | 0 | 0 | 0 |
| COMMA | 30,103 | 0 | 0 | 0 |
| PERIOD | 9,429 | 0 | 10,769 | 372 |
| QUESTION | 995 | 0 | 962 | 218 |

*HMM, punctuation* (rows = gold, columns = predicted)

| gold \ pred | O | COMMA | PERIOD | QUESTION |
|---|---|---|---|---|
| O | 351,852 | 2,892 | 147 | 11 |
| COMMA | 24,933 | 5,091 | 69 | 10 |
| PERIOD | 8,778 | 604 | 11,077 | 111 |
| QUESTION | 943 | 51 | 1,037 | 144 |

### Mode B

*shortcut, punctuation* (rows = gold, columns = predicted)

| gold \ pred | O | COMMA | PERIOD | QUESTION | EXCLAMATION | COLON | SEMICOLON | DASH |
|---|---|---|---|---|---|---|---|---|
| O | 354,902 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| COMMA | 29,153 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| PERIOD | 9,105 | 0 | 10,769 | 372 | 0 | 0 | 0 | 0 |
| QUESTION | 995 | 0 | 962 | 218 | 0 | 0 | 0 | 0 |
| EXCLAMATION | 204 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| COLON | 166 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SEMICOLON | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DASH | 784 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

*HMM, punctuation* (rows = gold, columns = predicted)

| gold \ pred | O | COMMA | PERIOD | QUESTION | EXCLAMATION | COLON | SEMICOLON | DASH |
|---|---|---|---|---|---|---|---|---|
| O | 349,790 | 2,635 | 144 | 4 | 233 | 25 | 58 | 2,013 |
| COMMA | 23,559 | 4,969 | 62 | 2 | 81 | 5 | 7 | 468 |
| PERIOD | 8,356 | 539 | 11,070 | 109 | 17 | 2 | 9 | 144 |
| QUESTION | 930 | 48 | 1,036 | 144 | 1 | 1 | 1 | 14 |
| EXCLAMATION | 153 | 35 | 2 | 0 | 7 | 1 | 0 | 6 |
| COLON | 132 | 5 | 1 | 0 | 6 | 7 | 3 | 12 |
| SEMICOLON | 114 | 4 | 0 | 0 | 0 | 0 | 1 | 1 |
| DASH | 724 | 32 | 5 | 1 | 3 | 1 | 2 | 16 |

### Mode C

*shortcut, punctuation* (rows = gold, columns = predicted)

| gold \ pred | O | COMMA | PERIOD | QUESTION |
|---|---|---|---|---|
| O | 354,257 | 0 | 0 | 0 |
| COMMA | 30,060 | 0 | 0 | 0 |
| PERIOD | 9,424 | 0 | 10,732 | 370 |
| QUESTION | 994 | 0 | 958 | 216 |

*shortcut, casing (unmasked)* (rows = gold, columns = predicted)

| gold \ pred | LOWER | FIRST_UPPER | ALL_UPPER |
|---|---|---|---|
| LOWER | 364,043 | 91 | 0 |
| FIRST_UPPER | 29,285 | 12,151 | 0 |
| ALL_UPPER | 940 | 22 | 0 |

*HMM, punctuation* (rows = gold, columns = predicted)

| gold \ pred | O | COMMA | PERIOD | QUESTION |
|---|---|---|---|---|
| O | 349,936 | 2,396 | 1,812 | 113 |
| COMMA | 23,302 | 4,481 | 2,246 | 31 |
| PERIOD | 7,150 | 428 | 12,844 | 104 |
| QUESTION | 791 | 42 | 1,172 | 163 |

*HMM, casing (unmasked)* (rows = gold, columns = predicted)

| gold \ pred | LOWER | FIRST_UPPER | ALL_UPPER |
|---|---|---|---|
| LOWER | 360,251 | 3,828 | 55 |
| FIRST_UPPER | 12,572 | 28,836 | 28 |
| ALL_UPPER | 431 | 40 | 491 |

## 5. Error pairs (validation, primary block)

Counts of wrong positions per (gold → predicted) pair; share = share of that model's errors. Examples are drawn at random by the evaluator (seed 2026), not chosen.

**Mode A**

| Model | Errors | Top pairs (count, share) |
|---|---|---|
| HMM | 59,145 | COMMA→O 37,206 (63%); PERIOD→O 13,108 (22%); O→COMMA 4,239 (7%); QUESTION→O 1,545 (3%); QUESTION→PERIOD 1,438 (2%); PERIOD→COMMA 914 (2%) |

**Mode B**

| Model | Errors | Top pairs (count, share) |
|---|---|---|
| HMM | 62,541 | COMMA→O 34,660 (55%); PERIOD→O 12,640 (20%); O→COMMA 3,948 (6%); O→DASH 3,114 (5%); QUESTION→O 1,532 (2%); QUESTION→PERIOD 1,437 (2%) |

**Mode C**

| Model | Errors | Top pairs (count, share) |
|---|---|---|
| HMM | 59,099 | COMMA→O 34,871 (59%); PERIOD→O 10,852 (18%); O→COMMA 3,514 (6%); COMMA→PERIOD 3,142 (5%); O→PERIOD 2,570 (4%); QUESTION→PERIOD 1,681 (3%) |

**Random examples, HMM, mode A, three most frequent pairs** (SponSpeech, CC BY-NC 4.0)

- `sponspeech:dev:EEj137ADBzY-43` position 48, COMMA → O
  - gold: … what i've been talking about that the vast majority of christians do not, do not, interpret that passage the way i'm interpreting it.
  - pred: … what i've been talking about that the vast majority of christians do not do not interpret that passage the way i'm interpreting it.
- `sponspeech:dev:ARYwJXcPpoo-13` position 4, PERIOD → O
  - gold: it could be a mantra. taking correction without defense. happy day, happy day. i like that, taking …
  - pred: it could be a mantra taking correction without defense happy day happy day i like that taking …
- `sponspeech:dev:MmJRbY53RwE-416` position 6, O → COMMA
  - gold: although, will anybody ever get hired again for anything? it's like a very scary time.
  - pred: although will anybody ever get hired again, for anything it's like a very scary time.

## 6. Sequence-level errors (mode A, test)

Micro F1 (%) over COMMA/PERIOD/QUESTION by utterance length in tokens.

| Length | Sequences | shortcut | HMM |
|---|---|---|---|
| 1–10 | 1,009 | 65.71 | 68.48 |
| 11–25 | 5,081 | 50.22 | 57.44 |
| 26–50 | 4,013 | 32.40 | 43.50 |
| > 50 | 2,218 | 17.57 | 31.22 |

QUESTION recall by position (gold: 1,180 utterance-final, 995 inside):

| Model | Final | Inside |
|---|---|---|
| shortcut | 18.47 | 0.00 |
| HMM | 12.12 | 0.10 |

## 7. Runtime (one machine, identical conditions)

Hardware: Intel64 Family 6 Model 158 Stepping 10, GenuineIntel, Python 3.13.1, one process, no other jobs. Prediction = median of 3 passes over the primary validation set after a 200-record warm-up; model loading excluded. HMM fit time is one in-process fit on `default_train`. The HMM's Viterbi decoding grows with the square of the label count (A 4, B 8, C 12), which is why its prediction slows from A to C.

| Model | Mode | Fit s | Load s | Predict s | Tokens/s | ms/utterance |
|---|---|---|---|---|---|---|
| HMM | A | 5.0 | 0.47 | 2.94 | 210,786 | 0.146 |
| HMM | B | 6.3 | 0.20 | 7.46 | 82,916 | 0.372 |
| HMM | C | 5.4 | 0.14 | 13.30 | 46,383 | 0.667 |

## 8. Supplementary: leakage and the native benchmark (HMM, mode A, test)

**Supplementary rows; the primary result is still section 1.** They use the same frozen HMM test predictions, which cover all 17,695 official test utterances. Numbers are macro F1 % with grouped-bootstrap 95% intervals (1,000 resamples).

| Scored set | Gold labels | Utterances | Macro F1 | non_final |
|---|---|---|---|---|
| Official test, all utterances | canonical | 17,695 | 34.98 (33.01–36.83) | 9.06 |
| Leakage-controlled test | canonical | 12,668 | 34.74 (32.46–36.90) | 9.14 |
| Excluded part, all reasons | canonical | 5,027 | 35.40 (31.47–38.42) | 8.63 |
| Excluded: video also in the training set | canonical | 2,706 | 33.54 (29.94–36.84) | — |
| Official test, all utterances | SponSpeech native | 17,695 | 35.10 (33.15–36.95) | 9.19 |
| Primary test (leakage-controlled, mode A policy) | canonical | 12,321 | 35.12 (32.80–37.31) | 9.13 |

- **What leakage is.** The official test shares videos and duplicate texts with train and dev.
  - 5,027 of 17,695 utterances (28.4%) are excluded from the leakage-controlled set.
  - 4,836 of them for leakage; 2,706 of those come from videos in the training set.
  - 191 for a quality filter (low comma density or a TTS source).
- **The HMM barely profits from leakage, unlike the CRF.**
  - On utterances from videos in its training set, the HMM scores 33.54, against 34.74 on the clean ones. A first-order HMM has little context to memorise.
  - On the same utterances, the improved CRF scores 67.09 against 49.50 ([EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §8).
- **Native labels.** Scoring against SponSpeech's own 4-class labels changes the result by +0.13 points. Our canonical label policy is therefore not what drives the HMM's score.

## Limitations

- **QUESTION|ALL_UPPER** has only 9 test tokens, so it can move mode C's 12-class macro by about 2.8 points. Never rank mode C models on a joint difference smaller than its interval.
- **Gold noise** affects the rare classes:
  - leftover speaker labels give 26 false COLON labels in validation;
  - subtitle cue numbers carry gold DASH;
  - one auto-caption-style test video (`JNxcznsrRb8`) lacks commas and sentence ends;
  - about 1% of source transcripts contain errors.
- **Channels overlap** across the original splits (149 of 522). Grouping by video does not guarantee independence from training.
- **Intervals cover test-sample variance only.** There is one deterministic fit per setting.
- **Runtime** was measured on one laptop CPU, single process. Absolute times will differ on other machines; the ratios are what to compare.
- **Abbreviation errors were not analysed separately.** Tokens are lowercased, and Member 1's filters drop records with ambiguous abbreviation boundaries.

## Reproduce

All commands are run from the repository root with the v2 release in `data/`.

```bash
python scripts/evaluate_models.py score --gold data/processed/sponspeech-m1-v2/mode_A/test.jsonl --ids data/splits/sponspeech-m1-v2/primary_eval/test_mode_A.ids --predictions <hmm>/final/mode_A/test/predictions.jsonl --bootstrap 1000 --out results/tables/hmm_A_test.json
```

- **Mode C:** add `--masks data/processed/sponspeech-m1-v2/masks/case_joint/test.jsonl`.
- **Paired comparisons:** `compare --predictions-a A --predictions-b B --bootstrap 1000`.
- **Error pairs:** `errors --per-pair 3`.
- **Report generator and timing script:** shared in the Member 5 results bundle.
- **Section 8 ID lists:** the "all utterances" list is every `sample_id` in `mode_A/test.jsonl`. The "excluded" list is that list minus `splits/sponspeech-m1-v2/leakage_controlled/test.ids`. The native rows use `--gold data/processed/sponspeech-m1-v2/benchmark_native/mode_A/test.jsonl`.
