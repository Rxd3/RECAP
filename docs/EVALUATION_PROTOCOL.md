# Evaluation protocol — v1

Owner: Member 5. Reviewers: Members 1–4 and 6. The core of this protocol is
implemented in `src/punct_restore/evaluation/` and `recap evaluate-models`
(2026-10-29), including grouped bootstrap intervals, paired model comparison,
error-example export, the prediction-manifest check and the data-policy record. This document holds
the rules. The results are in [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) (HMM, shortcut) and
[EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) (CRF, extensions), and the evaluator outputs behind them are
in [`results/scores/`](../results/scores/README.md).

## Data discipline and alignment

Training fits models and train-derived statistics. Validation selects smoothing,
features, hyperparameters, thresholds and any decoding settings. Freeze selected
settings before using the held-out test set for final reporting. Report a test
set change or test-driven iteration honestly and revise the protocol if needed.

**Data release.** The official release is `sponspeech-m1-v2` (policy `m1-v2`, see
`docs/DATA_RELEASE_VALIDATION.md`). Take the processed views and the ID lists from the
same release. The v2 views differ from v1 only in `metadata.policy_version`, but the v2
primary and default lists are smaller. Every report records the gold file's
`metadata.policy_version` (`policy_version` in the JSON, "Data policy" in the Markdown),
and a gold file that mixes versions is rejected.

Within each mode/condition, every model uses the same test IDs, tokenization,
sequence lengths and gold release. Validate prediction/input/gold manifests,
unique complete IDs, declared model/mode, ordering, lengths and label vocabulary
before scoring. Reject mismatches instead of truncating, silently dropping IDs,
or substituting gold. Flatten tokens only after sequence alignment is validated.

## Punctuation reporting

Use the full ordered mode vocabulary even if a class has no test support.
For each class, report token-level precision, recall, F1, gold support and
predicted count. Label names and support accompany every table.

- **Primary:** macro F1 over punctuation classes excluding O: A/C projection use
  COMMA, PERIOD, QUESTION; B uses all seven non-O punctuation classes.
- Also report micro precision/recall/F1 over those non-O classes and class-wise
  scores. An O→COMMA mistake counts as a non-O false positive; a COMMA→O mistake
  counts as a non-O false negative. Do not filter out gold-O tokens before scoring.
- Optionally report all-class macro/micro F1, explicitly labeled **including O**.
- Token accuracy including O is secondary because O may dominate.

### Why non-O macro F1 is the primary metric

Mode A validation numbers illustrate each choice.

- **O dominates, so accuracy compresses the differences between models.**
  - O is 87.06% of tokens (538,697 of 618,771).
  - The shortcut baseline never predicts a comma, yet it reaches 89.98% accuracy against the HMM's 90.44%: a 0.46-point gap.
  - The same two models differ by 8.33 points in non-O macro F1 (28.21 vs 36.54).
- **Leaving O out of the average rewards restored marks.**
  - O errors still count: O→COMMA is a false positive and COMMA→O a false negative.
  - So a model cannot score by predicting O everywhere.
- **Macro averaging gives every mark equal weight.**
  - COMMA and PERIOD make up 95.75% of non-O tokens, so a micro average would hardly see QUESTION (4.25%) or the rare mode B classes. Restoring those rare marks is the point of the extended task.
  - Micro P/R/F1 is still reported, because prior punctuation work usually quotes it.
- **Mode C keeps all 12 joint classes.** `O|FIRST_UPPER` and `O|ALL_UPPER` carry the casing of unpunctuated words, so dropping the O-punctuation classes would drop most casing decisions.
- Report an ordered confusion matrix including O and both gold/prediction class
  distributions; describe normalization if normalized figures are added.

Zero-division policy: any undefined precision, recall or F1 is `0.0`. Include all
specified classes in a macro average, including zero-support classes, and report
their support explicitly. If a whole scored subset has no gold/predicted
instances, retain this convention and flag the empty support so it is not
misread as observed poor performance. Any alternate metric must name its
denominator and averaging policy.

## Mode C (12 joint classes)

Evaluate each token's `PUNCT|CASE` label and report joint accuracy and macro F1
over all 12 classes, with class-wise support. Any joint micro F1 is secondary
and labeled as such. `O|LOWER`, `O|FIRST_UPPER` and `O|ALL_UPPER` are distinct
joint labels; do not treat the joint vocabulary as if it had one bare O label.

Project both gold and predicted joint labels to the first component for the
standard four-class punctuation metrics above (excluding O for the primary).
Project to the second component for casing accuracy, three-class macro F1,
per-case precision/recall/F1 and a casing confusion matrix. Apply the same
zero-division convention. Primary casing and joint metrics exclude the token
positions listed in the release's casing masks (see below); punctuation at those
positions is still scored. All-token casing is reported as the secondary
`casing/all` block. Any other subset analysis must define its support.
Case-only or separate-model experiments must declare how punctuation and casing
outputs are combined.

## Scored sets, masks, views and the shortcut baseline

**Scored set.**
- The `.ids` list defines which records are scored: `primary_eval/<split>_mode_<M>.ids` for primary results, the full split or `leakage_controlled/*.ids` for supplementary ones.
- Predictions must cover every scored ID.
- A prediction ID that is not in the gold file is an error.
- Prediction IDs that are in the gold file but outside the list are ignored and counted (`ignored_prediction_ids`), so one prediction file can be scored on several lists.
- The mode C lists are subsets of the mode A lists, so A-versus-C punctuation comparisons are scored on the mode C IDs.

**Casing masks.** Each line of `masks/case_joint/<split>.jsonl` is
`{"sample_id", "excluded_positions", "reason"}`, one line per released record.
Masks are required when scoring mode C and rejected for modes A and B.

**Views.** These show how much of a score comes from dataset shortcuts.
- Every mode A and C sequence ends in PERIOD or QUESTION. So does every mode B sequence in the default and primary lists; 10 full-view mode B records end in EXCLAMATION. 45.8–56.0% of PERIOD labels are utterance-final.
- `punctuation/non_final` repeats the punctuation metrics without each sequence's last position.
- `casing/unmasked_non_initial` repeats the casing metrics without position 0.

**Report blocks (the first is primary):**

| Mode | Blocks |
| --- | --- |
| A, B | `punctuation/all`, `punctuation/non_final` |
| C | `joint/unmasked`, `punctuation/all`, `punctuation/non_final`, `casing/unmasked`, `casing/unmasked_non_initial`, `casing/all` |

**Shortcut baseline (`shortcut_baseline`).** This is a reference row, not a model. It reads tokens only.
- Every position gets O, except the last.
- The last position gets QUESTION if the first token is a wh-word or an auxiliary (lists in `evaluation/baseline.py`), otherwise PERIOD.
- In mode C, casing is FIRST_UPPER at position 0 and LOWER elsewhere.

**Running.** All commands are run from the repository root.

```text
mkdir -p results/predictions results/tables
recap evaluate-models baseline --gold <processed>/mode_A/validation.jsonl \
    --ids <splits>/primary_eval/validation_mode_A.ids --out results/predictions/shortcut_A_validation.jsonl
recap evaluate-models score --gold <processed>/mode_C/validation.jsonl \
    --ids <splits>/primary_eval/validation_mode_C.ids --predictions <model predictions>.jsonl \
    --masks <processed>/masks/case_joint/validation.jsonl --out results/tables/<run_id>.json
```

- `score` writes `<run_id>.json`, with schema `recap-eval-v1`, the SHA-256 of every input and the gold file's data policy version, plus a `<run_id>.md` summary.
- Existing outputs are never overwritten, and folders are never created.
- Invalid input exits with code 1 and a one-line message naming the file and the line or `sample_id`.

## Comparability across modes and conditions

B has eight punctuation classes; C has twelve joint classes built from **four**
punctuation classes. Their native macro F1 values represent different tasks.
Do not rank them as equivalent measurements. Compare HMM/basic CRF/improved CRF
within the same mode first.

A compatible cross-mode punctuation comparison requires identical examples and
tokens, verified source gold, the same approved B→A projection applied to both
gold and predictions, and reporting on the shared four-class target. C projects
to its punctuation component. Record the projection policy version. The D5
projection is approved (policy `m1-v2`, `configs/label_sets.yaml`). It cannot
measure retained extended distinctions. Compare casing only when both systems
predict casing.

SponSpeech is the only source. Report `test` and `test_amb` separately and never
pool them, because they share 25 videos. `benchmark_native/mode_A` keeps the
original four-class gold (D5). Report it in its own table and never compare it
with canonical mode A scores. Keep clean, real-ASR and synthetic-noise results
separate. ASR insertions and deletions need a declared alignment policy.

**Known gold issues in the v2 primary lists** (from Member 5's data check; nothing is relabelled):
- **Leftover speaker labels.** The speaker-label rule of `recap find-gold-noise` flags 134 validation, 22 test and 63 test_amb primary mode B records (one example is `SPEAKER 1:`). Speaker-label COLONs are 144 of 425 gold COLONs on validation (33.88%), 23 of 166 on test (13.86%) and 76 of 273 on test_amb (27.84%), and 319 of 3,385 (9.42%) in `default_train`. Read COLON scores as an upper bound ([EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §12).
- **Subtitle cue numbers** (for example `sponspeech:test:nSVl92jJbwk-116`) appear as tokens with gold DASH, and some symbols are dropped from the model input without a flag.
- **Two auto-caption-style videos** miss commas and sentence ends in their gold: `732fql_jhJQ` (135 validation records) and `JNxcznsrRb8` (46 test records).
- **Source transcripts:** an informal, unmeasured estimate puts errors in about 1% of records; no method for it is committed, and the label audit checks our labels against the upstream text, not the upstream text itself. Treat a capital after O as possible gold noise when reading confusion matrices.
- **`QUESTION|ALL_UPPER`** has only 18, 9 and 15 primary tokens (validation, test, test_amb). Always give its support and the bootstrap interval, and never rank mode C models on a joint difference smaller than that interval.

## Efficiency, uncertainty and error analysis

Record fit duration, inference wall time, tokens/second and per-sequence latency
with hardware, threads, dependency versions, batch size, input lengths, warm-up
and repetitions. Distinguish loading, preprocessing, decoding and end-to-end
time. Use identical measurement conditions for runtime comparisons; do not
compare only one model's GPU run to another's CPU run without explanation.

Review representative errors and rare labels, confusion pairs, class imbalance,
sequence length, questions, abbreviations and capitalization cases. Preserve
sample IDs and provenance. Do not select only favorable examples. Optional
confidence intervals should use a documented seed and paired resampling at the
talk/document group level where practical; describe dependence and uncertainty.

**Implemented.**
- **Confidence intervals:** `score --bootstrap N [--seed S]` (N at least 100; the seed defaults to 2026).
  - Whole videos (`metadata.group_id`) are resampled with replacement.
  - Each block's macro F1 gets a 95% percentile interval, `macro_f1_ci`.
  - All blocks share the same draws.
- **Paired comparison:** `compare --predictions-a A --predictions-b B` scores two models on the same gold file, ID list and masks, giving each block's macro F1 difference B − A. With `--bootstrap`, both models use the same resampled videos, which adds the paired 95% interval of the difference and the share of resamples in which B beats A.
- **Error examples:** `errors [--block NAME] [--per-pair K] [--seed S]` counts every wrong position per (gold, predicted) pair. It then draws up to K examples per pair at random with the seed, and writes them as text rebuilt with the gold and with the predicted labels.
- **Prediction manifest:** `score --manifest predictions.manifest.json` checks that the manifest's `model_name`, `label_mode`, `ordered_labels`, `predictions_sha256` and `input.sha256` match the scored predictions, the vocabulary and the gold file. These are the HMM's field names, which every model should mirror. The manifest's identity is then recorded in the report.

**Runtime table: measured** for the HMM ([EVALUATION_RESULTS.md](EVALUATION_RESULTS.md), section 7) and the CRF ([EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md), section 10).
- **Method:** the HMM was re-timed on one machine, single process, with no other jobs running: the median of 3 prediction passes over the primary validation set after a warm-up, plus one in-process fit on `default_train`.
- **The CRF prediction speed was timed the same way** with `recap time-models`.
- **Fit times come from different machines.** The CRF fit times come from Member 3's training manifests (his machine), and the HMM bundle's own `runtime.json` comes from another machine. Neither is compared directly with the re-timed rows.

## Output bundle

Each run produces these outputs:
- a validated prediction JSONL with its manifest;
- metric settings and complete class support;
- tables, confusion matrices and error examples;
- runtime metadata;
- code, environment, configuration and data hashes;
- the selection record and limitations.

Where each output is kept:
- The evaluator JSON outputs are committed in [`results/scores/`](../results/scores/README.md).
- Predictions and trained models are shared through the team Drive with checksums; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
- A score enters a document only after an actual run produced it.
