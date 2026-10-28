# Evaluation protocol — planned v1

Owner: Member 5. Reviewers: Members 1–4 and 6. This document specifies future
behavior; no evaluation calculations or results exist yet.

## Data discipline and alignment

Training fits models and train-derived statistics. Validation selects smoothing,
features, hyperparameters, thresholds and any decoding settings. Freeze selected
settings before using the held-out test set for final reporting. Report a test
set change or test-driven iteration honestly and revise the protocol if needed.

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
zero-division convention. Report casing on all validated tokens; any subset
analysis must define its support. Case-only or separate-model experiments must
declare how punctuation and casing outputs are combined.

## Comparability across modes and conditions

B has eight punctuation classes; C has twelve joint classes built from **four**
punctuation classes. Their native macro F1 values represent different tasks.
Do not rank them as equivalent measurements. Compare HMM/basic CRF/improved CRF
within the same mode first.

A compatible cross-mode punctuation comparison requires identical examples and
tokens, verified source gold, the same approved B→A projection applied to both
gold and predictions, and reporting on the shared four-class target. C projects
to its punctuation component. Record the projection policy version. This is
blocked until the proposed mapping is approved, and it cannot measure retained
extended distinctions. Compare casing only when both systems predict casing.

Native IWSLT 2011 and derived IWSLT 2017 tests are separate benchmarks unless a
verified common test release exists. Their source years alone do not prove
compatibility or independence. Complete the cross-source leakage audit first.
Separate clean, real ASR, and synthetic-noise results. ASR insertions/deletions
require a declared alignment/reference policy; do not directly compare token
metrics over different token inventories as if they were paired unchanged input.

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

## Required future output bundle

For each run: validated prediction JSONL + manifest, metric settings and complete
class support, tables, confusion matrices, error examples, runtime metadata,
code/environment/configuration/data hashes, selection record and limitations.
Store under ignored `results/` paths; share a versioned bundle. All numerical
result files are absent initially. Never enter scores before an actual run.
