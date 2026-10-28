# Shared data contract — proposed schema v1

Owner: Member 1. Reviewers: all members. All future implementations use this
contract; annotation policies marked pending require agreement before release.

## Serialization and fields

UTF-8 JSONL: one JSON object on **one physical line** per nonempty token
sequence. The following illustrative record is a single JSONL line, not data:

```json
{"schema_version":1,"sample_id":"sample_0001","source":"illustrative","split":"train","label_mode":"A","tokens":["hello","everyone"],"punct_labels":["O","PERIOD"],"case_labels":["FIRST_UPPER","LOWER"]}
```

| Field | Requirement |
| --- | --- |
| `schema_version` | Integer `1`; reject unknown versions |
| `sample_id` | Stable, nonempty, unique string within the release; never based on a shuffled row number |
| `source` | Nonempty source identifier resolved by the release manifest |
| `split` | `train`, `validation`, or `test`; must agree with the frozen split map |
| `label_mode` | `A`, `B`, or `C`; fixes the allowed punctuation vocabulary |
| `tokens` | Nonempty list of nonempty strings, with no whitespace-only tokens |
| `punct_labels` | Exactly one punctuation label per token: A/C use standard 4; B uses extended 8 |
| `case_labels` | Exactly one of LOWER, FIRST_UPPER, ALL_UPPER per token in every annotated release |
| `metadata` | Optional original text, upstream ID, talk/document/group ID, offsets, source revision, policy version, parent ID, condition. SponSpeech releases use `upstream_id`, `source_split`, `group_id`, `original_text`, `normalized_text` (the text `offsets` index into), `offsets`, `source_revision`, `policy_version`, `annotation_view` (`canonical` or `native`) and `condition` |

Punctuation belongs to the boundary **after** the corresponding token. Casing
belongs to that token, not to the next word after a PERIOD. Every token/label
array has identical length. Original text and gold metadata are unavailable to
model predictors and feature functions.

IDs should combine an immutable source identity with stable upstream segment
identity, or a documented deterministic digest if none exists. Collision checks
are required. The same underlying sequence keeps its ID, tokens and split across
A/B/C views; the release manifest and `label_mode` distinguish the views. Do not
concatenate views into one file with duplicate IDs. Content duplicates are
audited separately from identifier uniqueness.

## Tokens, normalization and casing

Proposed baseline input policy: derive labels from original text first, retain
word alignment/offsets, remove target boundary punctuation according to the
approved policy, and apply Python `str.lower()` to input word tokens. Do not use
`casefold()` interchangeably. Version this policy before producing a release.
All models receive exactly the same ordered token strings and sequence windows.

Casing follows the approved D4 precedence ([DECISIONS.md](DECISIONS.md)), applied in this order:
(1) no cased letters, including numbers → LOWER; (2) a single uppercase letter such as `I` or `A` → FIRST_UPPER;
(3) several cased letters, all uppercase → ALL_UPPER; (4) otherwise the first cased letter decides (uppercase →
FIRST_UPPER, lowercase → LOWER). Step 4 approximates mixed-case words such as `iPhone`. Those token positions keep
their label and are listed in a per-release mask (`masks/case_joint/<file>.jsonl`: `sample_id`,
`excluded_positions`, `reason`), so primary casing and joint metrics can exclude them without changing token
alignment; punctuation at those positions is still scored. The mask format is a Member 1 proposal that Member 5
must accept for the evaluation contract. No fourth casing class may be added casually.

Define contractions, apostrophes, hyphens, decimals, abbreviations, Unicode dash
variants, quote/bracket attachment, repeated punctuation, leading punctuation,
missing terminal punctuation and sentence/window boundaries before extraction.
Record how loss of some orthographic detail limits exact text reconstruction.
The approved D2/D3 rules (policy `m1-v1`) keep contractions, numbers, URLs, abbreviations, word-internal hyphens
and negative signs inside tokens, map `--` and standalone spaced hyphens to DASH, and turn ellipses into O inside an
utterance and PERIOD at its end; one sequence is one source utterance (no windowing).

## Label modes and projection

The ordered vocabularies are in [label_sets.yaml](../configs/label_sets.yaml):

- A: O, COMMA, PERIOD, QUESTION.
- B: A plus EXCLAMATION, COLON, SEMICOLON, DASH.
- C: the Cartesian product of A and LOWER, FIRST_UPPER, ALL_UPPER.

`O` means no target punctuation after the word. It does not mean missing casing.
For C, derive the target at index `i` as
`punct_labels[i] + "|" + case_labels[i]`. Punctuation is the first component;
casing is the second. There are exactly **4 × 3 = 12** unique classes. Iterate
punctuation first and casing second in configuration order. Do not combine B
with casing to form a 24-class target; do not store an independent, diverging
`joint_labels` gold array.

### B → A projection (approved, D5)

| Extended | Standard |
| --- | --- |
| O | O |
| COMMA | COMMA |
| PERIOD | PERIOD |
| QUESTION | QUESTION |
| EXCLAMATION | PERIOD |
| COLON | COMMA |
| SEMICOLON | PERIOD |
| DASH | COMMA |

The mapping treats colon/dash as internal boundaries and exclamation/semicolon
as stronger boundaries. It is a modeling simplification, not a linguistic
equivalence. The team lead approved it on 2026-10-28 (D5, policy `m1-v1`;
`approved: true` in `label_sets.yaml`). Canonical modes A and C use it. A release
may also carry a separately versioned *native* four-class view of the source
(`benchmark_native/`, `metadata.annotation_view: "native"`) for comparison with
the original benchmark; its scores are never compared with canonical scores as
if the gold labels were the same. Any later change of mapping changes the
release/projection version and comparison target.

A is a subset of B, so A → B preserves those four names but cannot recover lost
extended distinctions. Native four-label benchmark data must not be presented
as eight-label ground truth. Deriving C needs verified casing in addition to A;
missing casing cannot be filled with guessed gold. A B-mode source can yield C
only after the B → A policy has been approved and casing has been validated.

## Validation, splitting and malformed input

Future validators reject invalid JSON, missing/wrongly typed fields, unknown
labels/modes, empty sequences, empty tokens, duplicate IDs, mismatched lengths,
and split-map conflicts. Keep rejection reasons/counts and source identifiers in
an audit log; fail the release validation on unhandled records. Blank lines are
malformed, not silently skipped. Exclusion after review must be documented and
counted. Never truncate labels or drop individual tokens to repair a record.

Fix shared train/validation/test assignments once, grouped by talk/document (or
an approved equivalent). Audit exact and near duplicates before freezing, across
splits, source configurations, and IWSLT years; mirrored translation directions
may repeat the same English text. Keep duplicates/paired variants in the same
group. Respect existing benchmark split semantics where possible; conflicts
require an explicit decision and a new version, not hidden resplitting.
For SponSpeech (D6) the overlap unit is the source video ID. `dev` becomes
`validation`; the second test set test-amb keeps `split: "test"`, records its
origin in `metadata.source_split` and is written to its own file, never merged
with `test`. Leakage-controlled validation/test ID lists, frozen before model
selection, are the primary evaluation sets; the full official splits are
supplementary.

The manifest records source revisions, sample/group assignments, normalization
and tokenizer versions, segmentation policy, seed where relevant, exclusions,
record/token totals and checksums. Split ratios and counts are unassigned today.

## Clean versus noisy records

Only label a condition ASR when its provenance supports that claim. Synthetic
noise must be named synthetic. Identical-token paired variants can share an
alignment manifest; changed-token ASR hypotheses need distinct stable IDs and
`parent_sample_id` plus an explicitly validated word alignment/reference policy.
Do not copy clean label arrays onto insertions/deletions. Cross-condition scores
must describe whether the evaluation target and tokenization are comparable.

All count/statistics tables remain pending until actual preprocessing. See
[DATASET_PLAN.md](DATASET_PLAN.md) for the inspection and release checklist.
