# Shared model and feature contracts — proposed v1

Owners: Members 2 and 3; reviewers: Members 4–6. These are conceptual interfaces,
not implemented methods or executable examples.

## Common future operations

| Operation | Proposed behavior |
| --- | --- |
| `fit(training_records, *, label_mode, config, feature_extractor=None)` | Validate training records; derive mode-specific targets; fit using training data only; return fitted model |
| `predict(records)` | Accept gold-free input records; return ordered prediction records, one per sequence |
| `save_model(path)` | Save learned state and complete, versioned compatibility/provenance metadata |
| `load_model(path)` | Class-level load returning an equivalent fitted instance; reject incompatible metadata |

Training records follow [DATA_CONTRACT.md](DATA_CONTRACT.md). Prediction inputs
contain `schema_version`, `sample_id`, and `tokens` (plus declared source/split
identity if required). An adapter strips gold labels, original text, and
gold-derived metadata before prediction. Predictions must depend on observable
input and learned state, never on an input record's gold labels.

HMM and CRF document support for the selected A/B/C configuration, expose one
target label per token, preserve IDs and sequence ordering, and fail clearly on
unsupported modes, unfitted state, invalid input or alignment mismatches. Do
not silently fall back to another mode. A mode-specific trained artifact has a
fixed ordered label vocabulary. Optional scores must identify that same order.

## Prediction JSONL

Required fields: `sample_id`, `model_name`, `label_mode`, `predicted_labels`.
An illustrative shape (not a computed prediction) is:

```json
{"sample_id":"sample_0001","model_name":"hmm","label_mode":"A","predicted_labels":["O","PERIOD"]}
```

Use `hmm`, `crf_basic`, or `crf_improved` as initial model names; include `run_id`
to distinguish trained settings. `predicted_labels` is a string array with the
same length as input tokens: A/B contain punctuation labels, C contains joint
`PUNCT|CASE` labels. No missing tokens, duplicate IDs, gold substitution, or
reordered records. Confidence and timing are optional and absent if unsupported;
do not invent calibrated probabilities for arbitrary model scores.

Each prediction file needs a sidecar manifest with input release/split hash,
schema and preprocessing versions, label configuration, run ID, artifact hash,
resolved configuration, code commit, environment and command. Optional runtime
metadata distinguishes end-to-end from decode-only measurements.

## CRF feature injection — Members 3 and 4

Proposed signature:

```text
extract_sequence_features(tokens, *, config) -> list[dict[str, str | int | float | bool]]
```

One dictionary per token; stable feature names and boundary handling; no mutation
of token sequences. The CRF accepts an injected extractor, so basic and improved
presets use the same fit/predict/save implementation. Members 3 and 4 agree this
interface in their first PR before adding feature logic. Member 4 owns the
feature module; Member 3 supplies the agreed basic preset with Member 4 review.

Extractors may use observable tokens and agreed context windows, never
`punct_labels`, `case_labels`, original punctuated text, test statistics, or future
gold boundaries. Gold casing is unavailable when input is lowercased. Any study
using observed input case is a separately described input condition, not hidden
access to original gold. Fit vocabularies/statistics on training only.

## Persistence and reproducibility

Save model family/version, ordered labels and mode, learned parameters,
hyperparameters, feature preset/version, random seeds, tokenizer/normalization
and annotation policy versions, data release and training/validation split
hashes, producing code commit, dependency versions and artifact checksums.
Never package credentials or absolute personal-machine paths. Loading must
check compatibility rather than quietly using current defaults.

Future contract tests verify one output per token, invariant IDs/order, valid
mode labels, no gold leakage, and equivalent predictions after save/load. They
must use tiny fixtures and need no downloaded dataset or pretrained model.

## Future decoding into readable text

For C, split each joint label into punctuation and casing. Restore each word's
case from the approved casing rule, append the approved punctuation symbol after
that word, then apply the versioned spacing policy. A/B alone restore punctuation
and leave lowercased input unless an explicitly configured separate casing model
is present. Sentence-initial capitalization is not an implicit model prediction.

Quotes, contractions, dash spacing and unsupported casing patterns require
approved policies. The compact labels may not reconstruct every original glyph.
No text decoder or model implementation exists in this scaffold.
