# Model contract (implemented)

This is the contract the code implements: the two model classes, the prediction files, the
prediction manifest the evaluator checks, and the decoder that turns labels back into text. Records
follow [DATA_CONTRACT.md](DATA_CONTRACT.md); label vocabularies come from
[configs/label_sets.yaml](../configs/label_sets.yaml).

## Common rules

- **Gold-free prediction.** Both models read only `sample_id` and `tokens` (the HMM also requires
  `schema_version` 1). Gold fields in an input record are ignored, never copied to the output. The
  CLIs strip them before prediction (`src/punct_restore/prediction_io.py`,
  `src/punct_restore/models/hmm_workflow.py`); `tests/test_hmm.py` checks that injected gold labels leave
  HMM predictions unchanged.
- **One label per token, in input order.** Every prediction record has exactly as many labels as
  tokens, and every label belongs to the model's mode vocabulary (A: 4 punctuation labels, B: 8, C: 12
  `PUNCT|CASE` labels).
- **Unfitted models fail.** `predict`, `predict_marginals` and `save_model` on an unfitted model raise
  `RuntimeError`. Invalid input raises `ValueError` (or `TypeError` for a non-mapping record in the CRF).

## `HMMModel` (`src/punct_restore/models/hmm.py`)

| Operation | Behaviour |
| --- | --- |
| `HMMModel(*, label_mode, labels, smoothing=0.1, min_token_count=2)` | `labels` must be exactly the mode's vocabulary; bad smoothing or cutoff raises `ValueError`. |
| `fit(training_records, *, metadata=None)` | Each record needs `schema_version` 1, a unique `sample_id`, `tokens`, `split == "train"`, the model's `label_mode`, `source`, and `punct_labels` / `case_labels` of token length. Mode C targets are `PUNCT|CASE`. A failed refit keeps the old state. |
| `predict(records)` | Returns `{"sample_id", "model_name": "hmm", "label_mode", "predicted_labels"}` per record, in input order; a duplicate `sample_id` raises `ValueError`. |
| `save_model(path)` / `HMMModel.load_model(path)` | One UTF-8 JSON file, no pickle. Top-level keys: `format_version` (1), `model_name` ("hmm"), `config` {`label_mode`, `labels`, `smoothing`, `min_token_count`}, `counts`, `metadata`. Saving refuses an existing file; loading rejects another format version or model name. |

## `CRFModel` (`src/punct_restore/models/crf.py`)

| Operation | Behaviour |
| --- | --- |
| `CRFModel(*, label_mode="A", algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True, feature_config=None, feature_extractor=None, model_name=None, config_path=None)` | `model_name` defaults to `crf_<preset>` (`crf_basic`, `crf_improved`). |
| `fit(training_records, *, label_mode=None, config=None, feature_extractor=None)` | Unique non-empty `sample_id`, non-empty `tokens`, `split` `train` (or absent), labels of token length in the mode vocabulary; mode C also checks `case_labels`. `scripts/train_crf.py` additionally enforces `split` and `label_mode` per record. |
| `predict(records)` | Same four output fields as the HMM, in input order. |
| `predict_tokens(tokens, *, sample_id="seq_0")` | Labels for one token list (used by the demo). |
| `predict_marginals(records)` | One list per record, one dict per token mapping every vocabulary label to its marginal probability (labels unseen in training get 0.0). |
| `save_model(path)` / `CRFModel.load_model(path)` | A `joblib` file holding one dict: `schema_version` (1), `model_family` ("crf"), `model_name`, `label_mode`, `vocabulary`, `hyperparameters`, `feature_config`, `training_metadata`, `environment`, `backend_state` (the fitted `sklearn_crfsuite.CRF`). Loading checks the schema version, family, mode and vocabulary; a missing file raises `FileNotFoundError`. Load only trusted `.joblib` files: they are pickles. |

**Features.** `extract_sequence_features(tokens, *, config=None)` in
`src/punct_restore/features/crf_features.py` returns one feature dict per token from the tokens
alone. A `FeatureConfig` selects the `basic` or `improved` preset, the ablation switches and the
filler cue; the same `CRFModel` trains every preset.

**CASE helper mode.** `CRFModel(label_mode="CASE")` trains a casing-only CRF whose target is
`case_labels` (LOWER, FIRST_UPPER, ALL_UPPER); records still need `punct_labels` of token length. It is
the casing half of the factored pipeline below. It is not a scoring mode: `train_crf.py --mode` and the
evaluator accept only A, B and C, and `train_crf.py predict` refuses CASE artifacts. The demo loads
a CASE artifact but `ModelAdapter` refuses it: `unsupported label mode 'CASE' in model`.

**Threshold decoding.** `decode_with_threshold(marginals, tau, *, priors=None, gamma=0.0)` in
`src/punct_restore/models/threshold.py` labels a token non-O when 1 − P(O) > τ, where P(O) sums every
O label (`O` in modes A/B, every `O|*` in mode C); otherwise it takes the most probable O label. With
γ = 0 the non-O label is the most probable one. With γ > 0 (prior-adjusted decoding) it is the non-O
label c that maximises P(c) / prior(c)^γ, with add-one training priors from `label_priors`; the priors
never move the gate. `check_tau` accepts a finite τ in [0, 1] and `check_gamma` a finite γ ≥ 0.

## Comparison models

Both read only `sample_id` and `tokens` at prediction time and return the same four output fields as
the HMM and the CRF. Unfitted models raise `RuntimeError`.

| Model | Operations | Artifact |
| --- | --- | --- |
| `GapClassifier(label_mode, *, feature_config=FeatureConfig.improved(), C=1.0, max_iter=200, model_name="gap_improved")` (`src/punct_restore/models/gap_classifier.py`) | `fit(records)` with the CRF's record rules (at least two distinct labels); `predict(records)`; `save_model(path)` / `GapClassifier.load_model(path)`. Per-token multinomial logistic regression (scikit-learn, lbfgs) on the CRF feature dicts, with no label transitions; modes A, B and C only; no marginals. | A `joblib` file (a pickle: load trusted files only) holding `schema_version` (1), `model_family` ("gap_classifier"), `model_name`, `label_mode`, `vocabulary`, `feature_config`, `hyperparameters`, `training_summary`, `vectorizer`, `classifier`. |
| `HiddenEventLM(label_mode, *, order=3, discount=0.75, min_count=1, model_name="hidden_event_lm")` (`src/punct_restore/models/hidden_event_lm.py`) | `fit(records)` (`split` train, matching `label_mode`; mode C also needs `case_labels`); `predict(records)`; `save_model(path)` (never overwrites) / `HiddenEventLM.load_model(path)`. An n-gram LM over words and punctuation-event tokens with interpolated absolute discounting; exact Viterbi picks no event or one event after each word. Mode C adds a `UnigramTruecaser` (most frequent casing per word). No marginals. | One JSON file: `format_version` (1), `model_family` ("hidden_event_lm"), `config` {`label_mode`, `order`, `discount`, `min_count`, `model_name`}, `training`, `words`, `ngrams`, `truecaser` (mode C only, else null). |

**Factored casing pipeline.** `combine(punct_predictions, case_predictions, *, model_name)` in
`src/punct_restore/models/factored.py` joins mode A predictions with CASE predictions into mode C labels
(`PUNCT|CASE`), in the order of the punctuation predictions. Missing or extra IDs, length mismatches,
wrong modes or unknown labels raise `ValueError`. `scripts/factor_predictions.py` wraps it.

## Prediction JSONL

One JSON object per line, one per sequence:

```json
{"sample_id": "sample_0001", "model_name": "hmm", "label_mode": "A", "predicted_labels": ["O", "PERIOD"]}
```

(Illustrative shape, not a computed prediction.) `model_name` is `hmm`, `crf_basic`, `crf_improved`,
an ablation name `crf_<variant>`, a threshold-decoded name `<model_name>_tau<τ>` (for example
`crf_improved_tau0.35`; prior-adjusted runs add `_gamma<γ>`), a mode C projection `<name>_joint_C`, a
mode B projection `<name>_from_B`, `gap_improved`, `hidden_event_lm` or `factored`. Records follow the input file's
order; the evaluator matches them by `sample_id`, so order does not affect scores.

## Prediction manifest

`scripts/train_crf.py predict --manifest-out` and `scripts/tune_threshold.py predict` write
(`src/punct_restore/prediction_io.py`): `model_name`, `label_mode`, `ordered_labels`,
`predictions_sha256` and `input` {`path`, `sha256`}. The HMM CLIs write a longer
`predictions.manifest.json` (run ID, model hash, counts, timing, provenance, command) that carries
the input identity under `inputs.input`.

`scripts/evaluate_models.py score --manifest` checks (`src/punct_restore/evaluation/manifest.py`):
`model_name`, `label_mode` and `ordered_labels` (exact order) equal the scored predictions and the
mode vocabulary, `predictions_sha256` equals the predictions file, and the input `sha256` (top-level
`input` or `inputs.input`; both must agree if both exist) equals the gold file's. CRF training
manifests are separate sidecars, not prediction manifests.

## Decoder: labels back to text (`src/punct_restore/pipeline.py`)

- **Loading:** `load_model(path)` loads four model families. The suffix (any letter case) picks a
  table, then a marker inside the file picks the class, which loads the file again:

  | Suffix | Marker in the file | Class |
  | --- | --- | --- |
  | `.json` | `"model_family": "hmm"`, or no `model_family` and `"model_name": "hmm"` | `HMMModel` |
  | `.json` | `"model_family": "hidden_event_lm"` | `HiddenEventLM` |
  | `.joblib`, `.pkl` | a dict with `"model_family": "crf"` | `CRFModel` |
  | `.joblib`, `.pkl` | a dict with `"model_family": "gap_classifier"` | `GapClassifier` |

  A `model_family` field wins over `model_name`. A family from the other suffix's table (for example
  `"crf"` in a `.json` file) is refused. `.joblib` and `.pkl` files are pickles: load trusted files only.
- **Order of the checks:** suffix, then folder, then missing file, then reading the file, then the
  family marker. So a folder without a model suffix gets the suffix error
  (`unsupported model file: <name> (expected .json, .joblib or .pkl)`), not the folder error.
- **Loading errors** (one line, file name only):
  - another suffix: `ValueError` `unsupported model file: <name> (expected .json, .joblib or .pkl)`;
  - a folder: `IsADirectoryError` `model path is a folder, not a file: <name>`;
  - a missing file: `FileNotFoundError` `model file not found: <name>`;
  - a file that cannot be read: `OSError` `cannot read model file <name>: <reason>`;
  - invalid JSON or a file that does not unpickle: `ValueError` `unsupported model file: <name> (unreadable)`;
  - no dict, or a missing or unknown family: `ValueError` `unsupported model file: <name> (unknown model family)`.

  After dispatch, the class loader's own errors pass through unchanged.
- **Label mode:** `ModelAdapter` accepts models in modes A, B and C only. Any other mode, such as a
  CASE-mode CRF, raises `ValueError` `unsupported label mode '<mode>' in model`.
- **Input cleaning:** `RestorationPipeline(adapter).process_text(text)` splits on whitespace,
  lowercases each token and strips `,.?!:;-"` from its edges. Internal apostrophes and hyphens stay;
  empty input raises `ValueError`.
- **Prediction:** `ModelAdapter(model_path)` builds one gold-free record
  (`schema_version` 1, `sample_id`, `tokens`) and checks that the model returns one label per token from
  the mode's vocabulary.
- **Reconstruction** (`reconstruct`): a label splits on its first `|` into punctuation and casing (no
  `|` means LOWER). FIRST_UPPER capitalises the first character and ALL_UPPER the whole word. After the
  word, COMMA adds `,`, PERIOD `.`, QUESTION `?`, EXCLAMATION `!`, COLON `:`, SEMICOLON `;` and DASH
  ` -`; O adds nothing. Words are joined by single spaces.
- **Modes A and B** predict no casing, so their restored text stays lowercase; there is no
  sentence-initial capitalisation rule. `scripts/run_demo.py` wraps this pipeline.
