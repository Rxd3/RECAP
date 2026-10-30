# CRF Engineering & Technical Notes — Member 3 Workstream

**Modules:** `src/punct_restore/models/crf.py`, `scripts/train_crf.py`, `src/punct_restore/features/crf_features.py`
**CRF Tests:** `tests/test_crf_features.py`, `tests/test_crf_model.py`, `tests/test_train_crf_cli.py`
**Status:** Six full-data CRF models (modes A, B, C × basic, improved) were trained on the official `sponspeech-m1-v2` `default_train` lists; see [CRF_EXPERIMENTS.md](../docs/CRF_EXPERIMENTS.md) §7. Member 5 reproduced all six validation prediction files byte for byte from the shared models and scored each frozen model once on test and test_amb. Validation chose the HMM smoothing and the threshold τ; the CRF settings (c1 = c2 = 0.1, 100 iterations) were fixed and not tuned; no test result influenced any choice. The final results are in [EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md). The test counts below describe the suite at commit `aa232be`.

---

## 1. Backend & Runtime Environment

- **CRF Backend Engine:** `python-crfsuite` (v0.9.12), wrapped via `sklearn-crfsuite` (v0.5.0).
- **Core C++ Library:** CRFsuite by Naoaki Okazaki (fast C++ implementation of first-order linear-chain CRFs).
- **Optimizer:** L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno).
- **Runtime Environment:** Windows x64, Python 3.15.2 (`.venv`), compiled natively via MSVC.
- **Serialization:** `joblib` (v1.6.0).

---

## 2. Model Architecture & API Signatures

The public class `CRFModel` in `src/punct_restore/models/crf.py` implements the shared model contract ([docs/MODEL_CONTRACT.md](../docs/MODEL_CONTRACT.md)):

```python
class CRFModel:
    def __init__(
        self,
        *,
        label_mode: str = "A",
        algorithm: str = "lbfgs",
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 100,
        all_possible_transitions: bool = True,
        feature_config: FeatureConfig | Mapping[str, Any] | None = None,
        feature_extractor: Callable[..., list[dict[str, Any]]] | None = None,
        model_name: str | None = None,
        config_path: str | Path | None = None,
    ) -> None: ...

    def fit(
        self,
        training_records: Sequence[Mapping[str, Any]],
        *,
        label_mode: str | None = None,
        config: Mapping[str, Any] | None = None,
        feature_extractor: Callable[..., list[dict[str, Any]]] | None = None,
    ) -> CRFModel: ...

    def predict(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]: ...

    def predict_tokens(
        self,
        tokens: Sequence[str],
        *,
        sample_id: str = "seq_0",
    ) -> list[str]: ...

    def save_model(self, path: str | Path) -> None: ...

    @classmethod
    def load_model(cls, path: str | Path) -> CRFModel: ...
```

### Integration with the demo (historical note)
An early draft of `pipeline.py` called `predict` with a bare token list. The final
`src/punct_restore/pipeline.py` calls `predict_tokens`, as recommended here; see
[MODEL_CONTRACT.md](../docs/MODEL_CONTRACT.md).

---

## 3. Label Modes & Mode C Casing-Mask Behavior

The model uses a single unified CRF engine across all three modes:

| Mode | Class Count | Target Derivation | Vocabulary |
| :---: | :---: | :--- | :--- |
| **A** | 4 | Direct `punct_labels` | `O`, `COMMA`, `PERIOD`, `QUESTION` |
| **B** | 8 | Direct `punct_labels` | Mode A + `EXCLAMATION`, `COLON`, `SEMICOLON`, `DASH` |
| **C** | 12 | Composed via `join_labels(punct, case, strict=True)` | $4 \times 3 = 12$ classes formatted as `PUNCT\|CASE` |

### Mode C Masking Analysis (Decisions D4 & D9)
- **Training Labels:** In SponSpeech v2, every token position in `case_labels` has a concrete class (`LOWER`, `FIRST_UPPER`, `ALL_UPPER`). Mixed-case tokens (e.g., *iPhone*, *eBay*) receive a deterministic step-4 heuristic label (*iPhone* $\to$ `LOWER`, *McDonald* $\to$ `FIRST_UPPER`). The CRF is trained on **all** positions in the sequence using the full 12-class Cartesian product targets.
- **Evaluation Masks:** Member 1 generates sidecar mask files (`masks/case_joint/<file>.jsonl`) documenting approximated token indices. During evaluation, Member 5's scoring pipeline reads these masks and **excludes** those specific token positions from joint and casing metrics. Punctuation at those positions is still evaluated.
- **Training targets:** linear-chain CRFs need a label at every position, and a masked loss would need custom forward-backward code. The models therefore train on the complete targets, and the masks apply at evaluation only (D4).

---

## 4. Configuration Hardening & Zero-Fallback Rule

- **Authoritative Configuration:** `get_mode_vocabulary()` strictly queries `load_label_sets()` from `src/punct_restore/data/labeling.py` which parses `configs/label_sets.yaml`.
- **No Broad Fallback:** Broad `except Exception:` fallbacks were eliminated. If `label_sets.yaml` is missing, malformed, or structurally inconsistent, an explicit exception (`FileNotFoundError`, `YAMLError`, or `ValueError`) is raised immediately.
- **Split Protection:** In `_validate_training_records()`, any record with `split != "train"` (including `validation` and `test`) raises `ValueError`, strictly preventing data leakage.

---

## 5. Artifact Provenance & Persistence Invariants

The artifact saved via `save_model(path)` bundles:
1. **Required Functional Fields (Executable):**
   - `schema_version`: Integer 1.
   - `model_family`: `"crf"`.
   - `label_mode`: `"A"`, `"B"`, or `"C"`.
   - `vocabulary`: Complete ordered list of valid target labels.
   - `hyperparameters`: Algorithm, L1/L2 penalties (`c1`, `c2`), `max_iterations`, and transition flags.
   - `feature_config`: Serialized feature configuration dictionary.
   - `backend_state`: Trained `sklearn_crfsuite.CRF` estimator.
2. **Provenance & Audit Fields (Informational):**
   - `model_name`: Descriptive identifier (e.g. `"crf_basic"`, `"crf_improved"`).
   - `training_metadata`: Number of training sequences, token counts, observed classes, and optional split/release hashes.
   - `environment`: Python version, platform, `sklearn_crfsuite`, and `joblib` versions.

---

## 6. Synthetic Test Evidence (`tests/test_crf_model.py`)

The tests verify:
- **Category A (Init):** Parameter boundary checks, invalid mode rejection, non-finite regularization rejection.
- **Category B (Training):** Target derivation, empty dataset rejection, duplicate ID rejection, split protection (`test` and `validation`), input immutability.
- **Category C (Prediction):** Record-oriented output schema matching `MODEL_CONTRACT.md`, length preservation, unseen word robustness, predict-before-fit guard.
- **Category D (Feature Integration):** Basic vs improved feature presets with identical model class.
- **Category E (Persistence):** Save/load round-trip equivalence (exact prediction match before and after serialization), corrupt artifact handling.
- **Category F (Modes A/B/C):** 4-, 8-, and 12-class target spaces and joint string format validation.
- **Category G (Configuration Hardening):** Missing config rejection, malformed YAML rejection, structural inconsistency rejection, and provenance tracking.

---

## 7. Scientific Limitations & Boundaries (Phase 4 vs Phase 5)

Passing these synthetic unit tests proves the **software correctness** and **contract compliance** of the model implementation. It does **not** prove:
- High non-O macro F1 scores on conversational speech.
- Generalization to out-of-domain conversations.
- Optimal hyperparameter selection ($c_1, c_2$).
- Superiority over the HMM baseline.

Real-data performance is measured by Member 5, not by these unit tests: validation results for the six full-data models are in [EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md).

---

## 8. Phase 5: Reproducible CRF Training Pipeline (`scripts/train_crf.py`)

**CLI Script:** `scripts/train_crf.py`
**CLI Tests:** `tests/test_train_crf_cli.py`

### 8.1 CLI Architecture & Supported Operations

`scripts/train_crf.py` provides a thin, deterministic command-line interface delegating directly to `CRFModel` and `extract_sequence_features`. It implements three primary subcommands:

1. **`train` — Train a linear-chain CRF model:**
   ```bash
   python scripts/train_crf.py train \
       --mode {A,B,C} \
       --input <path/to/data.jsonl> \
       --ids <path/to/mode_<MODE>.ids> \
       --output <path/to/model.joblib> \
       [--preset {basic,improved}] \
       [--algorithm lbfgs] \
       [--c1 0.1] \
       [--c2 0.1] \
       [--max-iterations 100] \
       [--window-size 2] \
       [--no-utterance-boundaries] \
       [--run-id <run_name>] \
       [--dataset-release sponspeech-m1-v2] \
       [--policy-version m1-v2] \
       [--manifest-out <path/to/custom_manifest.json>] \
       [--overwrite]
   ```

2. **`predict` — Generate gold-free contract-compliant predictions:**
   ```bash
   python scripts/train_crf.py predict \
       --model <path/to/model.joblib> \
       --input <path/to/eval_input.jsonl> \
       --output <path/to/predictions.jsonl> \
       [--ids <path/to/eval.ids>] \
       [--overwrite]
   ```

3. **`inspect` — Inspect metadata and parameters of a saved artifact:**
   ```bash
   python scripts/train_crf.py inspect \
       --model <path/to/model.joblib>
   ```

### 8.2 Frozen Training-ID Enforcement Invariants

In accordance with Step 3 and Member 1's dataset contracts, training data filtering enforces 10 strict invariants:
1. **Membership Verification:** Only records whose `sample_id` appears in the designated `.ids` manifest are admitted into training.
2. **Cardinality Match:** The number of loaded training records must match the ID-list length exactly (`len(selected_records) == len(frozen_ids)`).
3. **Completeness Guard:** Every single ID declared in the `.ids` manifest must be found in the input JSONL. If any ID is missing, training aborts with an actionable error.
4. **ID Uniqueness:** Duplicate IDs in the frozen `.ids` manifest or in the input JSONL trigger immediate failure.
5. **Split Integrity:** Every selected record must possess `split == "train"`. Records marked with `validation`, `test`, or any non-train split raise `ValueError`.
6. **Mode Integrity:** Selected records must strictly match the expected `--mode` (e.g. Mode A rejects Mode B records).
7. **Sequence Alignment:** Token array lengths must equal `punct_labels` length (and `case_labels` length in Mode C).
8. **Exclusion of Unselected Data:** Records in the input JSONL not listed in the frozen `.ids` manifest are completely excluded from memory and model fitting.
9. **Source Order Preservation:** Filtered training sequences maintain the exact relative ordering of the source JSONL.
10. **Zero Modification Policy:** The dataset is treated as immutable read-only input. No splitting or regeneration is performed.

### 8.3 Sidecar Provenance Manifest Schema

Every training run automatically generates a companion provenance manifest (`<model>.manifest.json`):
- `schema_version`: Integer 1.
- `run_id`: Unique experiment identifier.
- `label_mode`: `"A"`, `"B"`, or `"C"`.
- `model_name`: Artifact name.
- `vocabulary`: Authoritative ordered label vocabulary.
- `feature_preset`: `"basic"` or `"improved"`.
- `feature_preset_version`: Semantic version string.
- `feature_config`: Full serialized feature extractor configuration dictionary.
- `hyperparameters`: Optimization algorithm, L1 penalty (`c1`), L2 penalty (`c2`), `max_iterations`, and transition matrix flag.
- `provenance`:
  - `dataset_release`: Dataset release tag (e.g. `sponspeech-m1-v2`).
  - `policy_version`: Policy version (e.g. `m1-v2`).
  - `input_file`: POSIX-normalized path to input dataset JSONL.
  - `input_sha256`: SHA-256 hash of input dataset file.
  - `ids_file`: POSIX-normalized path to frozen ID manifest.
  - `ids_sha256`: SHA-256 hash of frozen ID manifest file.
  - `selected_sequences`: Integer count of selected sequences.
  - `selected_tokens`: Integer count of selected tokens.
- `training_command`: Command line invocation arguments (`sys.argv`).
- `training_duration_seconds`: High-precision float training duration.
- `artifact`: Path, SHA-256 integrity checksum, and file size in bytes.
- `backend`: Backend library name and version (`sklearn-crfsuite`, `python-crfsuite`).
- `environment`: Python version, operating system, machine architecture, CPU core count, and dependency versions (`joblib`, `numpy`, `scipy`, `recap-nlp`).
- `code_fingerprint`: SHA-256 digest computed across `scripts/train_crf.py`, `configs/label_sets.yaml`, `pyproject.toml`, and all 19 `punct_restore/**/*.py` source files.
- `fingerprinted_files`: Deterministically sorted repository-relative POSIX file list.
- `reproducibility_warning`: Explicit note stating local uncommitted modifications are present on `feat/m3-crf-model`.
- `created_at`: ISO-8601 UTC timestamp.

### 8.4 Prediction Export & Gold-Free Inference

In `cmd_predict`:
- Strictly observable input: Extracts only `sample_id` and `tokens`. Gold labels (`punct_labels`, `case_labels`) or raw text strings are completely ignored and stripped.
- Format compliance: Output lines conform strictly to `MODEL_CONTRACT.md`:
  ```json
  {"sample_id": "seq_0001", "model_name": "crf_basic", "label_mode": "A", "predicted_labels": ["O", "PERIOD"]}
  ```
- Strict token alignment: Output predictions produce exactly one label per input token (`len(predicted_labels) == len(tokens)`).
- Vocabulary enforcement: Every predicted label is validated against the model's loaded vocabulary.
- Non-overwriting safety: Output files will not be silently replaced unless `--overwrite` is explicitly passed.

### 8.5 Verification & Test Coverage (`tests/test_train_crf_cli.py`)

Automated tests at commit `aa232be` cover:
1. `test_train_mode_a_success`: Mode A training (4 labels), artifact and manifest generation.
2. `test_train_mode_b_success`: Mode B training (8 labels), vocabulary validation.
3. `test_train_mode_c_success`: Mode C training (12 joint labels), vocabulary validation.
4. `test_train_frozen_ids_excludes_unselected_records`: Unselected sequences completely excluded.
5. `test_train_missing_id_list_failure`: Missing `.ids` file exits with code 1.
6. `test_train_missing_selected_id_in_input_failure`: ID in list but missing in JSONL exits with code 1.
7. `test_train_duplicate_id_in_frozen_list_failure`: Duplicate IDs in list exit with code 1.
8. `test_train_empty_frozen_ids_failure`: Empty `.ids` list exits with code 1.
9. `test_train_rejects_non_train_split`: Non-train splits (e.g. `val`) exit with code 1.
10. `test_train_rejects_mode_mismatch`: Mode mismatch (e.g. Mode B record in Mode A run) exits with code 1.
11. `test_train_token_label_length_mismatch`: Token/label sequence length discrepancy exits with code 1.
12. `test_train_corrupt_jsonl_failure`: Corrupt JSON syntax exits with code 1.
13. `test_train_invalid_hyperparameters`: Invalid `c1 < 0` exits with code 1.
14. `test_train_non_overwriting_protection`: Existing artifact prevents overwrite without `--overwrite`.
15. `test_train_paths_with_spaces_and_windows_compatibility`: Paths with spaces on Windows succeed.
16. `test_predict_schema_compliance_and_gold_free`: Gold-free prediction JSONL matches contract.
17. `test_predict_with_frozen_ids_filtering`: Predict restricted to `--ids` subset.
18. `test_predict_missing_id_failure`: Missing ID in prediction input exits with code 1.
19. `test_predict_non_overwriting_protection`: Existing prediction file protected from overwrite.
20. `test_save_load_prediction_equivalence`: Serialized and reloaded models produce bitwise identical predictions.
21. `test_inspect_valid_artifact`: Model artifact inspection output with companion manifest details.
22. `test_inspect_missing_file_failure`: Inspection of missing file exits with code 1.
23. `test_code_fingerprint_deterministic_and_comprehensive`: Verifies deterministic multi-file source fingerprinting.

---

## 9. Scientific Validity & Result Classification (Phase 5.1 Audit)

To maintain university research integrity, we strictly delineate the following five categories:

### 9.1 Implemented Functionality (Code in Repository)
- Modular linear-chain CRF model (`CRFModel` in `src/punct_restore/models/crf.py`).
- Deterministic feature extraction (`FeatureConfig.basic()`, `FeatureConfig.improved()`).
- CLI pipeline (`scripts/train_crf.py`) supporting `train`, `predict`, and `inspect`.
- Frozen training ID filtering with strict cardinality and split protection.
- Comprehensive provenance generation and multi-file source fingerprinting.

### 9.2 Tested Functionality (Verified via Automated Tests)
- Member 3's feature, model and CLI tests (`tests/test_crf_features.py`, `tests/test_crf_model.py`, `tests/test_train_crf_cli.py`) run in the full pytest suite.
- Verified invariants: parameter boundaries, non-overwriting guards, save/load bitwise equivalence, vocabulary containment, gold-free inference, and cross-platform compatibility.

### 9.3 External Teammate-Reported Reference Numbers (Not CRF Results)
A teammate reported the following initial benchmark numbers from a local rebuild:
- **Baseline HMM (Untuned, 142,722 training sequences, ~4.14M tokens):**
  - Mode A Validation Macro F1: **31.95%**
  - Mode B Validation Macro F1: **15.25%**
  - Mode C Validation Macro F1: **39.48%**
- **Shortcut Baseline:**
  - Mode A: **28.21%**
  - Mode B: **12.14%**
  - Mode C: **18.53%**
- **Confidence Intervals:** Reported as video-grouped 95% CIs, but actual numeric interval bounds were not supplied.
- *Status:* External reference figures only. Not verified by Member 3. The untuned HMM figures are superseded by the tuned HMM (validation A 36.54, B 18.47, C 40.89) in [EVALUATION_RESULTS.md](../docs/EVALUATION_RESULTS.md).

### 9.4 Result Claims & Prohibitions
- **CRF Benchmark Claims:** The six CRF models were trained on the full official training lists ([CRF_EXPERIMENTS.md](../docs/CRF_EXPERIMENTS.md) §7). Benchmark numbers come only from Member 5's evaluator in [EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md): final test results in §1–4 and test_amb in §9, with validation (model selection) in §5.
- **Training data identity:** the six training manifests record `input_sha256` and `ids_sha256`, which match the release `SHA256SUMS` (checked 2026-10-30).
- **Configurable vs Tuned Hyperparameters:** CLI regularization options (`--c1`, `--c2`) are user-configurable runtime arguments. No validation-based hyperparameter tuning has been performed; all six models use c1 = c2 = 0.1 and 100 iterations.

### 9.5 Follow-up Work (status)
- **Not done:** a validation grid search over $c_1, c_2$. The CRF settings stayed fixed. A validation sensitivity check (six settings, 50k sample, not used for selection) is in [EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md) §5.
- **Done:** feature ablations on **validation only**, never on test: one feature group is removed at a time from the improved preset and trained on a 50k-sequence sample. Results are in [EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md) §6; `scripts/run_ablations.py --mode {A,B,C} --out DIR` reruns them. The utterance-final shortcut is measured with Member 5's `non_final` view on validation and test.

---

## 10. Team Integration Interface Specification

### 10.1 Target Consumers
- **Member 6's demo path:** `src/punct_restore/pipeline.py` loads a saved CRF (`load_model` dispatches `.joblib`/`.pkl` files with `"model_family": "crf"` to `CRFModel.load_model`) and calls `predict`; `scripts/run_demo.py` wraps it. No web application was built.

### 10.2 Required Input Representation
Input tokens must be:
- A sequence of **lowercased**, whitespace-tokenized string words.
- Strictly gold-free (no ground-truth punctuation or casing hints).
- Free of empty strings (`""`) or non-string objects.
- *Critical Caveat:* Input text must be explicitly lowercased prior to inference. Uppercase characters in input tokens will bypass casing restoration in Mode C and corrupt token shape features.

### 10.3 Interface Signatures

`CRFModel` exposes two complementary inference methods:

1. **`predict(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]` (Batch & Contract Evaluation):**
   - **Input:** Sequence of observation records:
     ```python
     records = [
         {"sample_id": "utt_001", "tokens": ["how", "are", "you"]},
         {"sample_id": "utt_002", "tokens": ["i", "am", "fine"]},
     ]
     ```
   - **Output:** List of dictionaries matching `docs/MODEL_CONTRACT.md`:
     ```python
     [
         {
             "sample_id": "utt_001",
             "model_name": "crf_basic",
             "label_mode": "A",
             "predicted_labels": ["O", "O", "QUESTION"],
         },
         ...
     ]
     ```

2. **`predict_tokens(tokens: Sequence[str], *, sample_id: str = "seq_0") -> list[str]` (Interactive Demo & Pipeline Adapter):**
   - **Input:** Single sequence of lowercased tokens:
     ```python
     tokens = ["hello", "world"]
     ```
   - **Output:** Flat list of predicted labels:
     ```python
     ["O", "PERIOD"]
     ```

### 10.4 Model Loading
```python
from punct_restore.models.crf import CRFModel

# Load serialized bundle (automatically reconstructs feature config & vocabulary)
model = CRFModel.load_model("crf_mode_a_basic_full_01.joblib")
assert model.is_fitted
print("Loaded model mode:", model.label_mode)
print("Vocabulary:", model.vocabulary)
```

### 10.5 Error Handling & Guarantees
- Calling `predict()` on an unfitted model raises `RuntimeError`.
- Empty token sequences or non-string tokens raise `ValueError`.
- Output sequence length is guaranteed to be strictly equal to input token length (`len(predicted_labels) == len(tokens)`).

---

## 11. Smoke-Test Protocol (historical)

A 1,000-sequence smoke-test protocol was written before the first real-data run. It is superseded by the
six full-data runs ([CRF_EXPERIMENTS.md](../docs/CRF_EXPERIMENTS.md) §7) and by Member 5's scoring with
`scripts/evaluate_models.py` ([EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md)).
