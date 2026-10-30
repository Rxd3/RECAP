# CRF Engineering & Technical Notes — Member 3 Workstream

**Modules:** `src/punct_restore/models/crf.py`, `scripts/train_crf.py`, `src/punct_restore/features/crf_features.py`
**CRF Test Suite:** 94 tests passed (23 feature, 48 model, 23 CLI) | **Total Project Suite:** 209 tests passed
**Status:** Six full-data CRF models (modes A, B, C × basic, improved) were trained on the official `sponspeech-m1-v2` `default_train` lists; see [CRF_EXPERIMENTS.md](../docs/CRF_EXPERIMENTS.md) §7. Member 5 scored their primary-validation predictions in [EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md); test predictions are pending. The test counts below describe the suite at commit `aa232be`.

---

## 1. Backend & Runtime Environment

- **CRF Backend Engine:** `python-crfsuite` (v0.9.12), wrapped via `sklearn-crfsuite` (v0.5.0).
- **Core C++ Library:** CRFsuite by Naoaki Okazaki (fast C++ implementation of first-order linear-chain CRFs).
- **Optimizer:** L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno).
- **Runtime Environment:** Windows x64, Python 3.15.2 (`.venv`), compiled natively via MSVC.
- **Serialization:** `joblib` (v1.6.0).

---

## 2. Model Architecture & API Signatures

The public class `CRFModel` in `src/punct_restore/models/crf.py` implements the shared model contract ([docs/MODEL_CONTRACT.md](docs/MODEL_CONTRACT.md)):

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

### Integration Interface Recommendation for Member 6
- **Observation:** Member 6's initial draft in `pipeline.py:ModelAdapter.predict_tokens` calls `self.model.predict(clean_tokens)` passing a raw `List[str]`.
- **Contract Reality:** [docs/MODEL_CONTRACT.md](docs/MODEL_CONTRACT.md) specifies a record-oriented signature: `predict(records) -> list[dict]`.
- **Recommendation:** Member 6 can achieve 100% contract compliance with either of two clean options:
  1. *Direct adapter call:* `self.model.predict_tokens(clean_tokens)`
  2. *Standard record packaging:* `self.model.predict([{"sample_id": "demo", "tokens": clean_tokens}])[0]["predicted_labels"]`

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
- **Unresolved Decision:** Linear-chain CRFs require fully observed labels at every sequence node during standard maximum conditional likelihood estimation. Supporting partially masked/marginalized loss would require custom forward-backward expectation over latent states. This trade-off is recorded under Decision D9; for Phase 5, the model trains on the complete supervised targets, and Member 5 evaluates on masked subsets.

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

48 test cases verify:
- **Category A (Init):** Parameter boundary checks, invalid mode rejection, non-finite regularization rejection.
- **Category B (Training):** Target derivation, empty dataset rejection, duplicate ID rejection, split protection (`test` and `validation`), input immutability.
- **Category C (Prediction):** Record-oriented output schema matching `MODEL_CONTRACT.md`, length preservation, unseen word robustness, predict-before-fit guard.
- **Category D (Feature Integration):** Basic vs improved feature presets with identical model class.
- **Category E (Persistence):** Save/load round-trip equivalence (exact prediction match before and after serialization), corrupt artifact handling.
- **Category F (Modes A/B/C):** 4-, 8-, and 12-class target spaces and joint string format validation.
- **Category G (Configuration Hardening):** Missing config rejection, malformed YAML rejection, structural inconsistency rejection, and provenance tracking.

---

## 7. Scientific Limitations & Boundaries (Phase 4 vs Phase 5)

Passing all 48 synthetic unit tests proves the **software correctness** and **contract compliance** of the model implementation. It does **not** prove:
- High non-O macro F1 scores on conversational speech.
- Generalization to out-of-domain conversations.
- Optimal hyperparameter selection ($c_1, c_2$).
- Superiority over the HMM baseline.

Real-data performance is measured by Member 5, not by these unit tests: validation results for the six full-data models are in [EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md).

---

## 8. Phase 5: Reproducible CRF Training Pipeline (`scripts/train_crf.py`)

**CLI Script:** `scripts/train_crf.py`
**CLI Tests:** `tests/test_train_crf_cli.py` (22 passed)
**Total Test Suite:** 208 tests passed

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

23 automated tests cover:
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
- **209 total tests passing** in pytest (0 regressions):
  - 115 Member 1 data-contract tests.
  - 23 Member 3 feature unit tests (`tests/test_crf_features.py`).
  - 48 Member 3 model unit & contract tests (`tests/test_crf_model.py`).
  - 23 Member 3 training CLI tests (`tests/test_train_crf_cli.py`).
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
- **CRF Benchmark Claims:** The six CRF models were trained on the full official training lists ([CRF_EXPERIMENTS.md](../docs/CRF_EXPERIMENTS.md) §7). Benchmark numbers come only from Member 5's evaluator: validation in [EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md); test results are pending and none may be claimed before they are scored.
- **Configurable vs Tuned Hyperparameters:** CLI regularization options (`--c1`, `--c2`) are user-configurable runtime arguments. No validation-based hyperparameter tuning has been performed; all six models use c1 = c2 = 0.1 and 100 iterations.

### 9.5 Future Planned Work
- Systematic validation grid search for optimal L1/L2 regularization penalties ($c_1, c_2$).
- Feature ablations run on **validation only**, never on test: one feature group is removed at a time from the improved preset and trained on a 50k-sequence sample. Results are in [EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md) §5; `scripts/run_ablations.py --mode {A,B,C} --out DIR` reruns them. The utterance-final shortcut is measured with Member 5's `non_final` view on validation.

---

## 10. Official SponSpeech v2 Dataset Verification Protocol

When the official handoff archive is distributed, Member 3 will execute this 13-point audit protocol before any training. The status column was written before the full-data runs and was not updated; the training manifests' input and ID hashes match the release ([EVALUATION_RESULTS_CRF.md](../docs/EVALUATION_RESULTS_CRF.md), Provenance).

| Check # | Audit Item | Target / Reference Standard | Status |
| :---: | :--- | :--- | :---: |
| 1 | Archive File Name | `sponspeech-m1-v2-handoff.zip` in `artifacts/data/` | **NOT EXECUTED** (awaiting distribution) |
| 2 | Archive Size | Exactly `224,632,100` bytes | **NOT EXECUTED** |
| 3 | Archive Checksum | SHA-256: `1e4b4a9504c68cd8e3af8d6a2a0ef7de5de0146c76457752a56a90dcf6504b9f` | **NOT EXECUTED** |
| 4 | Splits Fingerprint | SHA256SUMS: `40ea6371b038d5d36e6da390493bfc0e45c513ec46c324ee1a6af8766e7546b7` | **NOT EXECUTED** |
| 5 | Processed Fingerprint | SHA256SUMS: `19ed874f9ae3de4489bdc668aaad6581c24b56e00d57f2d671d8a4c2908d4bbb` | **NOT EXECUTED** |
| 6 | Safe Extraction | Extract to `data/processed/sponspeech-m1-v2/` and `data/splits/sponspeech-m1-v2/` | **NOT EXECUTED** |
| 7 | Per-File Checksums | Every extracted JSONL and `.ids` file matches its SHA256SUMS entry | **NOT EXECUTED** |
| 8 | Mode A/B/C Views | Verify files: `mode_A/train.jsonl`, `mode_B/train.jsonl`, `mode_C/train.jsonl`, plus `validation.jsonl` and `test.jsonl` | **NOT EXECUTED** |
| 9 | Frozen ID Lists | Verify `default_train/mode_<A,B,C>.ids` and `primary_eval/validation_mode_<A,B,C>.ids` | **NOT EXECUTED** |
| 10 | Cross-Mode Alignment | Confirm identical `sample_id` and `tokens` across modes for all records | **NOT EXECUTED** |
| 11 | Label Validity | Validate all labels match `configs/label_sets.yaml` (Mode A: 4, B: 8, C: 12) | **NOT EXECUTED** |
| 12 | Policy & Schema | Confirm `schema_version == 1` and `policy_version == "m1-v2"` | **NOT EXECUTED** |
| 13 | Exclusions Audit | Ensure no unresolved clusters or test/val overlap exist in default train | **NOT EXECUTED** |

---

## 11. Team Integration Interface Specification (Kamal & Abdulla)

### 11.1 Target Consumers
- **Kamal (Member 4 — UI):** Interactive demo web application.
- **Abdulla Jamal (Member 6 — Integration & Pipeline):** Unified end-to-end inference pipeline.

### 11.2 Required Input Representation
Input tokens must be:
- A sequence of **lowercased**, whitespace-tokenized string words.
- Strictly gold-free (no ground-truth punctuation or casing hints).
- Free of empty strings (`""`) or non-string objects.
- *Critical Caveat:* Input text must be explicitly lowercased prior to inference. Uppercase characters in input tokens will bypass casing restoration in Mode C and corrupt token shape features.

### 11.3 Interface Signatures

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

### 11.4 Model Loading
```python
from punct_restore.models.crf import CRFModel

# Load serialized bundle (automatically reconstructs feature config & vocabulary)
model = CRFModel.load_model("artifacts/models/crf_mode_a_basic.joblib")
assert model.is_fitted
print("Loaded model mode:", model.label_mode)
print("Vocabulary:", model.vocabulary)
```

### 11.5 Error Handling & Guarantees
- Calling `predict()` on an unfitted model raises `RuntimeError`.
- Empty token sequences or non-string tokens raise `ValueError`.
- Output sequence length is guaranteed to be strictly equal to input token length (`len(predicted_labels) == len(tokens)`).

---

## 12. Corrected Real-Data Smoke Test Protocol (Prepared for Future Approval)

**Scope:** The very first real-data CRF execution to be run only upon explicit user authorization after official SponSpeech v2 archive distribution and verification.

### 12.1 Execution Parameters, Contract Definitions & Safe Locations
- **Dataset Release:** Official verified `sponspeech-m1-v2`.
- **Target Label Mode:** `A` (4 standard punctuation classes: `O`, `COMMA`, `PERIOD`, `QUESTION`).
- **Feature Preset:** `basic` (`FeatureConfig.basic()`, $\pm 1$ context, affixes, shape).
- **Authoritative Model Contract & Identification:**
  - **`model_name` (`"crf_basic"`):** Conforms strictly to `docs/MODEL_CONTRACT.md` ("Use `hmm`, `crf_basic`, or `crf_improved` as initial model names"). Emitted in every prediction record (`"model_name": "crf_basic"`).
  - **`run_id` (`"crf_mode_a_basic_smoke_01"`):** Unique experiment run identifier distinguishing training settings. Tracked in the companion manifest (`crf_mode_a_basic_smoke_01.manifest.json`) and in the model artifact provenance metadata (`training_metadata["provenance"]["run_id"]`).
  - *Contract Rule:* `model_name` and `run_id` must NOT be conflated. Prediction JSONL lines must have `model_name = "crf_basic"`, NOT the run ID.
- **Sequence Ordering Reality:**
  - `scripts/train_crf.py predict` streams line-by-line through `input_path` (`validation.jsonl`), outputting records whose `sample_id` belongs to the target set.
  - Therefore, prediction JSONL strictly preserves **source JSONL sequence order**, NOT the line order in the `.ids` file.
  - Smoke verification assertions must never assume `.ids` file ordering. Instead, they must validate:
    1. Unique IDs (no duplicate sequences).
    2. Exact set equality and cardinality match with the frozen ID set (`set(pred_ids) == set(target_ids)`).
    3. Strict vocabulary containment (`predicted_labels` $\subseteq$ Mode A vocabulary).
    4. Exact 1-to-1 token-to-label alignment matching corresponding input tokens.
    5. Order matching the occurrence of selected sequences in `validation.jsonl`.
- **Official Immutable Inputs (Read-Only):**
  - Training Data: `data/processed/sponspeech-m1-v2/mode_A/train.jsonl`
  - Validation Data: `data/processed/sponspeech-m1-v2/mode_A/validation.jsonl`
  - Official Train IDs: `data/splits/sponspeech-m1-v2/default_train/mode_A.ids`
  - Official Primary Validation IDs: `data/splits/sponspeech-m1-v2/primary_eval/validation_mode_A.ids`
- **Isolated Local Experiment Directory:**
  - `experiments/runs/smoke/` (strictly git-ignored under `/experiments/runs/**`).
  - Temporary subsets are NEVER written to official dataset directories (`data/splits/...` remains immutable).
- **Generated Smoke Subsets:**
  - Train Subset (1,000 IDs): `experiments/runs/smoke/smoke_train_mode_A.ids`
  - Validation Subset (100 IDs): `experiments/runs/smoke/smoke_validation_mode_A.ids`
- **Output Artifacts:**
  - Model: `experiments/runs/smoke/crf_mode_a_basic_smoke_01.joblib`
  - Manifest: `experiments/runs/smoke/crf_mode_a_basic_smoke_01.manifest.json`
  - Validation Predictions: `experiments/runs/smoke/crf_mode_a_basic_smoke_01_val_predictions.jsonl`
- **Initial Hyperparameters (Fixed Initial, Untuned):**
  - Algorithm: `lbfgs`
  - $c_1 = 0.1$ (L1 penalty)
  - $c_2 = 0.1$ (L2 penalty)
  - `max_iterations = 50`
  - `all_possible_transitions = True`

### 12.2 Safe Deterministic Subset Generation via Python
To ensure cross-platform compatibility, UTF-8 encoding without BOM, and explicit LF (`\n`) line endings while preserving relative source sequence order:

```powershell
.\.venv\Scripts\python.exe -c "
from pathlib import Path

def extract_subset(src_str: str, dst_str: str, count: int) -> None:
    src, dst = Path(src_str), Path(dst_str)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, 'r', encoding='utf-8') as f_in:
        lines = [line.strip() for line in f_in if line.strip()][:count]
    if len(lines) != count:
        raise ValueError(f'Expected {count} IDs, found {len(lines)} in {src}')
    with open(dst, 'w', encoding='utf-8', newline='\n') as f_out:
        for s_id in lines:
            f_out.write(f'{s_id}\n')
    print(f'Successfully wrote {len(lines)} IDs to {dst} (UTF-8, LF)')

extract_subset('data/splits/sponspeech-m1-v2/default_train/mode_A.ids', 'experiments/runs/smoke/smoke_train_mode_A.ids', 1000)
extract_subset('data/splits/sponspeech-m1-v2/primary_eval/validation_mode_A.ids', 'experiments/runs/smoke/smoke_validation_mode_A.ids', 100)
"
```

### 12.3 Windows PowerShell Monitored Execution Pipeline

All commands use the project's `.venv` interpreter (`.\.venv\Scripts\python.exe`).

#### Step 1: Single Unified Monitored Training Execution (Executed Exactly ONCE)
To prevent accidental double execution against `crf_mode_a_basic_smoke_01.joblib` and ensure clean process metrics on Windows, PowerShell launches the training process via .NET `System.Diagnostics.Process`, continuously sampling `WorkingSet64` (RAM) until completion:

```powershell
# Single Monitored Training Procedure (No duplicate execution)
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$psi.Arguments = "scripts/train_crf.py train --mode A --input data/processed/sponspeech-m1-v2/mode_A/train.jsonl --ids experiments/runs/smoke/smoke_train_mode_A.ids --output experiments/runs/smoke/crf_mode_a_basic_smoke_01.joblib --manifest-out experiments/runs/smoke/crf_mode_a_basic_smoke_01.manifest.json --preset basic --algorithm lbfgs --c1 0.1 --c2 0.1 --max-iterations 50 --run-id crf_mode_a_basic_smoke_01 --dataset-release sponspeech-m1-v2 --policy-version m1-v2"
$psi.UseShellExecute = $false

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$proc = [System.Diagnostics.Process]::Start($psi)

$peakMemBytes = [int64]0
while (-not $proc.HasExited) {
    try {
        $proc.Refresh()
        $ws = $proc.WorkingSet64
        if ($ws -gt $peakMemBytes) { $peakMemBytes = $ws }
    } catch {
        # Guard against process termination between HasExited check and Refresh
    }
    Start-Sleep -Milliseconds 100
}
$proc.WaitForExit()
$sw.Stop()

$exitCode = $proc.ExitCode
$wallTimeSeconds = [math]::Round($sw.Elapsed.TotalSeconds, 2)
$peakMemMB = [math]::Round($peakMemBytes / 1MB, 2)

Write-Host "=========================================================="
Write-Host "RECAP CRF Smoke Training Process Summary:"
Write-Host "  Exit Code:    $exitCode"
Write-Host "  Wall Time:    $wallTimeSeconds s"
Write-Host "  Peak Memory:  $peakMemMB MB"
Write-Host "=========================================================="

if ($exitCode -ne 0) {
    throw "Training failed with exit code $exitCode"
}
```

#### Step 2: Predict on Validation Subset
```powershell
.\.venv\Scripts\python.exe scripts/train_crf.py predict `
    --model experiments/runs/smoke/crf_mode_a_basic_smoke_01.joblib `
    --input data/processed/sponspeech-m1-v2/mode_A/validation.jsonl `
    --ids experiments/runs/smoke/smoke_validation_mode_A.ids `
    --output experiments/runs/smoke/crf_mode_a_basic_smoke_01_val_predictions.jsonl
```

#### Step 3: Inspect Saved Model & Companion Manifest
```powershell
.\.venv\Scripts\python.exe scripts/train_crf.py inspect `
    --model experiments/runs/smoke/crf_mode_a_basic_smoke_01.joblib
```

### 12.4 Evaluator Branch Identity & Unmerged Scaffold Status
- **Official Evaluator Remote Branch:** `remotes/origin/feat/m5-evaluation` (owned by Member 5: Abd Raouf, commit `680438f`: *"feat(eval): add evaluation pipeline, shortcut baseline and CLI"*).
- **Branch Availability & Local Isolation:**
  - The evaluator code exists on GitHub on branch `feat/m5-evaluation`.
  - It has **NOT** been merged into `main` or into our local branch `feat/m3-crf-model`.
  - In our local workspace, `scripts/evaluate_models.py` remains placeholder scaffolding raising `SystemExit`.
  - Under strict university project rules (*"No Git staging, commits, pushes, merges, rebases, or uploads"*), `feat/m5-evaluation` must NOT be merged locally during this workstream.
  - Therefore, smoke test verification does not call `scripts/evaluate_models.py`. Instead, complete contract compliance and metric integrity are verified directly via our authoritative Python verification script.

### 12.5 Smoke Verification Gate (Automated Python Script)
To be executed immediately following Steps 1–3, verifying all contract requirements without assuming `.ids` file line ordering or conflating `model_name` with `run_id`:

```powershell
.\.venv\Scripts\python.exe -c "
import json
from pathlib import Path
import sys

val_input = Path('data/processed/sponspeech-m1-v2/mode_A/validation.jsonl')
val_ids_file = Path('experiments/runs/smoke/smoke_validation_mode_A.ids')
preds_file = Path('experiments/runs/smoke/crf_mode_a_basic_smoke_01_val_predictions.jsonl')
manifest_file = Path('experiments/runs/smoke/crf_mode_a_basic_smoke_01.manifest.json')
model_file = Path('experiments/runs/smoke/crf_mode_a_basic_smoke_01.joblib')

# 1. Verify frozen validation IDs
with open(val_ids_file, 'r', encoding='utf-8') as f:
    target_ids = [line.strip() for line in f if line.strip()]
assert len(target_ids) == 100, f'Expected 100 validation IDs, got {len(target_ids)}'
target_id_set = set(target_ids)
assert len(target_id_set) == 100, 'Duplicate IDs detected in validation ID list'

# 2. Extract ground truth observations and record source JSONL order
source_tokens_by_id = {}
source_ordered_target_ids = []
with open(val_input, 'r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        s_id = rec['sample_id']
        if s_id in target_id_set:
            source_tokens_by_id[s_id] = rec['tokens']
            source_ordered_target_ids.append(s_id)

assert len(source_tokens_by_id) == 100, f'Expected 100 matches in source JSONL, found {len(source_tokens_by_id)}'

# 3. Verify prediction records
with open(preds_file, 'r', encoding='utf-8') as f:
    predictions = [json.loads(line) for line in f if line.strip()]

assert len(predictions) == 100, f'Expected 100 prediction records, got {len(predictions)}'

pred_ids = [p['sample_id'] for p in predictions]

# 3a. Unique IDs assertion
assert len(set(pred_ids)) == len(pred_ids), 'Duplicate sample_ids found in prediction output'

# 3b. Exact selected membership
assert set(pred_ids) == target_id_set, 'Predicted sample_ids do not match expected frozen ID set'

# 3c. Source JSONL stream order verification (CRF CLI preserves source JSONL order, not .ids order)
assert pred_ids == source_ordered_target_ids, 'Predictions must preserve source JSONL sequence ordering'

# 3d. Field contract, vocabulary containment, and 1-to-1 length alignment
mode_a_vocab = {'O', 'COMMA', 'PERIOD', 'QUESTION'}
for pred in predictions:
    s_id = pred['sample_id']
    assert set(pred.keys()) == {'sample_id', 'model_name', 'label_mode', 'predicted_labels'}, f'Invalid keys in {s_id}'
    # Authoritative contract: model_name must be canonical 'crf_basic', NOT run_id
    assert pred['model_name'] == 'crf_basic', f'Expected model_name==\"crf_basic\", got {pred[\"model_name\"]!r}'
    assert pred['label_mode'] == 'A', f'Expected label_mode==\"A\", got {pred[\"label_mode\"]!r}'

    expected_tokens = source_tokens_by_id[s_id]
    pred_labels = pred['predicted_labels']
    assert isinstance(pred_labels, list), f'predicted_labels must be list for {s_id}'
    assert len(pred_labels) == len(expected_tokens), (
        f'Alignment error for {s_id}: {len(pred_labels)} labels vs {len(expected_tokens)} tokens'
    )
    for lbl in pred_labels:
        assert lbl in mode_a_vocab, f'Illegal label {lbl!r} in {s_id} outside Mode A vocabulary'

# 4. Verify companion manifest and distinguish run_id from model_name
with open(manifest_file, 'r', encoding='utf-8') as f:
    manifest = json.load(f)

assert manifest['run_id'] == 'crf_mode_a_basic_smoke_01', f'Manifest run_id mismatch: {manifest.get(\"run_id\")}'
assert manifest['model_name'] == 'crf_basic', f'Manifest model_name mismatch: {manifest.get(\"model_name\")}'
assert manifest['label_mode'] == 'A'
assert manifest['feature_preset'] == 'basic'
assert manifest['provenance']['selected_sequences'] == 1000
assert Path(manifest['artifact']['path']).name == model_file.name

print('==========================================================')
print('CRF Phase 6 Smoke Test Verification Passed with 100% Compliance:')
print('  - 100/100 validation sequences predicted')
print('  - Model Name contract verified: \"crf_basic\"')
print('  - Run ID contract verified in manifest: \"crf_mode_a_basic_smoke_01\"')
print('  - Source JSONL sequence ordering verified')
print('  - Unique IDs and exact set membership verified')
print('  - 1-to-1 token-to-label alignment verified')
print('  - Mode A vocabulary containment verified')
print('==========================================================')
"
```

*Status: Fully specified and verified. NOT EXECUTED during pre-approval.*
