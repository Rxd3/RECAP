# RECAP — CRF Modeling, Experiments & Reproducibility Guide

**Workstream:** Member 3 (CRF Architecture, Feature Engineering, Training & Reproducibility)  
**Author:** Omar Alzyod  
**Branch:** `feat/m3-crf-model`  
**Last Verified Commit:** `aa232be`  
**Status:** Evaluated by Member 5 in [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md): test results final (§1–4), validation selection in §5.

---

## 1. Member 3 Responsibilities & Scope

Member 3 owns the Conditional Random Field (CRF) modeling subsystem for RECAP Project 2 (English punctuation and casing restoration on speech transcripts):

- **Architecture & Implementation:** Design and implementation of linear-chain CRF models using `sklearn-crfsuite` / `python-crfsuite`.
- **Feature Engineering:** Development of both the baseline (`basic`) and enriched (`improved`) feature extraction presets for English text.
- **Model Training:** Execution and performance tracking of all six full-data CRF models across Modes A, B, and C.
- **Prediction Streaming:** Deterministic export of primary validation predictions adhering strictly to the shared JSONL evaluation contract.
- **Scientific Reproducibility:** Cryptographic verification of model artifacts, manifests, prediction streams, and handoff packaging.
- **Boundary of Ownership:** Official evaluation scripts, scoring algorithms, and benchmark metric publication belong exclusively to **Member 5**. Member 3 does not compute unofficial F1 scores or claim relative superiority between models.

---

## 2. CRF Backend Architecture

- **Engine:** Linear-chain Conditional Random Field implemented via `python-crfsuite` (v0.9.12) and wrapped by `sklearn-crfsuite` (v0.5.0).
- **Optimization Algorithm:** Limited-memory BFGS (L-BFGS).
- **Regularization:** Combined Elastic-Net style penalty with L1 coefficient $c_1 = 0.1$ and L2 coefficient $c_2 = 0.1$.
- **Transition Parameterization:** All ordered transitions between active vocabulary states are modeled (`all_possible_transitions = True`).
- **Sparsity & Serialization:** CRFSuite serializes learned transition and state features into sparse dictionaries. Feature weights that are driven to exact $0.0$ by L1 regularization are omitted from disk serialization.

---

## 3. Feature Configurations: Basic vs Improved Presets

Feature extraction is implemented in [`src/punct_restore/features/crf_features.py`](../src/punct_restore/features/crf_features.py):

### 3.1. Basic Baseline Preset (`basic` / `v1-basic`)
A compact, low-latency token-level feature set designed as a fast baseline:
- **Context Window:** $[-1, +1]$ (`window_size = 1`).
- **Surface Attributes:** Lowercased token string (`token`).
- **Character Affixes:** Prefix and suffix character n-grams of length 1, 2, and 3 (`pref1..3`, `suff1..3`).
- **Token Shape & Orthography:** Token character length (`len`), length bucket (`1`, `short` $\le 3$, `medium` $\le 7$, `long`), numeric check (`is_numeric`), digit presence (`has_digit`).
- **Internal Punctuation:** Boolean indicators for hyphen (`has_hyphen`) or apostrophe (`has_apostrophe`).
- **Sequence Boundary Cues:** Explicit utterance boundary flags (`BOS`, `EOS`, `[-1:BOS]`, `[+1:EOS]`).
- **Contextual Neighbors:** Adjacent token string and 3-gram affixes (`[-1:token]`, `[+1:token]`, `[-1:pref3]`, `[+1:pref3]`, `[-1:suff3]`, `[+1:suff3]`).

### 3.2. Improved Preset (`improved` / `v1-improved`)
An enriched feature set implementing linguistically motivated contextual signals:
- **Extended Context Window:** $[-2, +2]$ (`window_size = 2`), adding two-step context: `[-2:token]`, `[+2:token]`, `[-2:pref3]`, `[+2:pref3]`, `[-2:suff3]`, `[+2:suff3]`.
- **Contextual Token Bigrams:** `use_token_ngrams = True`, extracting preceding bigram `bigram_prev` ($w_{t-1}\_w_t$) and succeeding bigram `bigram_next` ($w_t\_w_{t+1}$).
- **Question Indicators & Syntactic Inversion:** `use_question_cues = True`:
  - Interrogative words (`WH_WORDS`: *what, why, when, where, who, whom, whose, how, which*): `is_wh`, `[-1:is_wh]`, `utt_starts_wh`.
  - Auxiliary verbs (`AUXILIARY_VERBS`: *do, did, does, is, are, was, were, am, can, could, will, would, should, shall, may, might, must, have, has, had*).
  - Subject pronouns (`SUBJECT_PRONOUNS`: *you, we, they, he, she, it, i, there*).
  - Subject-auxiliary inversion detection: `q_inversion_curr` (auxiliary at $t-1$ followed by subject pronoun at $t$), `q_inversion_next`, `utt_starts_aux`.
- **Clause Markers & Discourse Particles:** `use_clause_cues = True`:
  - Conjunctions (`CONJUNCTIONS`: *and, but, or, nor, for, yet, so, because, although, though, since, unless, while, whereas, if*): `is_conjunction`, `[-1:is_conj]`.
  - Discourse particles (`DISCOURSE_MARKERS`: *however, therefore, moreover, furthermore, meanwhile, actually, basically, well, anyway, first, second, finally*): `is_discourse`.
- **Mode C Casing Indicators:** `use_casing_cues = True`:
  - First-person singular pronoun: `is_pronoun_i` (`token == "i"`).
  - Honorific titles (`HONORIFICS`: *mr, mrs, ms, dr, prof*): `is_honorific`.
  - Calendar terms (`DAYS_AND_MONTHS`: days of the week, months): `is_calendar`.
- **Positional Bucketing:** `use_positional_buckets = True`: `pos_bucket` relative position (`start`, `early`, `middle`, `late`, `end`, `single`).
- **Baseline Inclusions:** Retains all shape, affix, internal punctuation, and boundary features from the basic preset.

---

## 4. Label Vocabularies Across Modes A, B, and C

All label definitions conform strictly to [`configs/label_sets.yaml`](../configs/label_sets.yaml):

| Mode | Task Type | Class Count | Label Vocabulary | Description |
|:---:|:---:|:---:|---|---|
| **A** | Punctuation | 4 | `O`, `COMMA`, `PERIOD`, `QUESTION` | Standard punctuation restoration. |
| **B** | Punctuation | 8 | `O`, `COMMA`, `PERIOD`, `QUESTION`, `EXCLAMATION`, `COLON`, `SEMICOLON`, `DASH` | Extended punctuation restoration only (no casing tokens). |
| **C** | Joint Punctuation & Casing | 12 | `O\|LOWER`, `O\|FIRST_UPPER`, `O\|ALL_UPPER`, `COMMA\|LOWER`, `COMMA\|FIRST_UPPER`, `COMMA\|ALL_UPPER`, `PERIOD\|LOWER`, `PERIOD\|FIRST_UPPER`, `PERIOD\|ALL_UPPER`, `QUESTION\|LOWER`, `QUESTION\|FIRST_UPPER`, `QUESTION\|ALL_UPPER` | Joint standard punctuation and lexical casing separated by `\|`. |

---

## 5. Dataset Releases & Frozen Partition Selections

- **Dataset Release:** `sponspeech-m1-v2` (Policy version: `m1-v2`).
- **Source Transcripts:** SponSpeech English informal (podcast-style) speech transcripts.
- **Training Sets:**
  - **Modes A & B:** 142,722 sequences, 4,137,833 tokens (`data/splits/sponspeech-m1-v2/default_train/mode_A.ids`, SHA-256: `593196fea62d...`).
  - **Mode C:** 141,745 sequences, 4,112,054 tokens (`data/splits/sponspeech-m1-v2/default_train/mode_C.ids`, SHA-256: `02df8b60422d...`).  
    *(977 sequences were filtered out due to ill-formed casing annotations per the official dataset policy).*
- **Primary Validation Sets:**
  - **Modes A & B:** 20,040 sequences, 618,771 tokens (`data/splits/sponspeech-m1-v2/primary_eval/validation_mode_A.ids`, SHA-256: `15e43a80088a...`).
  - **Mode C:** 19,936 sequences, 616,940 tokens (`data/splits/sponspeech-m1-v2/primary_eval/validation_mode_C.ids`, SHA-256: `c7300a84a715...`).

---

## 6. Training Hyperparameters

All six models were trained under identical solver parameters:
- `algorithm`: `lbfgs`
- `c1` (L1 regularization): `0.1`
- `c2` (L2 regularization): `0.1`
- `max_iterations`: `100`
- `all_possible_transitions`: `True`

---

## 7. The Six Full-Data Model Runs

| # | Run Identifier | Mode | Preset | Model Artifact SHA-256 (`.joblib`) | Model File Size | Prediction Stream SHA-256 (`.jsonl`) |
|---|---|:---:|:---:|---|---:|---|
| 1 | `crf_mode_a_basic_full_01` | A | basic | `78619f5e7b683ca733abbc8374f93e069a0e5932d064f44deef8765cb3d22bb6` | 13.01 MB | `cffd55c68a2880ea7d7873649443c7ae79f46b065b14b68a9d486c2431498a48` |
| 2 | `crf_mode_b_basic_full_01` | B | basic | `9cc54b5999852591507c0cb46e05646a135e6a1ae8926d34dcd408e662108e35` | 13.73 MB | `dcbd3656a793d2a8b5280b2ee0c972ac05368e3ff84737056ef10dba5e4c918c` |
| 3 | `crf_mode_c_basic_full_01` | C | basic | `3ea2d3cf719cce3fb3630cfe7c76d0bb059c7926fb22a9dfa7a6f558938deed3` | 17.89 MB | `8b17c8710ef57a745b48bbfcf64a1c73f6b51cd2d5505ed2e9226756eaaafeaf` |
| 4 | `crf_mode_a_improved_full_01` | A | improved | `f9e50d34528432507fcdb2f1c570408ffce743aa94ecc5a2024e00a0687a6b55` | 74.46 MB | `218a036efc063a20748f2dcca60813ccb179332452d2f15a8cc9a6f09ae58ecf` |
| 5 | `crf_mode_b_improved_full_01` | B | improved | `dbac5712db7d14406d676f212905684ae0d70ebc74dfbba2ae2f63cc973a6f98` | 74.65 MB | `8a84db7ba3fe99bf39b3c0efa70601f066fc49192da666538b006dbbe96adb2c` |
| 6 | `crf_mode_c_improved_full_01` | C | improved | `632671bb24430df39f7c9a5d210cedffa0df6312a824343114cb76a4588a161c` | 102.22 MB | `7bb0d788d0eadd77e78779b51a162ab16d71bfb200c44ca40320a2689cb3124d` |

---

## 8. Artifact & Prediction Conventions

All generated artifacts adhere to standard directory structures and schemas:

1. **Model Binaries (`.joblib`):** Stored under `experiments/runs/<run_id>/<run_id>.joblib`. Git-ignored due to size.
2. **Companion Manifests (`.manifest.json`):** Record full training environment, code fingerprints, dataset hashes, optimizer diagnostics, and parameter counts.
3. **Prediction Streams (`.jsonl`):** Streamed JSON records with one validation sequence per line:
   ```json
   {
     "sample_id": "sponspeech:dev:1StHS716-D0-1",
     "model_name": "crf_basic",
     "label_mode": "A",
     "predicted_labels": ["O", "COMMA", "O", "PERIOD"]
   }
   ```
   *Note:* The `model_name` field records the feature preset family (`crf_basic` or `crf_improved`), while the filename and manifest identify the specific run identifier.

---

## 9. Reproducibility Commands

To reproduce any training run or prediction export, run these from `release/final`. Mode A basic retrained from the `aa232be` code gives byte-identical predictions, and all six validation files reproduce from the shared models ([EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md), Provenance):

### 9.1. Training Command Template
```powershell
.\.venv\Scripts\recap.exe train-crf train `
  --mode <A|B|C> `
  --input data/processed/sponspeech-m1-v2/mode_<A|B|C>/train.jsonl `
  --ids data/splits/sponspeech-m1-v2/default_train/mode_<A|B|C>.ids `
  --output experiments/runs/<run_dir>/<run_id>.joblib `
  --manifest-out experiments/runs/<run_dir>/<run_id>.manifest.json `
  --preset <basic|improved> `
  --algorithm lbfgs `
  --c1 0.1 --c2 0.1 --max-iterations 100 `
  --run-id <run_id> `
  --dataset-release sponspeech-m1-v2 `
  --policy-version m1-v2
```

### 9.2. Primary Validation Prediction Export Template
The label mode and `model_name` come from the saved artifact. `--manifest-out` writes the prediction manifest that `recap evaluate-models score --manifest` checks.
```powershell
.\.venv\Scripts\recap.exe train-crf predict `
  --model experiments/runs/<run_dir>/<run_id>.joblib `
  --input data/processed/sponspeech-m1-v2/mode_<A|B|C>/validation.jsonl `
  --ids data/splits/sponspeech-m1-v2/primary_eval/validation_mode_<A|B|C>.ids `
  --output experiments/runs/<run_dir>/<run_id>_val_predictions.jsonl `
  --manifest-out experiments/runs/<run_dir>/<run_id>_val_predictions.manifest.json
```

---

## 10. Known Scientific & Technical Limitations

1. **Optimizer Budget vs Convergence:**  
   All six models stopped at the iteration budget (`max_iterations = 100`). The training manifests record only `num_iterations = 100`; the loss traces in the shared models are summarised in `results/scores/crf_convergence/convergence.json` (`recap crf-convergence`): the loss was still falling by 0.12–1.52% over the last 10 iterations, so formal convergence is not claimed.
2. **Dataset Partitioning (Source/Video Overlap):**  
   Training uses the `default_train` lists and validation uses the `primary_eval` lists. Those primary lists share no source video, no exact or normalized transcript and no qualifying six-gram with their reference splits ([DATASET_PLAN.md](DATASET_PLAN.md), [DATA_RELEASE_VALIDATION.md](DATA_RELEASE_VALIDATION.md)). Channel overlap remains: 149 of 522 channels span the original splits.
3. **Mode C Transition Sparsity:**  
   The improved Mode C model serializes 143 transition parameters instead of 144. The transition $(\text{COMMA\|ALL_UPPER} \to \text{QUESTION\|ALL_UPPER})$ occurs 0 times in the training corpus; under L1 regularization ($c_1=0.1$), its weight was driven to exact zero and omitted from the sparse parameter dictionary.
