# RECAP Member 3 — Evaluator Handoff Package

**Package Name:** `recap_m3_handoff_eval`  
**Author:** Omar Alzyod (Member 3: CRF Architecture, Feature Engineering, Training & Reproducibility)  
**Target Recipient:** Member 5 (Evaluation Suite & Benchmark Scoring)  
**Branch:** `feat/m3-crf-model`  
**Commit:** `aa232be`  
**Dataset Release:** `sponspeech-m1-v2` (Policy: `m1-v2`)  

---

## Package Overview

This package contains the complete, verified primary validation predictions and training manifests for all six full-data Conditional Random Field (CRF) models developed by Member 3.

```
recap_m3_handoff_eval/
├── README.md                 <- Evaluation instructions and package overview
├── HANDOFF_INVENTORY.md      <- Human-readable table of models, digests, counts, schemas
├── HANDOFF_INVENTORY.json    <- Machine-readable provenance and metadata
├── SHA256SUMS                <- Standard SHA-256 checksums of all prediction and manifest files
├── predictions/              <- 6 primary validation prediction JSONL files (streamed)
│   ├── crf_mode_a_basic_full_01_val_predictions.jsonl
│   ├── crf_mode_b_basic_full_01_val_predictions.jsonl
│   ├── crf_mode_c_basic_full_01_val_predictions.jsonl
│   ├── crf_mode_a_improved_full_01_val_predictions.jsonl
│   ├── crf_mode_b_improved_full_01_val_predictions.jsonl
│   └── crf_mode_c_improved_full_01_val_predictions.jsonl
└── manifests/                <- 6 companion training manifests with hyperparameter & feature counts
    ├── crf_mode_a_basic_full_01.manifest.json
    ├── crf_mode_b_basic_full_01.manifest.json
    ├── crf_mode_c_basic_full_01.manifest.json
    ├── crf_mode_a_improved_full_01.manifest.json
    ├── crf_mode_b_improved_full_01.manifest.json
    └── crf_mode_c_improved_full_01.manifest.json
```

---

## Instructions for Member 5 (Official Evaluator)

### 1. Prediction Stream Format
Each prediction file contains one JSON line per validation sequence adhering strictly to the contract:
```json
{"sample_id": "seq_000001", "model_name": "crf_mode_a_basic_full_01", "label_mode": "A", "predicted_labels": ["O", "COMMA", "O", "PERIOD"]}
```

### 2. Sequence and Token Alignments
- **Modes A & B:** 20,040 sequences, exactly 618,771 tokens matching `data/splits/sponspeech-m1-v2/primary_eval/validation_mode_A.ids`.
- **Mode C:** 19,936 sequences, exactly 616,940 tokens matching `data/splits/sponspeech-m1-v2/primary_eval/validation_mode_C.ids`.

### 3. Mode C Evaluation Masking
- When evaluating Mode C joint predictions and casing sub-tasks, apply the official evaluation mask:  
  `data/processed/sponspeech-m1-v2/masks/case_joint/validation.jsonl`  
  (SHA-256: `6dd419e4f45ebfc91d4bc74eb114e99d5a2798c0410bb4f26453ec3ab832e2f9`).
- Tokens where `eval_mask == False` must be excluded from joint and casing scoring per D4 decision specifications.

### 4. Comparisons & Metrics
- Compute official non-O macro F1 and micro F1 for each model.
- Compare `basic` baseline vs `improved` preset within Mode A, Mode B, and Mode C.
- Member 3 has **not** computed or published official F1 scores; all benchmark scoring belongs exclusively to Member 5.

### 5. Why Model Binaries Are Not Included
Member 5's evaluator operates purely on the exported prediction JSONL streams and gold label files. Binary model weights (`.joblib` files, ~295 MB) are retained locally under `experiments/runs/` for inference demonstrations and need not be transferred.
