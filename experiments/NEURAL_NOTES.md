# Neural comparisons

Status on 2026-10-31: all four real training/validation selections are complete on the requested local Windows PC.
`nvidia-smi` confirms an NVIDIA GeForce RTX 5070, 12,227 MiB VRAM, driver 617.42.
The isolated `.venv` uses 64-bit Python 3.13.14 and torch 2.9.1+cu128; a real
512-by-512 float32 CUDA matrix multiplication matched the CPU calculation
(maximum absolute error 0.0). The GPU has compute capability 12.0, the wheel
includes sm_120, and bf16 is supported. The machine has 16 GB physical RAM;
bootstrap found about 193 GB free disk. Preflight and all four real selections
are complete. Every required candidate and final validation/runtime record is
preserved in the frozen-run entries below. Held-out prediction and the final
handoff remain pending; all four records are sealed in this commit first.
Drive/Gmail account connections for final delivery remain pending; GitHub access
as Rxd3 works through the PC's existing credential. The deadline is Saturday
2026-10-31 at 18:00 +03:00.

The supplied ZIP, all six checksummed bundle members, all 27 processed files and
all 20 split files passed SHA-256 verification. Both release checksum manifests
match their published repository copies byte for byte. Held-out files were read
only as bytes for checksums, with no examples inspected. The selected training
lists contain 142,722 utterances / 4,137,833 words for A and B, and 141,745 /
4,112,054 for C. No training IDs were removed. Local operational evidence lives
under ignored `artifacts/neural/`, including `RUN_STATUS.md`, environment freezes,
CUDA/data verification JSON and durable job logs.

## Deadline scope change (2026-10-31)

At the user's request, the neural study is now limited to `bert_tagger_A`,
`bert_tagger_B`, `t5_seq2seq_A` and `t5_seq2seq_B` because of the deadline.
The two mode C neural runs are omitted before training starts. Existing HMM/CRF
mode C results remain available; this extension makes no neural truecasing or
joint punctuation/casing comparison. This is a reduced neural study, not the
original six-model study.

Completed runs and the active T5-B process are preserved. The A/B study retains
the full training ID lists, both BERT learning rates, all three epochs, four-beam
T5 decoding, validation selection, the required runtime passes, and bootstrap
1000. All four frozen selections and their notes must pass audit and be
committed before any held-out inference. Each included model predicts each
held-out split once, preserving its exclusive receipt.

The scoped handoff contains 12 prediction/manifest pairs: four models times
validation, test and test_amb. Its name is `neural-predictions-AB.zip`, with mode C
explicitly excluded. The original six-model CLI remains unchanged; distinct
ignored A/B operational helpers use the same official evaluator and bootstrap
settings. The existing T5-B timing still puts the complete A/B handoff after
the 2026-10-31 18:00 +03:00 deadline. A deadline snapshot may include only verified
completed validation runs and must be identified as an interim packet.

## Models and fixed settings

- `bert_tagger`: `bert-base-uncased`, one token classifier per A/B/C mode.
  Labels supervise the first subword only; special tokens and later subwords use
  `-100`. Inputs are split words with `is_split_into_words=True`. Truncate to 512
  pieces. A word retained only partially is also cut off: its first piece receives
  no training loss, and its prediction is O (C: O|LOWER). Each candidate records
  training cutoff counts and validation cutoff counts/share; prediction manifests
  record the counts/share for their own split.
- `t5_seq2seq`: `google/flan-t5-base`, one model per mode; `t5-small` is an explicit
  memory fallback, requiring a separate new run and validation selection. Input is
  single-space-joined words. Targets use `pipeline.reconstruct`; A/B retain lowercase
  and C restores casing. Source and target sequences are dynamically padded without
  truncation. Generation uses four beams, no sampling, with exactly `2 * words + 16`
  new tokens per utterance. Equal-length inputs are grouped, then input order is
  restored. Greedy decoding is available as a validation-time choice.
- Alignment strips trailing `,.?!:;`, remembers the last mark in text order, and
  attaches a standalone `-` to the preceding word. Internal/attached hyphens remain
  lexical. Generated words are lowercased only for `SequenceMatcher(...,
  autojunk=False)`; only matching blocks assign labels. Dropped or changed input
  words get defaults, inserted words assign no input label. C casing uses the
  generated surface; extended marks in A/C use the configured D5 projection.
  A punctuation-only whitespace piece attaches to the preceding word. Manifests
  report unaligned input word counts and share for every split.
- Seed 2026, AdamW with weight decay 0.01, constant learning rate, gradient norm
  clip 1.0, default batch size 4 and four accumulation steps; the final short group
  uses its actual size. Gradient checkpointing reduces memory. BERT uses fp16 on
  CUDA. T5 uses bf16 when supported (otherwise float32) to avoid fp16 overflow in
  the pretrained T5 activations. CPU verification uses float32.
- BERT compares learning rates 2e-5 and 5e-5, evaluating checkpoints after epochs
  1, 2 and 3. T5 defaults to LR 1e-4 and evaluates those epochs; 3e-4 is also
  accepted as an explicit validation-only experiment. No early stopping.

## Selection and test discipline

Train only on `mode_X/train.jsonl` filtered by `default_train/mode_X.ids`.
Every candidate exports primary validation predictions and an evaluator-compatible
manifest. Selection invokes the unchanged official evaluator on
`primary_eval/validation_mode_X.ids`; C requires its validation casing masks.
Use the evaluator's `primary.value`: A/B punctuation/non-O macro F1, C
`joint/unmasked` macro F1 over 12 labels. Equal scores prefer fewer epochs, then
the smaller learning rate. This uses the evaluator's recorded six-decimal values.

The training CLI writes a complete `selection.json` with every candidate, input
hashes, source hashes, environment, run ID, resolved pretrained revision, artifact
metadata checksum and selected settings. It is frozen before any final inference.
The full record and validation runtime are appended below by the training CLI;
the same record is in the training manifest. Until training completes, there is
no selection JSON and no selected model.

Held-out inference requires that frozen selection and its exact artifact. Each
test split gets an exclusive receipt beside the selection before reading inputs;
another attempt fails even with a different output path. An interrupted attempt
leaves a `started` receipt for explicit investigation, never an automatic retry.
Validation permits warm-up and repeated timing; test/test_amb run once each.
All outputs refuse existing files. Failed runs retain evidence and need a new
run directory. Never remove a receipt to obtain a better test score.

## Runtime and reproducibility

Pinned extra: torch 2.9.1, transformers 4.57.3, sentencepiece 0.2.1. The CUDA
build is installed separately from the official cu128 index for the RTX 5070;
the CPU wheel is sufficient for the download-free tests. Existing pins stay
unchanged. Artifacts save model and tokenizer plus ordered labels, mode, exact
pretrained revision, provenance and checksums of every artifact file; loading
rejects incompatible or changed artifacts. Prediction manifests include input
and ID hashes, artifact and selection identity, runtime, diagnostics and versions.

Training seconds measure each epoch including tokenization, collation and
optimization, with CUDA synchronization, excluding checkpoint writing and
validation. Cumulative selected training seconds and all-search training seconds
are recorded separately. Validation latency is the median ms/utterance of three
full passes after one full warm-up pass, synchronized on GPU, including
tokenization, forward/generation and word alignment, excluding model/data loading.
Record the exact GPU, CUDA, OS, Python, packages and batch size with each real run.

The published handoff ZIP was downloaded from the team's Drive and verified:
224,632,100 bytes; SHA-256
`1e4b4a9504c68cd8e3af8d6a2a0ef7de5de0146c76457752a56a90dcf6504b9f`.
Existing local v2 data is available; held-out examples have not been inspected.
The handoff was unpacked under ignored artifact storage; all six checksummed bundle members
passed, and the processed/split checksum lists match the already
unpacked release in `data/`.

RTX 5070 PC verification on Python 3.13.14 / Windows: full `.[dev,neural]` suite
**1,338 passed in 46.30 s**. A separate clean `tmp/base-test-env` with `.[dev]`
and torch confirmed absent passed **1,331 tests, 1 skipped in 36.72 s**.
`pip check` passes. Current neural versions are torch 2.9.1+cu128, transformers
4.57.3, sentencepiece 0.2.1, tokenizers 0.22.2 and safetensors 0.8.0. Verification
uses four OpenMP threads; tensor fixtures remain CPU-only. Logs and complete
environment freezes are retained under ignored `artifacts/neural/`.

Both backbones passed a CUDA memory preflight on the four longest selected mode C
training utterances (384, 224, 211 and 180 words), with gradient checkpointing,
batch size 4 and an actual Adam update. Peak allocated memory was 2,227,245,568
bytes for BERT fp16 and 5,070,731,264 bytes for FLAN-T5-base bf16; peak reserved
memory was 2,407,530,496 and 5,406,457,856 bytes. BERT's fp16 scaler skipped two
initial updates before allocating optimizer state; the final check includes that
state (199 parameter entries; T5 282). Preflight weights were discarded. No
backbone, batch or decoding fallback was required. Full runs retain batch 4,
accumulation 4 and four T5 beams. The T5 tokenizer's generic 512-token warning
does not change the documented untruncated relative-position policy; its
534-piece preflight batch completed successfully.

Resolved pretrained revisions, fixed for all subsequent real runs:
`bert-base-uncased` at `86b5e0934494bd15c9632b12f734a8a67f723594`, and
`google/flan-t5-base` at `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`.
The optional download helper hf_xet 1.8.0 is installed in the isolated environment;
required dependency pins are unchanged. Preliminary memory logs are preserved;
the final gates are `bert-memory-verified.json` and `t5-memory-verified.json`.

Previous laptop verification on Python 3.13.5, Windows: full suite with `.[dev,neural]`:
**1,338 passed**. Separate fresh `.[dev]` environment, with torch confirmed absent:
**1,331 passed, 1 skipped**. Optional tensor tests build a tiny BERT and T5 locally;
they download no weights/tokenizers and use CPU only. The tiny complete training
fixture exercised both BERT LRs, official validation scoring and bootstrap 1000,
freeze/save/load, runtime passes and refusal to repeat held-out inference. These
fixture scores are not SponSpeech results. `pip check` reports no broken requirements.
The verified neural environment uses torch 2.9.1+cpu, transformers 4.57.3,
sentencepiece 0.2.1, tokenizers 0.22.2 and safetensors 0.8.0.

## Limitations and remaining delivery work

The frozen-run records below are the source for empirical neural scores and GPU
runtimes. The revised A/B handoff of 12 prediction JSONL files plus 12 manifests,
held-out bootstrap scores, the scoped ZIP, its Drive upload and the link to Abd
Raouf remain pending. Mode C is omitted by the user's deadline scope change.
The original packaging command requires all six selections and all 18 prediction
pairs. The distinct A/B helper requires all four included frozen selections and
all 12 pairs, verifies them with the unchanged official scorer and bootstrap
1000, and creates a ZIP containing only predictions, manifests and SHA256SUMS.

T5 can alter words; matching-block defaults deliberately penalize those edits.
Long BERT inputs lose tail labels under the permitted truncation policy. T5
generation limits may stop before the whole text is copied. These effects are
measured in the manifests instead of being repaired with gold labels.

## Frozen run bert_tagger_A

```json
{
  "candidates": [
    {
      "artifact": "lr2e-05-epoch1/model",
      "artifact_metadata_sha256": "9ddb8cc4f1306cc67e347ab6e240fecf92e88c701941af7c50540503584215cb",
      "diagnostics": {
        "cutoff_share": 0.0,
        "cutoff_words": 0,
        "words": 618771
      },
      "epoch": 1,
      "epoch_seconds": 1298.5278603000042,
      "learning_rate": 2e-05,
      "mean_batch_loss": 0.12445613388956268,
      "pretrained": "bert-base-uncased",
      "pretrained_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
      "primary": {
        "block": "punctuation/all",
        "metric": "macro_f1",
        "value": 0.758433
      },
      "score": {
        "path": "experiments/runs/neural/bert_tagger_A/lr2e-05-epoch1/validation.score.json",
        "sha256": "38427a3a1a92012e7224dfabbbcebbdab74913c662c50c95e698715589b534fa"
      },
      "training_diagnostics": {
        "cutoff_words": 0,
        "words": 4137833
      },
      "training_seconds": 1298.5278603000042
    },
    {
      "artifact": "lr2e-05-epoch2/model",
      "artifact_metadata_sha256": "72e8d2b6a7011232a5d591a7e4e690fb72eed7619e18c3e6572390aa9e21b427",
      "diagnostics": {
        "cutoff_share": 0.0,
        "cutoff_words": 0,
        "words": 618771
      },
      "epoch": 2,
      "epoch_seconds": 1326.5467923999968,
      "learning_rate": 2e-05,
      "mean_batch_loss": 0.09682080851015286,
      "pretrained": "bert-base-uncased",
      "pretrained_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
      "primary": {
        "block": "punctuation/all",
        "metric": "macro_f1",
        "value": 0.765072
      },
      "score": {
        "path": "experiments/runs/neural/bert_tagger_A/lr2e-05-epoch2/validation.score.json",
        "sha256": "8c140e9347eb79d4260eb2f4dfd46caabca70b74389c34c8ef8d9bdcb7df1a6f"
      },
      "training_diagnostics": {
        "cutoff_words": 0,
        "words": 4137833
      },
      "training_seconds": 2625.074652700001
    },
    {
      "artifact": "lr2e-05-epoch3/model",
      "artifact_metadata_sha256": "b73b68007f8289d5e75cd9b0df47c47416f8fbca000a277402ece18e7b1bd705",
      "diagnostics": {
        "cutoff_share": 0.0,
        "cutoff_words": 0,
        "words": 618771
      },
      "epoch": 3,
      "epoch_seconds": 1219.8906589000107,
      "learning_rate": 2e-05,
      "mean_batch_loss": 0.08115140286751309,
      "pretrained": "bert-base-uncased",
      "pretrained_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
      "primary": {
        "block": "punctuation/all",
        "metric": "macro_f1",
        "value": 0.767223
      },
      "score": {
        "path": "experiments/runs/neural/bert_tagger_A/lr2e-05-epoch3/validation.score.json",
        "sha256": "1e7bcad83d300f6f498f0ea469a2341ee66fa138d16a9d972258bfff21f09b81"
      },
      "training_diagnostics": {
        "cutoff_words": 0,
        "words": 4137833
      },
      "training_seconds": 3844.965311600012
    },
    {
      "artifact": "lr5e-05-epoch1/model",
      "artifact_metadata_sha256": "6590fb074e36cd7b12770ff4fa7481ccf67ba9682988d5b32c83ccb6b427b4c3",
      "diagnostics": {
        "cutoff_share": 0.0,
        "cutoff_words": 0,
        "words": 618771
      },
      "epoch": 1,
      "epoch_seconds": 1228.678045800014,
      "learning_rate": 5e-05,
      "mean_batch_loss": 0.12361273991620261,
      "pretrained": "bert-base-uncased",
      "pretrained_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
      "primary": {
        "block": "punctuation/all",
        "metric": "macro_f1",
        "value": 0.741213
      },
      "score": {
        "path": "experiments/runs/neural/bert_tagger_A/lr5e-05-epoch1/validation.score.json",
        "sha256": "d7586422ec484164ede486971672b855d7ecd7ed01256dd96cc7d64a4852afe0"
      },
      "training_diagnostics": {
        "cutoff_words": 0,
        "words": 4137833
      },
      "training_seconds": 1228.678045800014
    },
    {
      "artifact": "lr5e-05-epoch2/model",
      "artifact_metadata_sha256": "27cc7462990d9bae511d56911dbd2471c7c9ff95bc080bdca9e77ec03f31bbaa",
      "diagnostics": {
        "cutoff_share": 0.0,
        "cutoff_words": 0,
        "words": 618771
      },
      "epoch": 2,
      "epoch_seconds": 1214.2209104000067,
      "learning_rate": 5e-05,
      "mean_batch_loss": 0.0975904981660391,
      "pretrained": "bert-base-uncased",
      "pretrained_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
      "primary": {
        "block": "punctuation/all",
        "metric": "macro_f1",
        "value": 0.75644
      },
      "score": {
        "path": "experiments/runs/neural/bert_tagger_A/lr5e-05-epoch2/validation.score.json",
        "sha256": "0c09eaf04d9f4d114773eaa3e2ba2e44dd9ecdb0ed4b5c8a7a0d1c9280269986"
      },
      "training_diagnostics": {
        "cutoff_words": 0,
        "words": 4137833
      },
      "training_seconds": 2442.8989562000206
    },
    {
      "artifact": "lr5e-05-epoch3/model",
      "artifact_metadata_sha256": "d08e4ec2d7dc9b838806e1462e463b74a32cd1a991b3554b30e56095818c5a9e",
      "diagnostics": {
        "cutoff_share": 0.0,
        "cutoff_words": 0,
        "words": 618771
      },
      "epoch": 3,
      "epoch_seconds": 1215.0233414999966,
      "learning_rate": 5e-05,
      "mean_batch_loss": 0.0817469020097458,
      "pretrained": "bert-base-uncased",
      "pretrained_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
      "primary": {
        "block": "punctuation/all",
        "metric": "macro_f1",
        "value": 0.757819
      },
      "score": {
        "path": "experiments/runs/neural/bert_tagger_A/lr5e-05-epoch3/validation.score.json",
        "sha256": "1168b2f056a484394cf75815df7af834409626ef0f86786fc20dc668372fe396"
      },
      "training_diagnostics": {
        "cutoff_words": 0,
        "words": 4137833
      },
      "training_seconds": 3657.922297700017
    }
  ],
  "command": [
    "scripts/train_neural_tagger.py",
    "train",
    "--mode",
    "A",
    "--input",
    "data/processed/sponspeech-m1-v2/mode_A/train.jsonl",
    "--ids",
    "data/splits/sponspeech-m1-v2/default_train/mode_A.ids",
    "--validation-input",
    "data/processed/sponspeech-m1-v2/mode_A/validation.jsonl",
    "--validation-ids",
    "data/splits/sponspeech-m1-v2/primary_eval/validation_mode_A.ids",
    "--output",
    "experiments\\runs\\neural\\bert_tagger_A",
    "--manifest-out",
    "experiments\\runs\\neural\\bert_tagger_A\\training.manifest.json",
    "--selection-out",
    "experiments\\runs\\neural\\bert_tagger_A\\selection.json",
    "--run-id",
    "bert_tagger_A",
    "--epochs",
    "3",
    "--batch-size",
    "4",
    "--gradient-accumulation",
    "4",
    "--revision",
    "86b5e0934494bd15c9632b12f734a8a67f723594"
  ],
  "criterion": "official evaluator primary macro F1; ties: fewer epochs, smaller LR",
  "environment": {
    "code_commit": "6654398b00b2dc8361fb7e2aae869aad63d8fad6",
    "cuda": "12.8",
    "device": "cuda",
    "dirty": false,
    "gpu": "NVIDIA GeForce RTX 5070",
    "packages": {
      "safetensors": "0.8.0",
      "sentencepiece": "0.2.1",
      "tokenizers": "0.22.2",
      "torch": "2.9.1+cu128",
      "transformers": "4.57.3"
    },
    "platform": "Windows-11-10.0.26300-SP0",
    "python": "3.13.14",
    "source_sha256": {
      "configs/label_sets.yaml": "eeba7991ce7e897fd610ed1928d6939242b2e76806e5839b63903f6e92fa1142",
      "pyproject.toml": "4c90a49ff4155d1d3314d1a2870b45f31a7dcb02f3b9eee0e4c59cc08ea88213",
      "scripts/train_neural_tagger.py": "8f83811a6ee33645ffbeae43f24dcf82cbb79903306ed7789ca17c75b79b8782",
      "scripts/train_seq2seq.py": "5fcec9068d81a6656207844752a44a3859e2f484cc0a83e0947960d1ee5b339e",
      "src/punct_restore/models/neural_tagger.py": "9bad18347c2eb1fb2d5fb2255bd7cb098b70a677e4adc0ea9202efd03edb86f9",
      "src/punct_restore/models/neural_utils.py": "a702b087229e942d7cee5d3f15ff062a2903602f9e87736c3843546c26c72b13",
      "src/punct_restore/models/neural_workflow.py": "520523345d64f2ab9d121b17751e65962e1dc8257427cf4ff49c941f237714fb",
      "src/punct_restore/models/seq2seq.py": "c8713241c7e0b7a1d1cfa0e1478c6633a9b4917832d0ca578a4d948ac81872ff"
    }
  },
  "frozen": true,
  "frozen_at": "2026-10-30T19:28:20.090879+00:00",
  "inputs": {
    "masks": null,
    "training": {
      "path": "data/processed/sponspeech-m1-v2/mode_A/train.jsonl",
      "sha256": "9412866dd8ae1aaf86e18611f47b1ee8da888792808da0a525ecd61c4bd7c56e"
    },
    "training_ids": {
      "path": "data/splits/sponspeech-m1-v2/default_train/mode_A.ids",
      "sha256": "593196fea62d7aa6eb9eb3bea6649f06f5cc43dab91f048943efa0d9d590492a"
    },
    "validation": {
      "path": "data/processed/sponspeech-m1-v2/mode_A/validation.jsonl",
      "sha256": "7ececedac4e58c307df0daaf6190b61b2ea715344b72a2910d9541e07b1ecf27"
    },
    "validation_ids": {
      "path": "data/splits/sponspeech-m1-v2/primary_eval/validation_mode_A.ids",
      "sha256": "15e43a80088a2189f0f0d34c6e1fc5dd867a2ee42637fc5e80d106c7eed487db"
    }
  },
  "label_mode": "A",
  "model_name": "bert_tagger",
  "run_id": "bert_tagger_A",
  "schema_version": 1,
  "search_training_seconds": 7502.887609300029,
  "selected": {
    "artifact": "lr2e-05-epoch3/model",
    "artifact_metadata_sha256": "b73b68007f8289d5e75cd9b0df47c47416f8fbca000a277402ece18e7b1bd705",
    "diagnostics": {
      "cutoff_share": 0.0,
      "cutoff_words": 0,
      "words": 618771
    },
    "epoch": 3,
    "epoch_seconds": 1219.8906589000107,
    "learning_rate": 2e-05,
    "mean_batch_loss": 0.08115140286751309,
    "pretrained": "bert-base-uncased",
    "pretrained_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
    "primary": {
      "block": "punctuation/all",
      "metric": "macro_f1",
      "value": 0.767223
    },
    "score": {
      "path": "experiments/runs/neural/bert_tagger_A/lr2e-05-epoch3/validation.score.json",
      "sha256": "1e7bcad83d300f6f498f0ea469a2341ee66fa138d16a9d972258bfff21f09b81"
    },
    "training_diagnostics": {
      "cutoff_words": 0,
      "words": 4137833
    },
    "training_seconds": 3844.965311600012
  },
  "settings": {
    "batch_size": 4,
    "dataset_release": "sponspeech-m1-v2",
    "epochs": 3,
    "gradient_accumulation": 4,
    "gradient_clip_norm": 1.0,
    "num_beams": 4,
    "optimizer": "AdamW",
    "policy_version": "m1-v2",
    "precision": "fp16",
    "pretrained": "bert-base-uncased",
    "schedule": "constant",
    "seed": 2026,
    "weight_decay": 0.01,
    "word_alignment": "first complete subword; truncate at 512; default cut words"
  },
  "training_manifest": "training.manifest.json",
  "training_sequences": 142722,
  "training_words": 4137833,
  "validation_runtime": {
    "batch_size": 4,
    "median_ms_per_utterance": 1.8462865419157557,
    "ms_per_utterance_passes": [
      1.8462865419157557,
      1.826662065868296,
      1.8484608333337038
    ],
    "scope": "tokenization + inference/generation + alignment; excludes model/data loading",
    "timed_passes": 3,
    "warmup_passes": 1
  }
}
```

## Frozen run t5_seq2seq_A

```json
{
  "candidates": [
    {
      "artifact": "lr0.0001-epoch1/model",
      "artifact_metadata_sha256": "1c9b7a74d5d523e3d49caac07f931a6eb77d9fe2b7c74fa74d2cb5c211a698c7",
      "diagnostics": {
        "unaligned_share": 0.0005753340088659617,
        "unaligned_words": 356,
        "words": 618771
      },
      "epoch": 1,
      "epoch_seconds": 4242.254812500003,
      "learning_rate": 0.0001,
      "mean_batch_loss": 0.0913295811745817,
      "pretrained": "google/flan-t5-base",
      "pretrained_revision": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
      "primary": {
        "block": "punctuation/all",
        "metric": "macro_f1",
        "value": 0.765599
      },
      "score": {
        "path": "experiments/runs/neural/t5_seq2seq_A/lr0.0001-epoch1/validation.score.json",
        "sha256": "52ce9f149e001623a45483f8e869855a7e9ba8d9cbc51decacbd9c318b3a40ce"
      },
      "training_diagnostics": {
        "cutoff_words": 0,
        "words": 0
      },
      "training_seconds": 4242.254812500003
    },
    {
      "artifact": "lr0.0001-epoch2/model",
      "artifact_metadata_sha256": "6b710c33ca5fef46358e068b582a5026cde92d89059ad026d581fcb084f04c52",
      "diagnostics": {
        "unaligned_share": 0.0006173527847943746,
        "unaligned_words": 382,
        "words": 618771
      },
      "epoch": 2,
      "epoch_seconds": 4098.606896799989,
      "learning_rate": 0.0001,
      "mean_batch_loss": 0.07406434465455841,
      "pretrained": "google/flan-t5-base",
      "pretrained_revision": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
      "primary": {
        "block": "punctuation/all",
        "metric": "macro_f1",
        "value": 0.770863
      },
      "score": {
        "path": "experiments/runs/neural/t5_seq2seq_A/lr0.0001-epoch2/validation.score.json",
        "sha256": "1728a8b739017395106e6a815260f1fb478e50d0a4eb5705813b03d0ff597b6a"
      },
      "training_diagnostics": {
        "cutoff_words": 0,
        "words": 0
      },
      "training_seconds": 8340.861709299992
    },
    {
      "artifact": "lr0.0001-epoch3/model",
      "artifact_metadata_sha256": "7caf8f79b2f663e7940ecff9b83d18d8ce1e2df0ae2c3a804ed18f8db25f008e",
      "diagnostics": {
        "unaligned_share": 0.0005155380585062972,
        "unaligned_words": 319,
        "words": 618771
      },
      "epoch": 3,
      "epoch_seconds": 4075.2503471999953,
      "learning_rate": 0.0001,
      "mean_batch_loss": 0.06558749806375243,
      "pretrained": "google/flan-t5-base",
      "pretrained_revision": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
      "primary": {
        "block": "punctuation/all",
        "metric": "macro_f1",
        "value": 0.771668
      },
      "score": {
        "path": "experiments/runs/neural/t5_seq2seq_A/lr0.0001-epoch3/validation.score.json",
        "sha256": "1889cc2b7765fc4cd9c7c0e87e60ee486a31904b04cd7f6364f6c006ac7fb8f8"
      },
      "training_diagnostics": {
        "cutoff_words": 0,
        "words": 0
      },
      "training_seconds": 12416.112056499987
    }
  ],
  "command": [
    "scripts/train_seq2seq.py",
    "train",
    "--mode",
    "A",
    "--input",
    "data/processed/sponspeech-m1-v2/mode_A/train.jsonl",
    "--ids",
    "data/splits/sponspeech-m1-v2/default_train/mode_A.ids",
    "--validation-input",
    "data/processed/sponspeech-m1-v2/mode_A/validation.jsonl",
    "--validation-ids",
    "data/splits/sponspeech-m1-v2/primary_eval/validation_mode_A.ids",
    "--output",
    "experiments\\runs\\neural\\t5_seq2seq_A",
    "--manifest-out",
    "experiments\\runs\\neural\\t5_seq2seq_A\\training.manifest.json",
    "--selection-out",
    "experiments\\runs\\neural\\t5_seq2seq_A\\selection.json",
    "--run-id",
    "t5_seq2seq_A",
    "--epochs",
    "3",
    "--batch-size",
    "4",
    "--gradient-accumulation",
    "4",
    "--revision",
    "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
  ],
  "criterion": "official evaluator primary macro F1; ties: fewer epochs, smaller LR",
  "environment": {
    "code_commit": "6654398b00b2dc8361fb7e2aae869aad63d8fad6",
    "cuda": "12.8",
    "device": "cuda",
    "dirty": true,
    "gpu": "NVIDIA GeForce RTX 5070",
    "packages": {
      "safetensors": "0.8.0",
      "sentencepiece": "0.2.1",
      "tokenizers": "0.22.2",
      "torch": "2.9.1+cu128",
      "transformers": "4.57.3"
    },
    "platform": "Windows-11-10.0.26300-SP0",
    "python": "3.13.14",
    "source_sha256": {
      "configs/label_sets.yaml": "eeba7991ce7e897fd610ed1928d6939242b2e76806e5839b63903f6e92fa1142",
      "pyproject.toml": "4c90a49ff4155d1d3314d1a2870b45f31a7dcb02f3b9eee0e4c59cc08ea88213",
      "scripts/train_neural_tagger.py": "8f83811a6ee33645ffbeae43f24dcf82cbb79903306ed7789ca17c75b79b8782",
      "scripts/train_seq2seq.py": "5fcec9068d81a6656207844752a44a3859e2f484cc0a83e0947960d1ee5b339e",
      "src/punct_restore/models/neural_tagger.py": "9bad18347c2eb1fb2d5fb2255bd7cb098b70a677e4adc0ea9202efd03edb86f9",
      "src/punct_restore/models/neural_utils.py": "a702b087229e942d7cee5d3f15ff062a2903602f9e87736c3843546c26c72b13",
      "src/punct_restore/models/neural_workflow.py": "520523345d64f2ab9d121b17751e65962e1dc8257427cf4ff49c941f237714fb",
      "src/punct_restore/models/seq2seq.py": "c8713241c7e0b7a1d1cfa0e1478c6633a9b4917832d0ca578a4d948ac81872ff"
    }
  },
  "frozen": true,
  "frozen_at": "2026-10-31T01:37:23.760379+00:00",
  "inputs": {
    "masks": null,
    "training": {
      "path": "data/processed/sponspeech-m1-v2/mode_A/train.jsonl",
      "sha256": "9412866dd8ae1aaf86e18611f47b1ee8da888792808da0a525ecd61c4bd7c56e"
    },
    "training_ids": {
      "path": "data/splits/sponspeech-m1-v2/default_train/mode_A.ids",
      "sha256": "593196fea62d7aa6eb9eb3bea6649f06f5cc43dab91f048943efa0d9d590492a"
    },
    "validation": {
      "path": "data/processed/sponspeech-m1-v2/mode_A/validation.jsonl",
      "sha256": "7ececedac4e58c307df0daaf6190b61b2ea715344b72a2910d9541e07b1ecf27"
    },
    "validation_ids": {
      "path": "data/splits/sponspeech-m1-v2/primary_eval/validation_mode_A.ids",
      "sha256": "15e43a80088a2189f0f0d34c6e1fc5dd867a2ee42637fc5e80d106c7eed487db"
    }
  },
  "label_mode": "A",
  "model_name": "t5_seq2seq",
  "run_id": "t5_seq2seq_A",
  "schema_version": 1,
  "search_training_seconds": 12416.112056499987,
  "selected": {
    "artifact": "lr0.0001-epoch3/model",
    "artifact_metadata_sha256": "7caf8f79b2f663e7940ecff9b83d18d8ce1e2df0ae2c3a804ed18f8db25f008e",
    "diagnostics": {
      "unaligned_share": 0.0005155380585062972,
      "unaligned_words": 319,
      "words": 618771
    },
    "epoch": 3,
    "epoch_seconds": 4075.2503471999953,
    "learning_rate": 0.0001,
    "mean_batch_loss": 0.06558749806375243,
    "pretrained": "google/flan-t5-base",
    "pretrained_revision": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
    "primary": {
      "block": "punctuation/all",
      "metric": "macro_f1",
      "value": 0.771668
    },
    "score": {
      "path": "experiments/runs/neural/t5_seq2seq_A/lr0.0001-epoch3/validation.score.json",
      "sha256": "1889cc2b7765fc4cd9c7c0e87e60ee486a31904b04cd7f6364f6c006ac7fb8f8"
    },
    "training_diagnostics": {
      "cutoff_words": 0,
      "words": 0
    },
    "training_seconds": 12416.112056499987
  },
  "settings": {
    "batch_size": 4,
    "dataset_release": "sponspeech-m1-v2",
    "epochs": 3,
    "gradient_accumulation": 4,
    "gradient_clip_norm": 1.0,
    "num_beams": 4,
    "optimizer": "AdamW",
    "policy_version": "m1-v2",
    "precision": "bf16",
    "pretrained": "google/flan-t5-base",
    "schedule": "constant",
    "seed": 2026,
    "weight_decay": 0.01,
    "word_alignment": "whitespace + SequenceMatcher matching blocks; D5 projection"
  },
  "training_manifest": "training.manifest.json",
  "training_sequences": 142722,
  "training_words": 4137833,
  "validation_runtime": {
    "batch_size": 4,
    "median_ms_per_utterance": 154.69562298902295,
    "ms_per_utterance_passes": [
      154.69765630738462,
      154.69562298902295,
      154.692661042914
    ],
    "scope": "tokenization + inference/generation + alignment; excludes model/data loading",
    "timed_passes": 3,
    "warmup_passes": 1
  }
}
```

## Frozen run bert_tagger_B

```json
{
  "candidates": [
    {
      "artifact": "lr2e-05-epoch1/model",
      "artifact_metadata_sha256": "afd9a4f44ff988156d7a2e6fcbe6bb3e14cb07c8f5a0efe330f5ef3f03a7dbbe",
      "diagnostics": {
        "cutoff_share": 0.0,
        "cutoff_words": 0,
        "words": 618771
      },
      "epoch": 1,
      "epoch_seconds": 1294.5479391000117,
      "learning_rate": 2e-05,
      "mean_batch_loss": 0.14075372031354544,
      "pretrained": "bert-base-uncased",
      "pretrained_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
      "primary": {
        "block": "punctuation/all",
        "metric": "macro_f1",
        "value": 0.379986
      },
      "score": {
        "path": "experiments/runs/neural/bert_tagger_B/lr2e-05-epoch1/validation.score.json",
        "sha256": "1e39c5db7bddefff37b674d93b57ec58f125a41177fc53e6205a67851f938455"
      },
      "training_diagnostics": {
        "cutoff_words": 0,
        "words": 4137833
      },
      "training_seconds": 1294.5479391000117
    },
    {
      "artifact": "lr2e-05-epoch2/model",
      "artifact_metadata_sha256": "519351f3dfd59fa4d8c57cbd8ef52c12c6ad2359db1dfc0794cf56bde4b85ac0",
      "diagnostics": {
        "cutoff_share": 0.0,
        "cutoff_words": 0,
        "words": 618771
      },
      "epoch": 2,
      "epoch_seconds": 1205.4362648999959,
      "learning_rate": 2e-05,
      "mean_batch_loss": 0.1087635044571963,
      "pretrained": "bert-base-uncased",
      "pretrained_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
      "primary": {
        "block": "punctuation/all",
        "metric": "macro_f1",
        "value": 0.412494
      },
      "score": {
        "path": "experiments/runs/neural/bert_tagger_B/lr2e-05-epoch2/validation.score.json",
        "sha256": "fbc9aced3b696225ff66aec04d32fc92bdd598735d0e1eb24630215ad70539c7"
      },
      "training_diagnostics": {
        "cutoff_words": 0,
        "words": 4137833
      },
      "training_seconds": 2499.9842040000076
    },
    {
      "artifact": "lr2e-05-epoch3/model",
      "artifact_metadata_sha256": "f30e29c130faf12aa4e358971be1097c6afaaf8e28836d1ba56ad4fc438a0a89",
      "diagnostics": {
        "cutoff_share": 0.0,
        "cutoff_words": 0,
        "words": 618771
      },
      "epoch": 3,
      "epoch_seconds": 1200.3205965999805,
      "learning_rate": 2e-05,
      "mean_batch_loss": 0.09164417602592777,
      "pretrained": "bert-base-uncased",
      "pretrained_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
      "primary": {
        "block": "punctuation/all",
        "metric": "macro_f1",
        "value": 0.422351
      },
      "score": {
        "path": "experiments/runs/neural/bert_tagger_B/lr2e-05-epoch3/validation.score.json",
        "sha256": "7d9f8cd5fc3cdc5c58163377e91f793f93712ca4282eb48de1d933cd6ead2554"
      },
      "training_diagnostics": {
        "cutoff_words": 0,
        "words": 4137833
      },
      "training_seconds": 3700.304800599988
    },
    {
      "artifact": "lr5e-05-epoch1/model",
      "artifact_metadata_sha256": "3bd2e301c427eb2a09095744ca609b97102c99c529d2ea6ebba524c8d7bf9342",
      "diagnostics": {
        "cutoff_share": 0.0,
        "cutoff_words": 0,
        "words": 618771
      },
      "epoch": 1,
      "epoch_seconds": 1202.9387815000082,
      "learning_rate": 5e-05,
      "mean_batch_loss": 0.13929101667541513,
      "pretrained": "bert-base-uncased",
      "pretrained_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
      "primary": {
        "block": "punctuation/all",
        "metric": "macro_f1",
        "value": 0.394124
      },
      "score": {
        "path": "experiments/runs/neural/bert_tagger_B/lr5e-05-epoch1/validation.score.json",
        "sha256": "dcf687b33cf3ebc9761c9986a0fcbdc150b69e040fabedc19652afc7fc625193"
      },
      "training_diagnostics": {
        "cutoff_words": 0,
        "words": 4137833
      },
      "training_seconds": 1202.9387815000082
    },
    {
      "artifact": "lr5e-05-epoch2/model",
      "artifact_metadata_sha256": "e1ffbac58b66072b3bf24a503be1e78639ea19495f80c5c46d52aa9653c9ace2",
      "diagnostics": {
        "cutoff_share": 0.0,
        "cutoff_words": 0,
        "words": 618771
      },
      "epoch": 2,
      "epoch_seconds": 1203.579577299999,
      "learning_rate": 5e-05,
      "mean_batch_loss": 0.10960576164823736,
      "pretrained": "bert-base-uncased",
      "pretrained_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
      "primary": {
        "block": "punctuation/all",
        "metric": "macro_f1",
        "value": 0.399796
      },
      "score": {
        "path": "experiments/runs/neural/bert_tagger_B/lr5e-05-epoch2/validation.score.json",
        "sha256": "6007d8b9f6e72da1c35714af07736f0065902f29dd6dd507629ec83f4128e097"
      },
      "training_diagnostics": {
        "cutoff_words": 0,
        "words": 4137833
      },
      "training_seconds": 2406.5183588000073
    },
    {
      "artifact": "lr5e-05-epoch3/model",
      "artifact_metadata_sha256": "e483259bb562de69beaee902461a72e7b4ec2ec4b903dcc4b17d351e56a07336",
      "diagnostics": {
        "cutoff_share": 0.0,
        "cutoff_words": 0,
        "words": 618771
      },
      "epoch": 3,
      "epoch_seconds": 1204.1861403000075,
      "learning_rate": 5e-05,
      "mean_batch_loss": 0.09213879033066323,
      "pretrained": "bert-base-uncased",
      "pretrained_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
      "primary": {
        "block": "punctuation/all",
        "metric": "macro_f1",
        "value": 0.423853
      },
      "score": {
        "path": "experiments/runs/neural/bert_tagger_B/lr5e-05-epoch3/validation.score.json",
        "sha256": "d7bc3535c952e824b8306797ea07b5019d7d67f8acdd843ec29a616a05376d82"
      },
      "training_diagnostics": {
        "cutoff_words": 0,
        "words": 4137833
      },
      "training_seconds": 3610.704499100015
    }
  ],
  "command": [
    "scripts/train_neural_tagger.py",
    "train",
    "--mode",
    "B",
    "--input",
    "data/processed/sponspeech-m1-v2/mode_B/train.jsonl",
    "--ids",
    "data/splits/sponspeech-m1-v2/default_train/mode_B.ids",
    "--validation-input",
    "data/processed/sponspeech-m1-v2/mode_B/validation.jsonl",
    "--validation-ids",
    "data/splits/sponspeech-m1-v2/primary_eval/validation_mode_B.ids",
    "--output",
    "experiments\\runs\\neural\\bert_tagger_B",
    "--manifest-out",
    "experiments\\runs\\neural\\bert_tagger_B\\training.manifest.json",
    "--selection-out",
    "experiments\\runs\\neural\\bert_tagger_B\\selection.json",
    "--run-id",
    "bert_tagger_B",
    "--epochs",
    "3",
    "--batch-size",
    "4",
    "--gradient-accumulation",
    "4",
    "--revision",
    "86b5e0934494bd15c9632b12f734a8a67f723594"
  ],
  "criterion": "official evaluator primary macro F1; ties: fewer epochs, smaller LR",
  "environment": {
    "code_commit": "6654398b00b2dc8361fb7e2aae869aad63d8fad6",
    "cuda": "12.8",
    "device": "cuda",
    "dirty": true,
    "gpu": "NVIDIA GeForce RTX 5070",
    "packages": {
      "safetensors": "0.8.0",
      "sentencepiece": "0.2.1",
      "tokenizers": "0.22.2",
      "torch": "2.9.1+cu128",
      "transformers": "4.57.3"
    },
    "platform": "Windows-11-10.0.26300-SP0",
    "python": "3.13.14",
    "source_sha256": {
      "configs/label_sets.yaml": "eeba7991ce7e897fd610ed1928d6939242b2e76806e5839b63903f6e92fa1142",
      "pyproject.toml": "4c90a49ff4155d1d3314d1a2870b45f31a7dcb02f3b9eee0e4c59cc08ea88213",
      "scripts/train_neural_tagger.py": "8f83811a6ee33645ffbeae43f24dcf82cbb79903306ed7789ca17c75b79b8782",
      "scripts/train_seq2seq.py": "5fcec9068d81a6656207844752a44a3859e2f484cc0a83e0947960d1ee5b339e",
      "src/punct_restore/models/neural_tagger.py": "9bad18347c2eb1fb2d5fb2255bd7cb098b70a677e4adc0ea9202efd03edb86f9",
      "src/punct_restore/models/neural_utils.py": "a702b087229e942d7cee5d3f15ff062a2903602f9e87736c3843546c26c72b13",
      "src/punct_restore/models/neural_workflow.py": "520523345d64f2ab9d121b17751e65962e1dc8257427cf4ff49c941f237714fb",
      "src/punct_restore/models/seq2seq.py": "c8713241c7e0b7a1d1cfa0e1478c6633a9b4917832d0ca578a4d948ac81872ff"
    }
  },
  "frozen": true,
  "frozen_at": "2026-10-31T08:01:56.797529+00:00",
  "inputs": {
    "masks": null,
    "training": {
      "path": "data/processed/sponspeech-m1-v2/mode_B/train.jsonl",
      "sha256": "b10a75e5eb126c97fbacdf69b0edf93ae0cbec3704620ab5c5f03d4a3882779d"
    },
    "training_ids": {
      "path": "data/splits/sponspeech-m1-v2/default_train/mode_B.ids",
      "sha256": "593196fea62d7aa6eb9eb3bea6649f06f5cc43dab91f048943efa0d9d590492a"
    },
    "validation": {
      "path": "data/processed/sponspeech-m1-v2/mode_B/validation.jsonl",
      "sha256": "efbc877efa64abb85579889c1c08ef5faf89b0b22a0eb3e629ca76c73f690fb0"
    },
    "validation_ids": {
      "path": "data/splits/sponspeech-m1-v2/primary_eval/validation_mode_B.ids",
      "sha256": "15e43a80088a2189f0f0d34c6e1fc5dd867a2ee42637fc5e80d106c7eed487db"
    }
  },
  "label_mode": "B",
  "model_name": "bert_tagger",
  "run_id": "bert_tagger_B",
  "schema_version": 1,
  "search_training_seconds": 7311.009299700003,
  "selected": {
    "artifact": "lr5e-05-epoch3/model",
    "artifact_metadata_sha256": "e483259bb562de69beaee902461a72e7b4ec2ec4b903dcc4b17d351e56a07336",
    "diagnostics": {
      "cutoff_share": 0.0,
      "cutoff_words": 0,
      "words": 618771
    },
    "epoch": 3,
    "epoch_seconds": 1204.1861403000075,
    "learning_rate": 5e-05,
    "mean_batch_loss": 0.09213879033066323,
    "pretrained": "bert-base-uncased",
    "pretrained_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
    "primary": {
      "block": "punctuation/all",
      "metric": "macro_f1",
      "value": 0.423853
    },
    "score": {
      "path": "experiments/runs/neural/bert_tagger_B/lr5e-05-epoch3/validation.score.json",
      "sha256": "d7bc3535c952e824b8306797ea07b5019d7d67f8acdd843ec29a616a05376d82"
    },
    "training_diagnostics": {
      "cutoff_words": 0,
      "words": 4137833
    },
    "training_seconds": 3610.704499100015
  },
  "settings": {
    "batch_size": 4,
    "dataset_release": "sponspeech-m1-v2",
    "epochs": 3,
    "gradient_accumulation": 4,
    "gradient_clip_norm": 1.0,
    "num_beams": 4,
    "optimizer": "AdamW",
    "policy_version": "m1-v2",
    "precision": "fp16",
    "pretrained": "bert-base-uncased",
    "schedule": "constant",
    "seed": 2026,
    "weight_decay": 0.01,
    "word_alignment": "first complete subword; truncate at 512; default cut words"
  },
  "training_manifest": "training.manifest.json",
  "training_sequences": 142722,
  "training_words": 4137833,
  "validation_runtime": {
    "batch_size": 4,
    "median_ms_per_utterance": 1.7876131586826216,
    "ms_per_utterance_passes": [
      1.7876131586826216,
      1.7704260179640756,
      1.8001921407184438
    ],
    "scope": "tokenization + inference/generation + alignment; excludes model/data loading",
    "timed_passes": 3,
    "warmup_passes": 1
  }
}
```

## Frozen run t5_seq2seq_B

```json
{
  "candidates": [
    {
      "artifact": "lr0.0001-epoch1/model",
      "artifact_metadata_sha256": "df5694d81d7639b6a1929409592062b1de64fd0523637153db9e4fbb4f67e76f",
      "diagnostics": {
        "unaligned_share": 0.0005527085141352778,
        "unaligned_words": 342,
        "words": 618771
      },
      "epoch": 1,
      "epoch_seconds": 4068.095457599993,
      "learning_rate": 0.0001,
      "mean_batch_loss": 0.09993815325556815,
      "pretrained": "google/flan-t5-base",
      "pretrained_revision": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
      "primary": {
        "block": "punctuation/all",
        "metric": "macro_f1",
        "value": 0.360556
      },
      "score": {
        "path": "experiments/runs/neural/t5_seq2seq_B/lr0.0001-epoch1/validation.score.json",
        "sha256": "e8d0e846ab813d6b34afbb6040b1dc5d7b6ba866a46942a522599c531903d198"
      },
      "training_diagnostics": {
        "cutoff_words": 0,
        "words": 0
      },
      "training_seconds": 4068.095457599993
    },
    {
      "artifact": "lr0.0001-epoch2/model",
      "artifact_metadata_sha256": "7bdb0f4270e43c24428db2bdba84a9ec67a71542e663dd390720a2e61de7d678",
      "diagnostics": {
        "unaligned_share": 0.0004702870690449294,
        "unaligned_words": 291,
        "words": 618771
      },
      "epoch": 2,
      "epoch_seconds": 4068.6167638000043,
      "learning_rate": 0.0001,
      "mean_batch_loss": 0.08130957780282472,
      "pretrained": "google/flan-t5-base",
      "pretrained_revision": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
      "primary": {
        "block": "punctuation/all",
        "metric": "macro_f1",
        "value": 0.388096
      },
      "score": {
        "path": "experiments/runs/neural/t5_seq2seq_B/lr0.0001-epoch2/validation.score.json",
        "sha256": "31d244dd3d4adcec3a9e7ed96fd7b955158afe8cf9e78472f47fc931bf0b4df2"
      },
      "training_diagnostics": {
        "cutoff_words": 0,
        "words": 0
      },
      "training_seconds": 8136.712221399997
    },
    {
      "artifact": "lr0.0001-epoch3/model",
      "artifact_metadata_sha256": "843d12308e2625d8a81eefa3128d6fd538420af76ac2bc1f3fec8ac8e9d45cd5",
      "diagnostics": {
        "unaligned_share": 0.0004315005066494713,
        "unaligned_words": 267,
        "words": 618771
      },
      "epoch": 3,
      "epoch_seconds": 4107.094311199995,
      "learning_rate": 0.0001,
      "mean_batch_loss": 0.07198497862403372,
      "pretrained": "google/flan-t5-base",
      "pretrained_revision": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
      "primary": {
        "block": "punctuation/all",
        "metric": "macro_f1",
        "value": 0.380203
      },
      "score": {
        "path": "experiments/runs/neural/t5_seq2seq_B/lr0.0001-epoch3/validation.score.json",
        "sha256": "5057fda63bd00460d1c5041a56d36c3a137e1477bd17427960f519ff8b269185"
      },
      "training_diagnostics": {
        "cutoff_words": 0,
        "words": 0
      },
      "training_seconds": 12243.806532599992
    }
  ],
  "command": [
    "scripts/train_seq2seq.py",
    "train",
    "--mode",
    "B",
    "--input",
    "data/processed/sponspeech-m1-v2/mode_B/train.jsonl",
    "--ids",
    "data/splits/sponspeech-m1-v2/default_train/mode_B.ids",
    "--validation-input",
    "data/processed/sponspeech-m1-v2/mode_B/validation.jsonl",
    "--validation-ids",
    "data/splits/sponspeech-m1-v2/primary_eval/validation_mode_B.ids",
    "--output",
    "experiments\\runs\\neural\\t5_seq2seq_B",
    "--manifest-out",
    "experiments\\runs\\neural\\t5_seq2seq_B\\training.manifest.json",
    "--selection-out",
    "experiments\\runs\\neural\\t5_seq2seq_B\\selection.json",
    "--run-id",
    "t5_seq2seq_B",
    "--epochs",
    "3",
    "--batch-size",
    "4",
    "--gradient-accumulation",
    "4",
    "--revision",
    "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
  ],
  "criterion": "official evaluator primary macro F1; ties: fewer epochs, smaller LR",
  "environment": {
    "code_commit": "6654398b00b2dc8361fb7e2aae869aad63d8fad6",
    "cuda": "12.8",
    "device": "cuda",
    "dirty": true,
    "gpu": "NVIDIA GeForce RTX 5070",
    "packages": {
      "safetensors": "0.8.0",
      "sentencepiece": "0.2.1",
      "tokenizers": "0.22.2",
      "torch": "2.9.1+cu128",
      "transformers": "4.57.3"
    },
    "platform": "Windows-11-10.0.26300-SP0",
    "python": "3.13.14",
    "source_sha256": {
      "configs/label_sets.yaml": "eeba7991ce7e897fd610ed1928d6939242b2e76806e5839b63903f6e92fa1142",
      "pyproject.toml": "4c90a49ff4155d1d3314d1a2870b45f31a7dcb02f3b9eee0e4c59cc08ea88213",
      "scripts/train_neural_tagger.py": "8f83811a6ee33645ffbeae43f24dcf82cbb79903306ed7789ca17c75b79b8782",
      "scripts/train_seq2seq.py": "5fcec9068d81a6656207844752a44a3859e2f484cc0a83e0947960d1ee5b339e",
      "src/punct_restore/models/neural_tagger.py": "9bad18347c2eb1fb2d5fb2255bd7cb098b70a677e4adc0ea9202efd03edb86f9",
      "src/punct_restore/models/neural_utils.py": "a702b087229e942d7cee5d3f15ff062a2903602f9e87736c3843546c26c72b13",
      "src/punct_restore/models/neural_workflow.py": "520523345d64f2ab9d121b17751e65962e1dc8257427cf4ff49c941f237714fb",
      "src/punct_restore/models/seq2seq.py": "c8713241c7e0b7a1d1cfa0e1478c6633a9b4917832d0ca578a4d948ac81872ff"
    }
  },
  "frozen": true,
  "frozen_at": "2026-10-31T14:11:29.662297+00:00",
  "inputs": {
    "masks": null,
    "training": {
      "path": "data/processed/sponspeech-m1-v2/mode_B/train.jsonl",
      "sha256": "b10a75e5eb126c97fbacdf69b0edf93ae0cbec3704620ab5c5f03d4a3882779d"
    },
    "training_ids": {
      "path": "data/splits/sponspeech-m1-v2/default_train/mode_B.ids",
      "sha256": "593196fea62d7aa6eb9eb3bea6649f06f5cc43dab91f048943efa0d9d590492a"
    },
    "validation": {
      "path": "data/processed/sponspeech-m1-v2/mode_B/validation.jsonl",
      "sha256": "efbc877efa64abb85579889c1c08ef5faf89b0b22a0eb3e629ca76c73f690fb0"
    },
    "validation_ids": {
      "path": "data/splits/sponspeech-m1-v2/primary_eval/validation_mode_B.ids",
      "sha256": "15e43a80088a2189f0f0d34c6e1fc5dd867a2ee42637fc5e80d106c7eed487db"
    }
  },
  "label_mode": "B",
  "model_name": "t5_seq2seq",
  "run_id": "t5_seq2seq_B",
  "schema_version": 1,
  "search_training_seconds": 12243.806532599992,
  "selected": {
    "artifact": "lr0.0001-epoch2/model",
    "artifact_metadata_sha256": "7bdb0f4270e43c24428db2bdba84a9ec67a71542e663dd390720a2e61de7d678",
    "diagnostics": {
      "unaligned_share": 0.0004702870690449294,
      "unaligned_words": 291,
      "words": 618771
    },
    "epoch": 2,
    "epoch_seconds": 4068.6167638000043,
    "learning_rate": 0.0001,
    "mean_batch_loss": 0.08130957780282472,
    "pretrained": "google/flan-t5-base",
    "pretrained_revision": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
    "primary": {
      "block": "punctuation/all",
      "metric": "macro_f1",
      "value": 0.388096
    },
    "score": {
      "path": "experiments/runs/neural/t5_seq2seq_B/lr0.0001-epoch2/validation.score.json",
      "sha256": "31d244dd3d4adcec3a9e7ed96fd7b955158afe8cf9e78472f47fc931bf0b4df2"
    },
    "training_diagnostics": {
      "cutoff_words": 0,
      "words": 0
    },
    "training_seconds": 8136.712221399997
  },
  "settings": {
    "batch_size": 4,
    "dataset_release": "sponspeech-m1-v2",
    "epochs": 3,
    "gradient_accumulation": 4,
    "gradient_clip_norm": 1.0,
    "num_beams": 4,
    "optimizer": "AdamW",
    "policy_version": "m1-v2",
    "precision": "bf16",
    "pretrained": "google/flan-t5-base",
    "schedule": "constant",
    "seed": 2026,
    "weight_decay": 0.01,
    "word_alignment": "whitespace + SequenceMatcher matching blocks; D5 projection"
  },
  "training_manifest": "training.manifest.json",
  "training_sequences": 142722,
  "training_words": 4137833,
  "validation_runtime": {
    "batch_size": 4,
    "median_ms_per_utterance": 191.93460800399293,
    "ms_per_utterance_passes": [
      194.48831525449046,
      191.93460800399293,
      179.17326387724532
    ],
    "scope": "tokenization + inference/generation + alignment; excludes model/data loading",
    "timed_passes": 3,
    "warmup_passes": 1
  }
}
```

## All-four validation seal

All four selections, complete candidate grids, saved artifact hashes, official
validation scores with bootstrap 1000, and full timing/notes records passed the
local pre-inference audit before this commit. No held-out receipt existed at
this seal. The subsequent inference driver must use these exact frozen records.
