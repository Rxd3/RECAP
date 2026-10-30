# Neural comparisons

Status on 2026-10-30: the study has moved to the requested local Windows PC.
`nvidia-smi` confirms an NVIDIA GeForce RTX 5070, 12,227 MiB VRAM, driver 617.42.
The isolated `.venv` uses 64-bit Python 3.13.14 and torch 2.9.1+cu128; a real
512-by-512 float32 CUDA matrix multiplication matched the CPU calculation
(maximum absolute error 0.0). The GPU has compute capability 12.0, the wheel
includes sm_120, and bf16 is supported. The machine has 16 GB physical RAM;
bootstrap found about 193 GB free disk. Preflight is complete. Finished real
selections are recorded in the frozen-run entries below; held-out prediction and
the final handoff remain pending.
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
The handoff was unpacked under ignored artifact storage; all seven bundle file
checksums passed, and the processed/split checksum lists match the already
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
runtimes. The complete handoff of 18 prediction JSONL files plus 18 manifests,
held-out bootstrap scores, the final ZIP, its Drive upload and the link to Abd
Raouf remain pending.
The packaging command requires all six frozen selections and all 18 prediction
pairs, verifies them with the unchanged official scorer and bootstrap 1000, then
creates a ZIP containing only predictions, manifests and SHA256SUMS.

T5 can alter words; matching-block defaults deliberately penalize those edits.
Long BERT inputs lose tail labels under the permitted truncation policy. T5
generation limits may stop before the whole text is copied. These effects are
measured in the manifests instead of being repaired with gold labels.
