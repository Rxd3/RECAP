# Member 2 HMM results: SponSpeech v2

Study: `hmm-sponspeech-v2-20261029`. Data release `sponspeech-m1-v2`,
policy `m1-v2`, canonical views. These numbers come from completed experiments;
all 27 validation candidates and the final reports passed independent verification.

Training implementation commit: `e4c31834799bb261e78218c67dd85e148ad8d493` (clean).
Shared Member 5 scorer: `3044a58e777200bcac64e6e3364922404be359bf` (clean, pinned PR #2).
Environment: CPython 3.13.5, PyYAML 6.0.3,
Windows-11-10.0.26200-SP0. Runtime hardware: 13th Gen Intel(R) Core(TM) i7-13700H,
14 physical cores / 20 logical CPUs,
15.74 GiB OS-visible RAM; one model thread.

## Selection and training

Nine candidates per mode: alpha 0.01/0.1/1.0 and cutoff 1/2/5. Selection uses
six-decimal primary validation macro F1; ties prefer lower cutoff, then lower alpha.
All modes froze before test access. Models use only default-training IDs; validation
is not added to training. No test-driven retuning occurred. Best means best within
this declared grid, not a global optimum.

| Mode | Alpha | Cutoff | Train sequences | Train tokens | Vocabulary | Fit + input seconds | Validation primary F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.01 | 1 | 142722 | 4137833 | 67885 | 8.485 | 36.54 |
| B | 0.01 | 1 | 142722 | 4137833 | 67885 | 8.159 | 18.47 |
| C | 0.01 | 2 | 141745 | 4112054 | 41005 | 8.862 | 40.89 |

Fit timing includes JSONL reading, selection, validation, counting and probability construction; it excludes initial checksums and artifact writes. A/B choose non-O punctuation macro F1. C chooses masked joint macro F1 over all 12 classes. These primary values describe different tasks.

## Primary results

| Mode | Split | Primary block | Sequences | Tokens | HMM macro F1 % | Shortcut macro F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| A | validation | punctuation/all | 20040 | 618771 | 36.54 | 28.21 |
| A | test | punctuation/all | 12321 | 407750 | 35.12 | 27.48 |
| A | test_amb | punctuation/all | 12734 | 484304 | 31.50 | 23.90 |
| B | validation | punctuation/all | 20040 | 618771 | 18.47 | 12.14 |
| B | test | punctuation/all | 12321 | 407750 | 16.79 | 11.87 |
| B | test_amb | punctuation/all | 12734 | 484304 | 15.32 | 10.31 |
| C | validation | joint/unmasked | 19936 | 616940 | 40.89 | 18.53 |
| C | test | joint/unmasked | 12276 | 407011 | 37.41 | 18.09 |
| C | test_amb | joint/unmasked | 12704 | 483760 | 36.34 | 16.86 |

All averages keep the complete declared vocabulary, including zero-support classes, with zero_division=0. No gold-O tokens are removed before punctuation scoring. Test and test_amb are reported separately; their 25 shared videos mean they must not be pooled as independent sets.

## Diagnostics

| Mode | Split | Punctuation F1 % | Non-final punctuation F1 % | Punctuation accuracy % |
| --- | --- | --- | --- | --- |
| A | test | 35.12 | 9.13 | 90.29 |
| A | test_amb | 31.50 | 8.61 | 89.63 |
| B | test | 16.79 | 5.51 | 89.76 |
| B | test_amb | 15.32 | 5.44 | 89.12 |
| C | test | 34.38 | 17.42 | 90.27 |
| C | test_amb | 31.92 | 17.20 | 89.73 |

| C split | Joint/casing positions | Masked positions | Casing F1 % | Non-initial casing F1 % |
| --- | --- | --- | --- | --- |
| validation | 616049 | 891 | 84.31 | 80.67 |
| test | 406532 | 479 | 79.81 | 76.42 |
| test_amb | 483227 | 533 | 83.01 | 79.45 |

Punctuation remains scored at casing-mask positions. A/C punctuation comparisons require identical IDs; the mode-specific rows above are diagnostic results, not a paired cross-mode ranking.

## Test class support and F1

### Mode A: punctuation/all

| Class | Gold support | Predicted | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- |
| COMMA | 30103 | 8638 | 58.94 | 16.91 | 26.28 |
| O | 354902 | 386506 | 91.03 | 99.14 | 94.91 |
| PERIOD | 20570 | 12330 | 89.84 | 53.85 | 67.34 |
| QUESTION | 2175 | 276 | 52.17 | 6.62 | 11.75 |

### Mode B: punctuation/all

| Class | Gold support | Predicted | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- |
| COLON | 166 | 42 | 16.67 | 4.22 | 6.73 |
| COMMA | 29153 | 8267 | 60.11 | 17.04 | 26.56 |
| DASH | 784 | 2674 | 0.60 | 2.04 | 0.93 |
| EXCLAMATION | 204 | 348 | 2.01 | 3.43 | 2.54 |
| O | 354902 | 383758 | 91.15 | 98.56 | 94.71 |
| PERIOD | 20246 | 12320 | 89.85 | 54.68 | 67.99 |
| QUESTION | 2175 | 260 | 55.38 | 6.62 | 11.83 |
| SEMICOLON | 120 | 81 | 1.23 | 0.83 | 1.00 |

### Mode C: joint/unmasked

| Class | Gold support | Predicted | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- |
| COMMA\|ALL_UPPER | 171 | 41 | 26.83 | 6.43 | 10.38 |
| COMMA\|FIRST_UPPER | 4357 | 1908 | 46.80 | 20.50 | 28.51 |
| COMMA\|LOWER | 25450 | 5384 | 56.41 | 11.93 | 19.70 |
| O\|ALL_UPPER | 657 | 491 | 66.40 | 49.62 | 56.79 |
| O\|FIRST_UPPER | 35294 | 29117 | 83.57 | 68.95 | 75.56 |
| O\|LOWER | 317972 | 351172 | 88.82 | 98.09 | 93.23 |
| PERIOD\|ALL_UPPER | 125 | 41 | 53.66 | 17.60 | 26.51 |
| PERIOD\|FIRST_UPPER | 1637 | 1665 | 34.35 | 34.94 | 34.65 |
| PERIOD\|LOWER | 18708 | 16304 | 71.17 | 62.03 | 66.29 |
| QUESTION\|ALL_UPPER | 9 | 1 | 100.00 | 11.11 | 20.00 |
| QUESTION\|FIRST_UPPER | 148 | 14 | 28.57 | 2.70 | 4.94 |
| QUESTION\|LOWER | 2004 | 394 | 37.82 | 7.44 | 12.43 |

## Runtime

Identical observable workload: 19,936 mode-C primary-validation
sequences / 616,940 tokens in every mode. Warmup 200;
3 timed repetitions. In-memory predict includes model input checks,
decoding and prediction construction; it excludes file IO, loading and scoring.
Latency comes from a separate per-sequence decode pass. Python default GC is enabled.

| Mode | Median predict seconds | Tokens/second | Decode p50 ms | Decode p95 ms |
| --- | --- | --- | --- | --- |
| A | 1.762 | 350207 | 0.066 | 0.225 |
| B | 4.554 | 135481 | 0.172 | 0.570 |
| C | 8.084 | 76321 | 0.313 | 1.041 |

These measurements describe this host/run, not a general CPU or accuracy comparison. Full repetitions and timing scopes are in runtime.json and prediction manifests.

## Observed behavior and limits

- Mode A: Every supported primary class has nonzero test F1. Largest confusion counts: COMMA → O (24,933); PERIOD → O (8,778); O → COMMA (2,892).
- Mode B: Every supported primary class has nonzero test F1. Largest confusion counts: COMMA → O (23,559); PERIOD → O (8,356); O → COMMA (2,635).
- Mode C: Every supported primary class has nonzero test F1. Largest confusion counts: COMMA|LOWER → O|LOWER (20,072); O|FIRST_UPPER → O|LOWER (10,162); PERIOD|LOWER → O|LOWER (6,581).

The first-order HMM cannot model long linguistic dependencies directly. Lexical
emissions and one shared UNK observation limit unseen-word behavior. Three casing
classes cannot reconstruct arbitrary mixed casing; those approximation positions
are masked for joint/casing metrics. Strong utterance-final and initial conventions
are why the diagnostic position subsets are reported. Rare classes must remain
visible even when recall is zero. This study reports point estimates, without
claims of statistical significance or a globally optimal search.

## Verification and handoff

- 214 repository tests passed; the pinned Member 5 scorer passed 428 tests.
- A complete synthetic A/B/C workflow passed before the real study.
- The real study and audit ran in an isolated environment containing only pip and
  the pinned PyYAML runtime. All three independent full-data refits reproduce
  artifact bytes when frozen metadata is retained.
- Every full-split prediction from the independent refits matches the original;
  gold-field poisoning leaves predictions unchanged.
- All 27 candidate primary scores, nine final HMM reports and nine shortcut reports
  were independently recounted; final reports include every metric block.
- Training support matches the approved subsets. IDs/order/lengths, masks, code,
  data/config/artifact hashes and probability normalization pass.
- Frozen selections precede final test access. Source code/configuration did not
  change during the study. No dataset files or trained models are committed to Git (only a 10-utterance sample and the release metadata).

[HMM_RUNBOOK.md](HMM_RUNBOOK.md) contains the exact reconstruction commands.
The ignored run directory holds raw reports, confusion matrices, representative
errors, full commands, model/prediction manifests and verification evidence.
The local handoff is `artifacts/hmm/hmm-sponspeech-v2-20261029.zip`, with an outer
`.zip.sha256` and internal per-file `SHA256SUMS`. Its `MODEL_INDEX.json` identifies
the three selected artifacts; the packager verifies every archived byte. The raw
dataset is distributed separately. The ZIP is on the team Drive
(`Models/hmm-sponspeech-v2-20261029.zip`); individual acknowledgments are not recorded.

### Handoff archive and delivery checks

- **Archive:** `hmm-sponspeech-v2-20261029.zip`, 27,095,563 bytes, 365 checksummed files plus the
  checksum index. Outer SHA-256 `4cdc5725b6538d87a1c9aa237b47743683f547e11061b208be8ce1436880aa02`.
  It was packaged from commit `34849b9dca7eea8b6a60ff0ac9889a908d1cf38b` (clean) and is shared on the team
  Drive as `Models/hmm-sponspeech-v2-20261029.zip` ([REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md)).
- **Fresh extraction:** all 365 file hashes passed, each model loaded through `MODEL_INDEX.json`,
  and 20 gold-free test predictions per mode were reproduced.
- **Portable verification:** the documented verifier, run on the freshly extracted study with a separate
  output directory, refit all three models, reproduced every full-split prediction (178,257 sequence
  predictions in total), rechecked all 27 candidate primary scores and recounted every final metric block.
- **Frozen code:** training implementation `e4c31834799bb261e78218c67dd85e148ad8d493`. Later
  documentation and packaging commits keep all six frozen implementation files byte-identical, and the
  study's `verification/verification.json` reports PASS.
- **Use:** load a selected model with `HMMModel.load_model(path)` and pass gold-free records
  (`sample_id`, `tokens`) to `predict`; `scripts/run_demo.py --model-path <model>.json` does this for raw text.

Selected model SHA-256 values:

| Mode | Artifact SHA-256 |
| --- | --- |
| A | 6101f75785146d8be6c613b10c5236f45f5668b354f09297493a3eaf9310c97b |
| B | 2a2ce6fad2d1997881802a2df49dc90f10d7c5c0b366c6b67aa22e910d5e2f74 |
| C | 5b1ee9e9cc94a42eaa80159f5357d3be7301531e553fe6fca6cda694575f61ab |
