# Dataset card: RECAP release of SponSpeech (`sponspeech-m1-v2`)

Every number below names its source document. "Computed" marks a ratio we derived from those
numbers; nothing else was calculated for this card.

| Field | Value |
| --- | --- |
| Release | `sponspeech-m1-v2`, annotation policy `m1-v2`, schema v1 ([DATA_RELEASE_VALIDATION.md](DATA_RELEASE_VALIDATION.md)) |
| Language and modality | English text: transcripts of informal, podcast-style speech. No audio. |
| Upstream | SponSpeech, Liu & Beigi (2024), [arXiv:2409.11241](https://arxiv.org/abs/2409.11241) |
| Licence | CC BY-NC 4.0, non-commercial only ([DATA_LICENSE.md](DATA_LICENSE.md)) |
| Built by | Member 1 (Mutasim); the v2 corrections shipped in [PR #1](https://github.com/Rxd3/RECAP/pull/1) ([DATASET_PLAN.md](DATASET_PLAN.md)) |
| How to get it | rebuild with `scripts/prepare_data.py`, or download the team archive ([data/README.md](../data/README.md)) |

## Source: what upstream SponSpeech ships

- **Content.** Audio of informal speech from YouTube videos published under CC BY 4.0, with the
  transcripts that the creators uploaded (the paper excludes automatic captions and ASR output as
  ground truth). The dataset itself is CC BY-NC 4.0 (arXiv:2409.11241).
- **Size.** 206,632 single-line transcripts from 2,408 source videos ([DATASET_PLAN.md](DATASET_PLAN.md), Source).
- **Official splits.** train 147,209; dev 25,253; test 17,697; test-amb 16,473 utterances
  ([DATASET_PLAN.md](DATASET_PLAN.md), Source; the same counts as the paper's Table 2).
- **Upstream benchmark.** Three marks are scored: full stop, comma and question mark (arXiv:2409.11241).
- **Video overlap upstream.** The paper's per-split video counts (1,736 + 277 + 205 + 397) sum to
  2,615, against 2,408 distinct videos (computed), so some videos appear in more than one split.
- **What RECAP takes.** Only the transcripts: about 97 MB of pinned byte ranges from the 63.5 GB
  archive, checked file by file against a SHA-256 manifest ([DATASET_PLAN.md](DATASET_PLAN.md)).

## What RECAP adds

| Addition | What it is | Source |
| --- | --- | --- |
| Token alignment | One sequence per utterance; tokens, order and IDs are identical across all modes and models; casing is read before lowercasing. | [DECISIONS.md](DECISIONS.md) D2 |
| 8-class punctuation (mode B) | `? ! . ; : ,` map to their own classes; `--` and spaced hyphens map to DASH. | D3 |
| 3-way casing with masks | LOWER, FIRST_UPPER, ALL_UPPER by fixed precedence; approximate mixed casing (`iPhone`) is masked per position from casing and joint scores. | D4 |
| 12 joint labels (mode C) | Mode A punctuation × casing, written `PERIOD\|FIRST_UPPER`. | [label_sets.yaml](../configs/label_sets.yaml) |
| B → A projection | EXCLAMATION, SEMICOLON → PERIOD; COLON, DASH → COMMA. Canonical A and C use it. | D5 |
| Native benchmark view | `benchmark_native/mode_A` keeps the original four-class gold, for comparison only. | D5 |
| Leak-free primary splits | Validation is filtered against train; test and test_amb against train ∪ validation, by shared video, exact text, exact token tuple and ≥ 50% shared word 6-grams. | D6, [DATASET_PLAN.md](DATASET_PLAN.md) |
| Unresolved-record flags | Ambiguous records stay in the full views with reasons and positions, and are excluded from default training and primary evaluation. | D3, [DATASET_PLAN.md](DATASET_PLAN.md) |
| Gold-noise audit | Known gold errors are listed, not relabelled. | [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) |
| Robustness and noise checks | Mode B rescored without the known noisy records, and the test inputs with simulated ASR word errors | [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §12, §13 |

The projection changes 13,721 labels relative to the native view, all O → COMMA at DASH
boundaries: train 9,762, validation 1,869, test 1,258, test_amb 832
([DATA_RELEASE_VALIDATION.md](DATA_RELEASE_VALIDATION.md), [DATASET_PLAN.md](DATASET_PLAN.md)).

## Leakage in the official splits

**27% of the official test comes from videos in train or dev:** 4,716 of the 17,695 released test
utterances share a source video with train or validation (computed: 26.7%). In validation the share
is 4,390 of 25,252 (17.4%), and in test_amb 3,096 of 16,472 (18.8%). Source: the Released records and
Leakage-controlled pool tables of [DATASET_PLAN.md](DATASET_PLAN.md); percentages computed.

| Pool | Released | Kept | Shared video | Raw exact | Normalized exact | Near duplicate | Low-comma/TTS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| validation | 25,252 | 20,532 | 4,390 | 29 | 15 | 115 | 171 |
| test | 17,695 | 12,668 | 4,716 | 24 | 13 | 83 | 191 |
| test_amb | 16,472 | 13,184 | 3,096 | 16 | 8 | 89 | 79 |

Independent full-record checks found zero shared videos, exact matches and qualifying six-gram
matches between each primary subset and its reference splits ([DATA_RELEASE_VALIDATION.md](DATA_RELEASE_VALIDATION.md)).

### Raw overlap between the official splits

Before any filtering, the official splits overlap. Each row compares every record of the first split with
the whole second split. Source: the `overlap` block of
`data/release_metadata/sponspeech-m1-v2/splits/leakage_controlled/report.json` (committed, byte-identical to
the release); percentages computed.

| Official pair | Records | Shared videos | Records in shared videos | Identical model inputs | Identical texts | Near duplicates (≥ 50% shared 6-grams) |
| --- | --- | --- | --- | --- | --- | --- |
| validation vs train | 25,252 | 54 | 4,390 | 1,700 | 1,683 | 4,210 |
| test vs train | 17,695 | 36 | 2,713 | 998 | 986 | 2,527 |
| test vs validation | 17,695 | 13 | 2,003 | 727 | 712 | 1,853 |
| test_amb vs train | 16,472 | 52 | 1,420 | 542 | 531 | 1,311 |
| test_amb vs validation | 16,472 | 26 | 1,676 | 544 | 539 | 1,572 |
| test_amb vs test | 16,472 | 26 | 1,664 | 594 | 587 | 1,521 |

So 998 official test utterances (5.6%) give a model exactly the same input as a training utterance, and
1,700 validation utterances (6.7%) do. After filtering, the same report shows zero overlap of every kind
between each leakage-controlled pool and train or validation. The one exception is by design: test_amb
and test still share 25 videos (1,637 records, 576 identical model inputs), which is why they are
reported separately and never pooled.

### What the leakage costs

The frozen HMM and improved CRF also predicted the full official mode A test once, with the excluded
utterances split by cause ([EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §8; macro F1 %):

| Scored set | Utterances | HMM | CRF improved |
| --- | --- | --- | --- |
| Leakage-controlled test (kept) | 12,668 | 34.74 | 49.50 |
| Excluded: video also in the training set | 2,706 | 33.54 | 67.09 |
| Excluded: other leakage (video shared with validation only, or duplicate text) | 2,130 | 37.50 | 53.97 |
| Excluded: quality filter (low comma density or TTS source) | 191 | 27.93 | 43.12 |

The improved CRF scores about 18 points higher on utterances from videos it was trained on, while the
HMM does not gain. In §8, "training set" means a video with an utterance in `default_train`, so its
2,706 differs from the 2,713 above, which counts videos shared with the full official train file.

## Splits and sizes

Released records per file ([DATASET_PLAN.md](DATASET_PLAN.md)); 48 source records are rejected with
reasons (`inverted_mark_folded` 45, `question_inside_word` 3):

| File | Released | Accepted | Flagged | Tokens |
| --- | --- | --- | --- | --- |
| train | 147,165 | 141,745 | 5,420 | 4,324,477 |
| validation | 25,252 | 24,350 | 902 | 793,071 |
| test | 17,695 | 16,918 | 777 | 574,988 |
| test_amb | 16,472 | 15,833 | 639 | 609,532 |

ID lists used by every model ([DATASET_PLAN.md](DATASET_PLAN.md)):

| Mode | Default train | Primary validation | Primary test | Primary test_amb |
| --- | --- | --- | --- | --- |
| A | 142,722 | 20,040 | 12,321 | 12,734 |
| B | 142,722 | 20,040 | 12,321 | 12,734 |
| C | 141,745 | 19,936 | 12,276 | 12,704 |

Report test and test_amb separately: they still share 25 videos after filtering.

## Label distribution

Mode B token counts in default training and primary evaluation ([DATASET_PLAN.md](DATASET_PLAN.md));
the share of non-O tokens is computed:

| Label | train | validation | test | test_amb | Share of non-O (train) |
| --- | --- | --- | --- | --- | --- |
| O | 3,613,963 | 538,697 | 354,902 | 420,170 | — |
| COMMA | 259,511 | 42,346 | 29,153 | 35,068 | 49.5% |
| PERIOD | 229,258 | 32,132 | 20,246 | 25,111 | 43.8% |
| QUESTION | 18,999 | 3,403 | 2,175 | 2,773 | 3.6% |
| EXCLAMATION | 1,922 | 156 | 204 | 155 | 0.4% |
| COLON | 3,385 | 425 | 166 | 273 | 0.6% |
| SEMICOLON | 1,553 | 104 | 120 | 139 | 0.3% |
| DASH | 9,242 | 1,508 | 784 | 615 | 1.8% |

O is 87.3% of training tokens (computed). The four extended classes together are 3.1% of non-O
training labels and 2.4% of non-O test labels (computed). Masked casing positions: train 9,075,
validation 1,212, test 900, test_amb 697 ([DATASET_PLAN.md](DATASET_PLAN.md)). Full casing and
per-mode distributions are in `data/release_metadata/sponspeech-m1-v2/processed/stats.json`. Ten example
records with their masks are in `data/sample/`.

## Known gold issues

From [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md), Known gold issues; nothing is relabelled:

- about 50 primary validation records keep a speaker label (for example `SPEAKER 1:`), giving 26
  false COLON labels, 6.1% of mode B validation COLON support;
- subtitle cue numbers appear as tokens with gold DASH;
- two auto-caption-style videos miss commas and sentence ends in their gold (135 validation and
  46 test records);
- about 1% of source transcripts contain errors;
- `QUESTION|ALL_UPPER` has only 18, 9 and 15 primary tokens (validation, test, test_amb).

## Limitations

From [DATASET_PLAN.md](DATASET_PLAN.md), Limitations:

- Every canonical A/C sequence ends in PERIOD or QUESTION, so the utterance boundary is a shortcut;
  results should also be read without the final token.
- Channel overlap remains: 149 of 522 channels span the original splits.
- Low-comma, TTS, speaker, casing and punctuation rules are documented heuristics.
- Canonical and native scores use different gold labels and are never compared directly.
- The transcripts are creator-written, not ASR output. There is no real-ASR condition; a simulated one is
  evaluated in [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §13.

## Licence and citation

CC BY-NC 4.0, inherited from SponSpeech; keep `SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt` with every
copy. Changes from the source, the upstream attribution text and the citation are in
[DATA_LICENSE.md](DATA_LICENSE.md).
