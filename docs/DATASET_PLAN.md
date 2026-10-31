# SponSpeech dataset and preparation

Owner: Member 1 (Mutasim). D1–D6 were approved by the team lead on 2026-10-28;
corrections were reviewed on 2026-10-29. Current release **sponspeech-m1-v2** uses
policy **m1-v2** and supersedes the v1/rev2 handoff. The corrected bundle is on the team
Drive as `dataset/sponspeech-m1-v2-handoff.zip` ([data/README.md](../data/README.md)).
See [release validation](DATA_RELEASE_VALIDATION.md), [decisions](DECISIONS.md)
and [source/build commands](../data/README.md).

## Source and provenance

- SponSpeech: Liu & Beigi (2024), *Spontaneous Informal Speech Dataset for Punctuation
  Restoration*, arXiv:2409.11241. Source archive:
  `https://storage.googleapis.com/sponspeech/sponspeech.tar.gz`, 63,547,617,096 bytes,
  GCS generation `1716975130398953`.
- CC BY-NC 4.0; non-commercial use with the archive's attribution list, shipped as
  `SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt`. Transcripts only; audio is not redistributed.
- 206,632 single-line ASCII transcripts, four `utt2spk` mappings, 2,408 source videos.
  Original split counts: train 147,209; dev 25,253; test 17,697; test-amb 16,473.
- No ASR transcripts are distributed. Records have `metadata.condition = "clean"`;
  low-comma filtering does not establish a separate ASR condition.
- The extractor reads about 97 MB of pinned HTTP ranges rather than the full 63.5 GB
  archive. Size, generation, deflate positions, range hashes and tar headers are checked.
  All 206,638 resulting files match `MANIFEST.sha256`. The full-archive fallback is
  tested with a synthetic archive; the full 63.5 GB fallback was not run in this review.
- The fixed `video_metadata_oembed.jsonl` snapshot was fetched on 2026-10-28: 2,408
  videos, 2,333 responses. It flags nine TTS videos (425 records) and supplies channel
  statistics. Do not re-fetch it when reproducing this release.

## Annotation policy m1-v2

- D2: contractions, numbers, URLs, emails, abbreviations and meaningful internal
  punctuation stay in tokens. Casing is extracted before lowercasing. One sequence
  per utterance; tokens, order and IDs are shared across all modes.
- Normalization removes documented timecodes, non-speech/name tags and speaker labels;
  `audit/normalization.jsonl` records every removal.
- D3: `? ! . ; : ,` map to their extended classes; `--` and standalone spaced hyphens
  map to DASH. Negative signs stay in numbers; a glued cutoff hyphen maps to O.
  Ellipsis alone maps to O internally and PERIOD at utterance end. A mixed ellipsis
  and another punctuation mark is an unresolved `mark_cluster`, including at the end.
- Unresolved examples remain in all full views with reasons and positions in
  `audit/unresolved.jsonl`; they are excluded from default training/primary evaluation.
- D5: canonical A projects B: EXCLAMATION/SEMICOLON → PERIOD, COLON/DASH → COMMA.
  C uses canonical A × casing (12 states). `benchmark_native/mode_A` preserves the
  SponSpeech four-class view for benchmark comparison only; it maps dashes to O.
- D4: no cased letters → LOWER; single uppercase letter → FIRST_UPPER; multiple letters
  all uppercase → ALL_UPPER; otherwise use the first cased letter. Approximate mixed
  casing is masked by position from primary casing/joint metrics, while punctuation
  remains scored. The 4/8/12 vocabularies and these mappings are unchanged from v1.

## Released records

All counts below come from the version 2 `stats.json`. Each of the four views
contains 206,584 utterances; 48 source records are rejected with reasons.

| File | Released | Accepted | Flagged | Rejected | Tokens | Length mean / median / max |
| --- | --- | --- | --- | --- | --- | --- |
| train | 147,165 | 141,745 | 5,420 | 44 | 4,324,477 | 29.39 / 23 / 384 |
| validation | 25,252 | 24,350 | 902 | 1 | 793,071 | 31.41 / 24.0 / 182 |
| test | 17,695 | 16,918 | 777 | 2 | 574,988 | 32.49 / 25 / 182 |
| test_amb | 16,472 | 15,833 | 639 | 1 | 609,532 | 37.0 / 29.0 / 307 |

Rejected reasons: `inverted_mark_folded` 45, `question_inside_word` 3.

Unresolved reasons in A/B (records may have multiple reasons): `abbrev_boundary_ambiguous` 295, `asterisk_text` 93, `kept_bracket_or_paren` 1,351, `low_comma_video` 1,108, `mark_cluster` 2,586, `possible_speaker_label` 483, `reconstruction_mismatch` 91, `single_initial_ambiguous` 376, `tts_source` 425, `uncertain_words` 30. C additionally excludes unusable whole-transcript casing.

## Default training and primary evaluation

Original membership is preserved (`dev` → `validation`; test-amb remains a separate
`test_amb.jsonl` with `split: "test"`). Validation is checked against full train;
test and test_amb against full train ∪ validation. Filtering checks source video,
original text, exact lowercased token tuple (including sequences shorter than six
tokens), and ≥50% shared word 6-grams. Low-comma/TTS sources and unresolved examples
are excluded. Excluded IDs and reasons are retained.

| Mode | Default train | Primary validation | Primary test | Primary test_amb |
| --- | --- | --- | --- | --- |
| A | 142,722 | 20,040 | 12,321 | 12,734 |
| B | 142,722 | 20,040 | 12,321 | 12,734 |
| C | 141,745 | 19,936 | 12,276 | 12,704 |

| Leakage-controlled pool | Kept | Videos | Shared video excluded | Raw exact excluded | Normalized exact excluded | Near-duplicate excluded | Low-comma/TTS excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| validation | 20,532 | 218 | 4,390 | 29 | 15 | 115 | 171 |
| test | 12,668 | 153 | 4,716 | 24 | 13 | 83 | 191 |
| test_amb | 13,184 | 316 | 3,096 | 16 | 8 | 89 | 79 |

These pools still include unresolved records; the primary mode-specific ID lists
apply the additional annotation exclusions. Independent full-record checks found
zero shared videos, raw exact matches, normalized exact matches and qualifying
six-gram matches between each primary subset and its required reference splits.

## Label distributions and masks

Canonical versus native differences (all are DASH boundaries changing O → COMMA):

| File | Labels changed |
| --- | --- |
| train | 9,762 |
| validation | 1,869 |
| test | 1,258 |
| test_amb | 832 |

Mode B token counts in default training / primary evaluation:

| Label | train | validation | test | test_amb |
| --- | --- | --- | --- | --- |
| O | 3,613,963 | 538,697 | 354,902 | 420,170 |
| COMMA | 259,511 | 42,346 | 29,153 | 35,068 |
| PERIOD | 229,258 | 32,132 | 20,246 | 25,111 |
| QUESTION | 18,999 | 3,403 | 2,175 | 2,773 |
| EXCLAMATION | 1,922 | 156 | 204 | 155 |
| COLON | 3,385 | 425 | 166 | 273 |
| SEMICOLON | 1,553 | 104 | 120 | 139 |
| DASH | 9,242 | 1,508 | 784 | 615 |

Casing masks preserve sequence positions:

| File | Masked positions |
| --- | --- |
| train | 9,075 |
| validation | 1,212 |
| test | 900 |
| test_amb | 697 |

Full per-mode distributions, casing categories, lengths and pairwise overlap
counts are in `stats.json` and `leakage_controlled/report.json`.

## Limitations and handoff

- Every canonical A/C sequence ends PERIOD or QUESTION; source boundaries are a shortcut.
- The full official splits retain source overlap and unresolved labels; they are
  supplementary evaluation only. Apply the frozen ID lists explicitly.
- test and test_amb share 25 videos after filtering. Report them separately; do not
  pool them as independent observations.
- Channel overlap remains (149 of 522 channels span original splits). Video grouping
  and duplicate checks do not guarantee independence between every podcast source.
- Low-comma, TTS, speaker/casing and punctuation rules are documented heuristics.
- Canonical and native scores use different gold labels; distinguish them.
- The corrected bundle is on the team Drive next to the superseded v1 folder. Individual consumer
  acknowledgments are not recorded.
- The evaluator applies the `case_joint` masks to casing and joint scores
  (`src/punct_restore/evaluation/inputs.py`; [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)).

No model was tuned or officially evaluated as part of this data review.
