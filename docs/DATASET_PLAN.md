# Dataset inspection and preparation plan

Owner: Member 1; Member 5 reviews leakage/split/evaluation decisions, Member 6 reviews reproducibility.
**Status:** SponSpeech is the project dataset (D1). Policies D2–D6 were approved by the team lead on 2026-10-28
(policy `m1-v1`, [DECISIONS.md](DECISIONS.md)); sign-off by the listed reviewers happens in the Member 1 PR.
Release `sponspeech-m1-v1` is built locally and is ready for the team's
[Google Drive folder](https://drive.google.com/drive/folders/1ZKDNnRHXsLJf3UjOdKVgQ0bYpo12IQfH?usp=sharing);
**nothing has been uploaded yet.** All numbers below come from `data/processed/sponspeech-m1-v1/stats.json`.

## Sources

| Source | Role | Observations |
| --- | --- | --- |
| SponSpeech (**selected, D1**) | Spontaneous podcast speech with punctuated, cased transcripts | See next section |
| [IWSLT/iwslt2017](https://huggingface.co/datasets/IWSLT/iwslt2017) | Not selected | Revision `c18a4f8`; card license `cc-by-nc-nd-4.0`; data in per-pair zip files read by `iwslt2017.py` (not executed) |
| [cdminix/iwslt2011](https://huggingface.co/datasets/cdminix/iwslt2011) | Not selected | Revision `787665f`; repository holds only `iwslt2011.py`, no card, data or license |

## SponSpeech source

| Item | Value |
| --- | --- |
| Reference | Liu & Beigi (2024), *Spontaneous Informal Speech Dataset for Punctuation Restoration*, arXiv:2409.11241 |
| Archive | `https://storage.googleapis.com/sponspeech/sponspeech.tar.gz`, 63,547,617,096 bytes, GCS generation `1716975130398953` (source revision) |
| License | CC BY-NC 4.0 (non-commercial; attribution list in the archive `README.txt`, shipped with every release as `SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt`) |
| Content | 206,632 single-line ASCII transcripts + `utt2spk` per split; 16 kHz audio (not downloaded, not redistributed) |
| Native splits | train 147,209 · dev 25,253 · test 17,697 · test-amb 16,473 (matches the paper's Table 2) |
| Groups | source video ID (`utt2spk`), 2,408 videos |
| ASR | none distributed: only the clean condition exists (`metadata.condition = "clean"`); no synthetic noise generated (D10) |

**Download procedure (transcripts only).** `src/punct_restore/data/extraction.py` reads the text sections of the
archive with HTTP range requests, without the audio. The archive is a single gzip stream, so each range is decoded
from a recorded deflate block start with an empty 32 KiB window; ranges start inside audio data so the window is
refilled before the first transcript, and only checksum-valid tar headers are used. The archive size, GCS generation,
range offsets, deflate block positions and the SHA-256 of every range are pinned in the module; any change stops the
extraction. The result is compared file by file with the raw `MANIFEST.sha256` (206,638 files: 206,632 transcripts,
four `utt2spk`, `README.txt`, `.build`). The original extraction was re-verified independently (other offsets,
Python `tarfile`: 206,632/206,632 byte-identical). Fallback: `--archive` streams a fully downloaded archive. Commands
and the raw layout: [data/README.md](../data/README.md).

**Metadata snapshot.** `video_metadata_oembed.jsonl` holds the YouTube oEmbed title and channel of each of the 2,408
videos (2,333 answered), fetched once on 2026-10-28 and kept fixed. It is used for two things: flagging `tts_source`
(9 videos, 425 records) and the channel statistics in the leakage report. Re-fetching it can change titles, so the
flags, ID lists and checksums would no longer match the release. It ships in the release's provenance package.

## Provenance checklist

- [x] Source identity, revision, structure, splits, license and fields recorded; `configs/datasets.yaml` updated.
- [x] Immutable source files under `data/raw/<source_revision>/` (local, ignored by Git).
- [x] Regeneration command: [data/README.md](../data/README.md).
- [x] Extraction tool, raw `MANIFEST.sha256` and the dated metadata snapshot published with the release (provenance
  package); their checksums are recorded in `release_manifest.json`.

## Annotation policy `m1-v1` (approved D2–D5)

- **D2 tokens.** Contractions, numbers (`28.5`, `10,000`, `9:30`, `-12`), URLs, emails, abbreviations (`u.s.`),
  identifiers (`build.rs`, `rack::attack`) and word-internal hyphens stay inside tokens; target boundary marks go to
  the gap after the word; casing is read before `str.lower()`; one sequence per utterance (1–384 tokens).
- **Cleaning** (logged per record in `audit/normalization.jsonl`): timecodes 874, non-speech/name tags 4,502,
  ALL-CAPS speaker labels 3,535, recurring Title-case speaker labels 6,055, glued labels 24, `Speaker N` 203, `>>` 12,
  leading turn dashes 3,059, `[? word ?]` markers 38 (the word is kept).
- **D3 mode B.** `? ! . ; : ,` → QUESTION, EXCLAMATION, PERIOD, SEMICOLON, COLON, COMMA; `--` and standalone spaced
  hyphens → DASH; a hyphen glued to a word (cut-off) is O; negative signs stay in the number. Ellipsis → O inside an
  utterance, PERIOD at its end; this loses the distinction between hesitation, trailing off and sentence end.
- **Not covered by the approval, flagged instead:** several marks in one gap (the strongest is used:
  QUESTION > EXCLAMATION > PERIOD > SEMICOLON > COLON > COMMA > DASH), ambiguous abbreviation or initial
  boundaries, kept bracket/paren text, possible speaker labels, asterisk text, and cross-check mismatches.
- **D5 canonical mode A** = approved projection of mode B (`configs/label_sets.yaml`); mode C = canonical A × casing.
  **Native view** (`benchmark_native/mode_A`): the SponSpeech four-class labels (`!`/`;` → PERIOD, `:` → COMMA,
  dash → O) for comparison with the original benchmark only.
- **D4 casing** precedence: no cased letters → LOWER; single uppercase letter → FIRST_UPPER; all uppercase →
  ALL_UPPER; otherwise first cased letter. Mixed-case approximations are masked per position in `masks/case_joint/`.

### Canonical versus native labels (exact changes)

| File | Tokens | Labels changed | Records changed | Change type |
| --- | ---: | ---: | ---: | --- |
| train | 4,324,477 | 9,762 | 7,301 | O → COMMA |
| validation | 793,071 | 1,869 | 1,405 | O → COMMA |
| test | 574,988 | 1,258 | 936 | O → COMMA |
| test_amb | 609,532 | 832 | 645 | O → COMMA |

Every difference is a dash-only boundary. Scores on the two views are never compared as if their gold were the same.

## Splits and leakage (approved D6)

- Original membership kept: train → `train`, dev → `validation`, test → `test`, test-amb → `split: "test"` in its own
  `test_amb.jsonl` with `metadata.source_split = "test-amb"`. `sample_id = sponspeech:<source split>:<upstream id>`.
- **Overlap unit:** source video ID. Also audited: identical transcripts and near-duplicates (≥ 50% of an
  utterance's word 6-grams found in the reference). Channel (podcast) overlap is a documented remaining risk:
  149 of 522 channels span splits in the dated metadata snapshot.
- **Frozen leakage-controlled subsets** (`data/splits/sponspeech-m1-v1/leakage_controlled/`, excluded IDs with
  reasons in `*_excluded.tsv`): validation excludes overlap with train; test and test_amb exclude overlap with
  train ∪ validation; ASR-like (low-comma) and TTS sources are excluded as well.

| Subset | Official | Kept | Videos | Shared video | Identical text | Near-duplicate | Low-comma / TTS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 25,252 | 20,539 | 218 | 4,390 | 29 | 123 | 171 |
| test | 17,695 | 12,669 | 153 | 4,716 | 24 | 95 | 191 |
| test_amb | 16,472 | 13,188 | 316 | 3,096 | 16 | 93 | 79 |

**Pairwise verification** (independent string-set check; shared videos / identical texts / near-duplicates):

| Pair | Official splits | Leakage-controlled subsets |
| --- | --- | --- |
| validation vs train | 54 / 1,683 / 4,210 | 0 / 0 / 0 |
| test vs train | 36 / 986 / 2,527 | 0 / 0 / 0 |
| test_amb vs train | 52 / 531 / 1,311 | 0 / 0 / 0 |
| test vs validation | 13 / 712 / 1,853 | 0 / 0 / 0 |
| test_amb vs validation | 26 / 539 / 1,572 | 0 / 0 / 0 |
| test_amb vs test | 26 / 587 / 1,521 | 25 / 574 / 1,463 |

test and test_amb still share videos; they are two test sets and must be reported separately, never pooled.
No test labels were used to set any preprocessing rule.

## Release statistics (`sponspeech-m1-v1`)

| File | Source | Released | Accepted | Flagged (unresolved) | Rejected | Videos | Tokens | Mean / median / max length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | 147,209 | 147,165 | 141,814 | 5,351 | 44 | 1,736 | 4,324,477 | 29.4 / 23 / 384 |
| validation | 25,253 | 25,252 | 24,361 | 891 | 1 | 277 | 793,071 | 31.4 / 24 / 182 |
| test | 17,697 | 17,695 | 16,928 | 767 | 2 | 205 | 574,988 | 32.5 / 25 / 182 |
| test_amb | 16,473 | 16,472 | 15,848 | 624 | 1 | 397 | 609,532 | 37.0 / 29 / 307 |

- **Rejected (48):** 45 `inverted_mark_folded` (upstream turned `¿`/`¡` or another non-ASCII mark into `?`/`!`),
  3 `question_inside_word`; listed in `audit/rejected.jsonl` and never in a view.
- **Unresolved reasons** (records, all files; A and B identical, C adds casing): mark cluster 2,443, kept bracket 1,351,
  low-comma video 1,108, possible speaker label 483, TTS source 425, single-initial boundary 376, abbreviation boundary
  295, asterisk text 93, reconstruction mismatch 91, uncertain words 30; C only: no capitals in transcript 777,
  all-caps transcript 484. Listed with token positions in `audit/unresolved.jsonl`.

**Default training and primary evaluation subsets** (unresolved records excluded; evaluation = leakage-controlled):

| Mode | Default train | Primary validation | Primary test | Primary test_amb |
| --- | ---: | ---: | ---: | ---: |
| A | 142,791 | 20,055 | 12,328 | 12,751 |
| B | 142,791 | 20,055 | 12,328 | 12,751 |
| C | 141,814 | 19,951 | 12,283 | 12,721 |

**Canonical mode A, % of tokens (all released records)**

| File | O | COMMA | PERIOD | QUESTION |
| --- | ---: | ---: | ---: | ---: |
| train | 87.31 | 6.58 | 5.65 | 0.47 |
| validation | 86.98 | 7.19 | 5.26 | 0.57 |
| test | 86.94 | 7.25 | 5.25 | 0.56 |
| test_amb | 86.75 | 7.32 | 5.32 | 0.61 |

**Mode B token counts — default train / primary validation / primary test / primary test_amb**

| Label | Default train | Validation | Test | Test_amb |
| --- | ---: | ---: | ---: | ---: |
| O | 3,616,996 | 538,946 | 355,132 | 420,585 |
| COMMA | 259,733 | 42,370 | 29,175 | 35,094 |
| PERIOD | 229,397 | 32,158 | 20,261 | 25,136 |
| QUESTION | 19,029 | 3,410 | 2,179 | 2,778 |
| EXCLAMATION | 1,933 | 160 | 207 | 158 |
| COLON | 3,402 | 425 | 169 | 277 |
| SEMICOLON | 1,557 | 104 | 120 | 139 |
| DASH | 9,265 | 1,512 | 784 | 620 |

**Casing (all released tokens):** LOWER / FIRST_UPPER / ALL_UPPER = train 87.85 / 11.26 / 0.89 %; validation
89.36 / 10.30 / 0.34 %; test 89.04 / 10.64 / 0.32 %; test_amb 89.81 / 9.93 / 0.26 %.
**Casing masks:** train 9,075 positions in 7,603 records (0.21% of tokens); validation 1,212 / 1,020 (0.15%);
test 900 / 775 (0.16%); test_amb 697 / 568 (0.11%).

**Mode C (train, all released):** O|LOWER 3,336,910 · O|FIRST_UPPER 407,924 · COMMA|LOWER 230,913 · PERIOD|LOWER
213,154 · COMMA|FIRST_UPPER 49,976 · O|ALL_UPPER 30,752 · PERIOD|FIRST_UPPER 27,144 · QUESTION|LOWER 18,254 ·
PERIOD|ALL_UPPER 3,852 · COMMA|ALL_UPPER 3,636 · QUESTION|FIRST_UPPER 1,735 · QUESTION|ALL_UPPER 227.
The rarest primary-evaluation C class is QUESTION|ALL_UPPER: 18 / 9 / 15 tokens in validation / test / test_amb.

**Sentence-final position:** every utterance ends in PERIOD or QUESTION; 45.8–56.0% of PERIOD labels are utterance-final.

## Release handoff

- [x] Validation: all 16 view files (A, B, C, native A × 4 files) pass the DATA_CONTRACT checks on read-back; all views
  share IDs, order, tokens and splits; raw files unchanged before and after.
- [x] Audits: rejected, unresolved, normalization, leakage report with excluded IDs, split map, casing masks.
- [x] Fixture-only tests in `tests/test_data_contract.py`; two independent builds compared through `SHA256SUMS`.
- [x] Release notes, license/attribution file and checksums inside the release.
- [ ] Upload to the shared Google Drive folder and acceptance of the same release by Members 2, 3, 5 and 6.
- [ ] Member 5 to accept the casing/joint mask format (DATA_CONTRACT.md) for the evaluation contract.
