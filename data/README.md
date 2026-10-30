# Data storage and release plan

Owner: Member 1. Generated contents of `raw/`, `processed/` and `splits/` are ignored by Git. Tracked: this guide,
the folder markers, a 10-utterance sample in [`sample/`](sample/README.md) and small release metadata in
[`release_metadata/`](release_metadata/README.md).
Do not add large data with `git add -f`. Releases are shared through the team's Google Drive folder
([RECAP -NLP2](https://drive.google.com/drive/folders/1ZKDNnRHXsLJf3UjOdKVgQ0bYpo12IQfH?usp=sharing); decision D11) with SHA-256 sums.
Current release `sponspeech-m1-v2` uses policy `m1-v2` and supersedes `sponspeech-m1-v1-rev2`.
Version 2 fixes Windows paths/newlines, flags mixed ellipsis boundaries, and excludes exact model-input duplicates
from primary subsets. Version 1 remains unchanged as historical data. The corrected bundle is on the team
Drive as `dataset/sponspeech-m1-v2-handoff.zip` (see [below](#getting-the-release-from-google-drive)).

- `raw/`: immutable source files, one folder per source revision.
- `processed/`: versioned JSONL releases following the [contract](../docs/DATA_CONTRACT.md).
- `splits/`: sample/group assignments, ID lists and checksum manifests for each release.

## Source: SponSpeech (D1 approved by the team lead)

The team lead approved SponSpeech as the project dataset (`configs/datasets.yaml`, `docs/DECISIONS.md`).
See [DATASET_PLAN.md](../docs/DATASET_PLAN.md).

Raw root `RAW = data/raw/sponspeech-gcs-generation-1716975130398953/` (ignored by Git):

```text
RAW/MANIFEST.sha256               SHA-256 of all 206,638 raw files (from the release's provenance package)
RAW/video_metadata_oembed.jsonl   dated YouTube title snapshot, 2026-10-28 (from the provenance package)
RAW/sponspeech_text/              train/ dev/ test-amb/ test/ (text/*.txt, utt2spk), README.txt, .build
```

1. **Environment.** From the repository root: `python -m venv .venv`, activate it, then
   `python -m pip install -e ".[dev]"` (the pinned PyYAML, sklearn-crfsuite, python-crfsuite and joblib, plus pytest;
   the data code itself needs only PyYAML and the standard library).
2. **Provenance files.** Take `<release>-provenance.tar.gz` from the Drive handoff ZIP (below), check it against
   `<release>.SHA256SUMS`, and unpack it into `RAW`. Do not re-fetch the title snapshot: titles change over time.
3. **Recreate the transcripts** (no audio):

   ```bash
   python scripts/prepare_data.py extract --raw-root data/raw/sponspeech-gcs-generation-1716975130398953
   ```

   This downloads only the pinned byte ranges of the official archive (about 97 MB: the first 20 MiB plus the four
   text sections) and fails if the archive size, its GCS generation or the SHA-256 of any range has changed, or if
   any extracted file differs from `MANIFEST.sha256`. **Fallback** (the ranges can no longer be read): download the
   whole 63.5 GB `sponspeech.tar.gz` and run the same command with `--archive <path>/sponspeech.tar.gz`; it streams
   the archive, keeps only non-audio members and runs the same manifest check. If the official archive itself has
   changed, both methods stop: the published release then cannot be rebuilt from source, only used from Drive.
4. **Build.**

   ```bash
   python scripts/prepare_data.py build --raw-root data/raw/sponspeech-gcs-generation-1716975130398953 \
     --release-id sponspeech-m1-v2
   ```

   Both provenance files are required. `MANIFEST.sha256` is checked before and after the build. The title snapshot
   flags `tts_source` (9 "Wikipedia audio article" videos, 425 records: 405 train, 20 validation); those records are
   unresolved, so without the snapshot the default training and primary evaluation ID lists and all checksums differ.
5. **Verify.** `python -m pytest`, then check the build against the published checksums:
   `diff data/processed/<release>/SHA256SUMS <published copy>` (same for `data/splits`).

## Release layout (`sponspeech-m1-v2`, policy `m1-v2`)

```text
data/processed/<release>/
  mode_A/{train,validation,test,test_amb}.jsonl   canonical standard punctuation (D5 projection) + case_labels
  mode_B/{...}.jsonl                              extended punctuation (D3) + case_labels
  mode_C/{...}.jsonl                              canonical standard punctuation + case_labels
  benchmark_native/mode_A/{...}.jsonl             native SponSpeech four-class labels (comparison only)
  masks/case_joint/{...}.jsonl                    per record: positions excluded from casing/joint metrics (D4)
  audit/rejected.jsonl       records excluded from every view, with reasons and source text
  audit/unresolved.jsonl     released records outside the approved rules, per mode, with token positions
  audit/normalization.jsonl  every removed non-lexical string (timecodes, non-speech tags, speaker labels)
  RELEASE_NOTES.md  SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt  stats.json  release_manifest.json  SHA256SUMS
data/splits/<release>/
  split_map.tsv                          every source record: sample_id, split, source split, file, video, status
  default_train/mode_{A,B,C}.ids         training records without unresolved labels
  primary_eval/{validation,test,test_amb}_mode_{A,B,C}.ids   leakage-controlled and resolved: primary evaluation
  leakage_controlled/{validation,test,test_amb}.ids, *_excluded.tsv, report.json   frozen subsets (D6)
  SHA256SUMS
```

All views hold the same records in the same order with the same `sample_id`, `tokens` and `split`.
`test_amb.jsonl` is SponSpeech's second test set: `split: "test"`, `metadata.source_split: "test-amb"`; never
concatenate it with `test.jsonl` and report it separately (D6).

## Getting the release from Google Drive

The team Drive folder [`dataset/`](https://drive.google.com/drive/folders/1jVl5pOQ6PvWPQ6MQxXLcbUksJJ1KDo5u) holds:

- `sponspeech-m1-v2-handoff.zip`: the current release, 224,632,100 bytes, SHA-256
  `1e4b4a9504c68cd8e3af8d6a2a0ef7de5de0146c76457752a56a90dcf6504b9f` ([DATA_RELEASE_VALIDATION.md](../docs/DATA_RELEASE_VALIDATION.md));
- `sponspeech-m1-v1-rev2/`: version 1, superseded and kept as history only.

The ZIP contains one folder, `sponspeech-m1-v2/`, with seven files: `<id>-processed.tar.gz`,
`<id>-splits.tar.gz`, `<id>-provenance.tar.gz` (raw manifest, title snapshot, extraction logs; only
needed to rebuild from source), `<id>.SHA256SUMS`, `<id>-RELEASE_NOTES.md`, `<id>-VALIDATION.json` and
`SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt` (here `<id>` is `sponspeech-m1-v2`). Data or selection changes
need a new release ID, not replacement files under an old name. Check the ZIP's SHA-256, unzip it, then
run these from the extracted `sponspeech-m1-v2/` folder, with `data/` meaning the repository's data folder:

```bash
shasum -a 256 -c <id>.SHA256SUMS
tar -xzf <id>-processed.tar.gz -C data/processed
tar -xzf <id>-splits.tar.gz -C data/splits
(cd data/processed/<id> && shasum -a 256 -c SHA256SUMS)
(cd data/splits/<id> && shasum -a 256 -c SHA256SUMS)
```

SponSpeech is CC BY-NC 4.0: no commercial use; keep `SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt` with any copy.

## Rules for consumers

```python
from punct_restore.data import join_labels, load_records, read_ids
rel, spl = "data/processed/sponspeech-m1-v2", "data/splits/sponspeech-m1-v2"
train_ids = set(read_ids(f"{spl}/default_train/mode_A.ids"))
train = [r for r in load_records(f"{rel}/mode_A/train.jsonl") if r["sample_id"] in train_ids]
```

- `load_records` raises on any contract violation. Model inputs are `sample_id` and `tokens` only; strip `metadata`
  and gold labels before prediction (MODEL_CONTRACT.md). Do not retokenize, resplit or re-derive casing.
- Train on `default_train`, select on `primary_eval/validation_*`, report on `primary_eval/test_*` and, separately,
  `primary_eval/test_amb_*`. Full official files give supplementary results; report them in a separate table.
- Mode C target: `join_labels(r["punct_labels"], r["case_labels"])`; exclude `masks/case_joint` positions from
  primary casing and joint metrics (punctuation at those positions is still scored); report all-token scores separately.
- `benchmark_native/` is only for comparison with the original SponSpeech four-class benchmark; never compare its
  scores with canonical mode A scores as if the gold labels were identical.
- Every canonical A/C sequence ends in PERIOD or QUESTION, so the evaluation also reports scores without the final token (the `non_final` view).

## Release validation and immutability

The builder refuses an existing release ID and stops on raw-manifest or output-validation failures. Manifest
paths and SHA256SUMS use POSIX separators; generated text uses UTF-8/LF on every platform. Primary subsets reject
shared videos, exact original transcripts, exact lowercased token tuples (including short phrases) and the
documented six-gram near-duplicates. Mixed ellipsis/other-mark boundaries remain available in the full views but
are flagged and excluded from default training/primary evaluation under D3.

See [the release validation record](../docs/DATA_RELEASE_VALIDATION.md) for checks, corrected counts and handoff.
