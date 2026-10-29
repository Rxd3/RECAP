# Member 1 corrected release validation

Reviewed on 2026-10-29 on Windows, Python 3.13.5. Release **sponspeech-m1-v2**, policy
**m1-v2**, schema v1, supersedes `sponspeech-m1-v1-rev2`. The team lead authorized
these corrections, verification and merging [PR #1](https://github.com/Rxd3/RECAP/pull/1)
when the checks passed. No individual team-member sign-off is implied.

## Findings corrected

1. Windows paths broke inferred label modes, manifest keys and raw-file matching.
   Persisted paths now use POSIX separators; generated text uses UTF-8/LF.
2. Terminal mixed ellipses bypassed the unresolved cluster flag. All mixed
   ellipsis/other-mark boundaries are now flagged before the terminal fallback;
   affected records remain in full views and are excluded from primary use.
3. Raw-text and six-gram matching missed short, model-identical inputs. Primary
   filtering and the independent overlap report now check exact token tuples.

Existing release IDs cannot be overwritten. Raw/checksum, view alignment or
validation failures prevent publication of manifests/checksum sets. Bounded
parallel source reads preserve order; cross-view checks stream aligned rows to
reduce memory use. Truncated-view and other failure paths have regression tests.

## Verification completed

- **115 data tests passed**; **174 tests passed** with Member 2's HMM branch merged
  in an isolated integration checkout. Ruff F/E9, `git diff --check` and a package
  installation dry run pass. PyYAML is declared as a runtime dependency.
- Pinned source ranges were re-extracted. All **206,638** raw files match the raw
  manifest, with no missing or extra files. The build checks raw hashes both before
  and after processing.
- **826,336 records** independently audited across 16 view files: shared IDs,
  ordering/tokens/splits, lowercasing, allowed labels and exact array alignment.
- Original-text hashes reconcile all **206,632** source IDs with **206,584** retained
  utterances and **48** rejected IDs. Split-map membership and rejection accounting pass.
- All offsets, casing labels and mask positions were independently recomputed.
  Canonical A = projection(B); C = A × casing. The **13,721** native/canonical
  differences are exclusively O → COMMA at DASH boundaries.
- Default/primary IDs exactly match the documented exclusions. All primary subsets
  have zero source-video, raw exact, normalized exact and qualifying six-gram overlap
  with their reference splits. No selected terminal mixed-ellipsis examples remain.
- Compatibility-only HMM fits on 20 selected training utterances per mode and
  predictions on five primary-validation utterances per mode pass alignment,
  gold-field invariance and save/load checks; no scores were calculated.
- All release/split manifests and SHA256SUMS entries match. The fixed metadata
  snapshot, raw manifest, code and label configuration fingerprints match.
- Every full view is byte-identical to v1 after changing only `policy_version`.
  Masks, rejected records and normalization logs are byte-identical. Changed primary
  selections are strict subsets of v1; reasons/counts are in the new audit artifacts.
- Two independent fixture builds produce identical bytes. The full real-data release
  was regenerated once and independently audited; a second full real-data rebuild
  was not run in this review.

## Frozen fingerprints

| Item | SHA-256 |
| --- | --- |
| Data package source files | `e6fcbddaa3c7d9ef7d35ed12533bd37c8ed1ea5982a5241f7cf50ae09c6c6600` |
| Label configuration | `eeba7991ce7e897fd610ed1928d6939242b2e76806e5839b63903f6e92fa1142` |
| Raw manifest | `69f9b227d2d25bdb245b2a6b4129a0f5b2bd9d14bda14b9cc491d54d31a10591` |
| Metadata snapshot | `86d8d3681f193d27ae990aaf1dd4b6a8e280e6e8d2dbeb08649aa48fa1c669a3` |
| Processed SHA256SUMS | `19ed874f9ae3de4489bdc668aaad6581c24b56e00d57f2d671d8a4c2908d4bbb` |
| Splits SHA256SUMS | `40ea6371b038d5d36e6da390493bfc0e45c513ec46c324ee1a6af8766e7546b7` |

The manifest records the Python version and exact checksums for every generated file.
Corrected subset counts/distributions and remaining source limitations are recorded
in [DATASET_PLAN.md](DATASET_PLAN.md).

## Handoff

- Local outputs: `data/processed/sponspeech-m1-v2/` and `data/splits/sponspeech-m1-v2/`.
- Corrected archives, provenance, validation JSON, notes/license and checksums:
  `artifacts/data/sponspeech-m1-v2/`; combined archive: `artifacts/data/sponspeech-m1-v2-handoff.zip`.
- Data stays outside Git. Upload the new bundle to the existing shared Drive folder;
  version 2 is **not yet uploaded**. The old v1/rev2 Drive folder is superseded.
- Consumers must select the mode's `default_train` / `primary_eval` IDs explicitly;
  passing a split-manifest path to the existing HMM CLI only hashes that path.
- The 174-test integration check verifies implementation compatibility. No official
  full-data HMM training, hyperparameter selection or benchmark scoring was run.
