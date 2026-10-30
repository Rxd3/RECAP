# sponspeech-m1-v2

Source: SponSpeech, Liu, X. Y. and Beigi, H. (2024). Spontaneous Informal Speech Dataset for Punctuation Restoration. arXiv:2409.11241. https://storage.googleapis.com/sponspeech/sponspeech.tar.gz
License: CC BY-NC 4.0 (non-commercial use only). Attribution list: SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt
Source revision: gcs-generation-1716975130398953. Policy: m1-v2. Schema: v1.
Code SHA-256: e6fcbddaa3c7d9ef7d35ed12533bd37c8ed1ea5982a5241f7cf50ae09c6c6600

## Records

| file | released | flagged (unresolved) | rejected |
|---|---:|---:|---:|
| train | 147165 | 5420 | 44 |
| validation | 25252 | 902 | 1 |
| test | 17695 | 777 | 2 |
| test_amb | 16472 | 639 | 1 |

## Default and primary subsets (sample counts)

| mode | default train | validation | test | test_amb |
|---|---:|---:|---:|---:|
| A | 142722 | 20040 | 12321 | 12734 |
| B | 142722 | 20040 | 12321 | 12734 |
| C | 141745 | 19936 | 12276 | 12704 |

## Reproducing from the original source

- Archive: https://storage.googleapis.com/sponspeech/sponspeech.tar.gz (63547617096 bytes, GCS generation 1716975130398953).
- `python scripts/prepare_data.py extract` recreates the transcripts from pinned byte ranges (or from a fully downloaded archive with `--archive`) and checks them against `MANIFEST.sha256` (SHA-256 69f9b227d2d25bdb245b2a6b4129a0f5b2bd9d14bda14b9cc491d54d31a10591).
- The dated metadata snapshot (SHA-256 86d8d3681f193d27ae990aaf1dd4b6a8e280e6e8d2dbeb08649aa48fa1c669a3) flags 9 text-to-speech videos; flagged records per file: {'train': 405, 'validation': 20, 'test': 0, 'test_amb': 0}. They are excluded from the default and primary ID lists, so building without this exact snapshot gives different lists.

## Use

- Train on `mode_<M>/train.jsonl` restricted to `default_train/mode_<M>.ids`.
- Select on `primary_eval/validation_mode_<M>.ids`; report on `primary_eval/test_mode_<M>.ids` and, separately, `primary_eval/test_amb_mode_<M>.ids`. Full official files are supplementary results.
- Mode C: target = `join_labels(punct_labels, case_labels)`; exclude `masks/case_joint` positions from casing and joint metrics.
- `benchmark_native/` is for comparison with the original SponSpeech four-class benchmark only. Never compare its scores with canonical mode A scores as if the gold labels were the same.
- Models receive `sample_id` and `tokens` only.
