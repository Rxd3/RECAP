# Data sample (10 utterances)

Ten real utterances from release `sponspeech-m1-v2`, so the record format can be seen without
downloading the release. They are SponSpeech transcripts: CC BY-NC 4.0, non-commercial use only, with
the upstream attribution in [DATA_LICENSE.md](../../docs/DATA_LICENSE.md).

**How they were drawn.** `random.Random(2026).sample(...)` picked 10 IDs from the primary validation
lists that modes A, B and C share (the 19,936 IDs of `primary_eval/validation_mode_C.ids`, which are
also in the A and B lists). Each row was then copied byte for byte from the release's validation files.
No test or test_amb utterance is included.

| File | Content |
| --- | --- |
| `sample.ids` | The 10 `sample_id`s, sorted. |
| `mode_A_validation.jsonl` | Mode A rows: 4 punctuation labels (canonical B→A projection). |
| `mode_B_validation.jsonl` | Mode B rows: 8 punctuation labels. |
| `mode_C_validation.jsonl` | Mode C rows: the same punctuation as mode A; the joint target is `PUNCT\|CASE`. |
| `case_joint_mask_validation.jsonl` | The matching `case_joint` mask rows (none of these 10 has a masked position). |

The three mode files hold the same 10 utterances with the same tokens (241 tokens in total).

**Record fields** ([DATA_CONTRACT.md](../../docs/DATA_CONTRACT.md)):

- `schema_version` (1), `sample_id`, `source` (`sponspeech`), `split` (`validation`), `label_mode` (`A`, `B` or `C`);
- `tokens`: the model input, lowercased words;
- `punct_labels`: the mark after each token; `case_labels`: LOWER, FIRST_UPPER or ALL_UPPER for each token;
- `metadata`: `group_id` (source video), `original_text`, `normalized_text`, character `offsets` per token,
  `source_split` (`dev`), `upstream_id`, `policy_version`, `source_revision`, `annotation_view`, `condition`.

**Mask fields:** `sample_id`, `excluded_positions` (token positions left out of casing and joint scores)
and `reason`.

Models read only `sample_id` and `tokens`; everything else is gold or provenance.
