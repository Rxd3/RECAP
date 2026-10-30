# Decision register

Record proposal, rationale, affected contracts, reviewers, resolution evidence,
policy version and PR link. All decisions D1–D12 are resolved. Members must discuss shared
interface changes before implementation.

## Established scaffold choices

| ID | Choice | Owners |
| --- | --- | --- |
| S1 | Independent `src/punct_restore` package; thin future scripts; YAML settings | All, coordinated by Member 6 |
| S2 | A=4 punctuation, B=8 punctuation, C=4×3=12 joint labels | All |
| S3 | Labels refer to punctuation after a token and casing of that token | Members 1–5 |
| S4 | Shared versioned JSONL, fixed splits, no gold exposed at prediction | All |
| S5 | Non-O punctuation macro F1 primary; zero_division=0; accuracy secondary | Member 5, reviewed by model owners |
| S6 | Six feature branches from one scaffold commit; PRs into main | All |

## Resolved decisions

Approved by the team lead on 2026-10-28 and reviewed with corrections on 2026-10-29.
Current policy `m1-v2` and release `sponspeech-m1-v2` supersede `m1-v1`/`sponspeech-m1-v1`.
The lead authorized fixing the release findings, verifying the corrected data and merging
[Member 1 PR #1](https://github.com/Rxd3/RECAP/pull/1) if all checks pass. This records the lead's decision;
it does not impersonate individual reviewer sign-offs. Verification is recorded in
[DATA_RELEASE_VALIDATION.md](DATA_RELEASE_VALIDATION.md).

| ID | Resolution | Rationale | Affected contracts | Reviewers |
| --- | --- | --- | --- | --- |
| D1 | SponSpeech (Liu & Beigi 2024, arXiv:2409.11241), archive GCS generation `1716975130398953`, CC BY-NC 4.0; transcripts only (no audio); IWSLT sources not used | Spontaneous speech with cased, punctuated transcripts; license allows non-commercial derived data with attribution | `configs/datasets.yaml`, `docs/DATASET_PLAN.md` | 5, 6 |
| D2 | Contractions, numbers, URLs and abbreviations stay whole; meaningful internal punctuation is kept; target boundary punctuation is removed from tokens; casing is extracted before `str.lower()`; one sequence per utterance; identical tokens across A/B/C and all models | Keeps word alignment stable and avoids leaking targets into inputs | DATA_CONTRACT tokens/normalization | 2, 3, 4, 5, 6 |
| D3 | `--` and standalone spaced hyphens → DASH; word-internal hyphens and negative signs stay in tokens; ellipsis → O inside an utterance, PERIOD at the end (this loses some punctuation detail); unresolved records keep their reasons and source IDs, are excluded from default training and primary evaluation lists, and stay available for ambiguity analysis. Cases not covered (e.g. several marks in one gap) are flagged for review, not decided | Uses only the extended labels; no silent deletion | DATA_CONTRACT label modes, release audit files | 5, 6 |
| D4 | Casing precedence: (1) no cased letters → LOWER; (2) single uppercase letter → FIRST_UPPER; (3) several cased letters, all uppercase → ALL_UPPER; (4) otherwise first cased letter (upper → FIRST_UPPER, lower → LOWER). Step-4 approximations (`iPhone`) are masked per token position from primary casing/joint metrics; punctuation stays scored | Deterministic, no fourth casing class; masks keep sequences and positions intact | DATA_CONTRACT casing; mask interface to agree with Member 5 | 3, 4, 5, 6 |
| D5 | Canonical B → A projection as in `configs/label_sets.yaml` (EXCLAMATION, SEMICOLON → PERIOD; COLON, DASH → COMMA); canonical modes A and C use it. Native SponSpeech four-class labels are kept as a separate, versioned benchmark view; canonical and native scores are never compared as if their gold labels were the same | One consistent standard target; native view for original-benchmark comparison | `configs/label_sets.yaml`, EVALUATION_PROTOCOL cross-mode comparison | 2, 3, 5 |
| D6 | Original split membership kept; `dev` → `validation`; test-amb in its own file with `split: "test"` and `metadata.source_split`; overlap unit = source video ID; exact original-text and lowercased token-tuple checks (including short sequences), near-duplicate (≥ 50% shared word 6-grams) and shared-video audits; leakage-controlled validation/test subsets frozen before model selection and used for primary evaluation; full official splits for supplementary reporting | Upstream splits share videos; primary results must not depend on leaked content | DATA_CONTRACT splits, `data/splits/` manifests | 5, 6 |

## Resolved model, evaluation and distribution decisions (D7–D12)

These were open questions until the CRF and the evaluation were built; the outcomes below are what
the repository implements.

| ID | Question | Outcome | Evidence |
| --- | --- | --- | --- |
| D7 | CRF feature injection, feature names, basic/improved presets | `extract_sequence_features(tokens, *, config)` with a `FeatureConfig` in `src/punct_restore/features/crf_features.py`; the basic and improved presets were written by Member 3 (`aa232be`). Member 5 later added the filler-cue switch, off in both presets. | [FEATURE_NOTES.md](../experiments/FEATURE_NOTES.md) |
| D8 | CRF backend and tuning protocol | `sklearn-crfsuite` 0.5.0 over `python-crfsuite` 0.9.12, L-BFGS; c1 = c2 = 0.1 and 100 iterations, fixed and not tuned (every fit reached the iteration cap). HMM choices are recorded below. | [CRF_EXPERIMENTS.md](CRF_EXPERIMENTS.md) §2, §6 |
| D9 | Separate casing versus joint study | Joint mode C is compared with punctuation-only mode A (H1). Each mode C model, projected to punctuation, is scored against its mode A twin on mode A gold over the same 19,936 mode C validation IDs; the training sets differ (A 142,722 sequences, C 141,745). | [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md), Team hypotheses |
| D10 | Real ASR versus synthetic noise | No ASR or synthetic-noise condition: SponSpeech ships no ASR transcripts, and RECAP evaluates the reference transcripts only. | [DATASET_PLAN.md](DATASET_PLAN.md) |
| D11 | Artifact distribution, environment and CLI flags | Data, HMM bundle and CRF models are shared on the team Google Drive, each with SHA-256 sums; the CRF dependencies are pinned in `pyproject.toml` and the HMM environment in `requirements-hmm.txt`; CLI flags are documented per script. | [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [scripts/README.md](../scripts/README.md) |
| D12 | Confidence intervals, resampling unit, runtime settings | Grouped bootstrap by video (`metadata.group_id`), 1,000 resamples, seed 2026, 95% percentile intervals, paired for comparisons. Runtime: median of 3 passes after a 200-record warm-up, model loading excluded. | [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md), [EVALUATION_RESULTS_CRF.md](EVALUATION_RESULTS_CRF.md) §10 |

Do not add label classes to avoid an annotation decision. Preserve the agreed
4/8/12 target spaces; review and version any explicit mapping/exclusion policy.

## Member 2 implementation record (2026-10-29)

The team lead authorized completing and double-checking the HMM baseline.
This records implementation and experimental evidence; it does not substitute
for individual teammates' review or resolve CRF and integration choices.

- D8, HMM: supervised counts, additive smoothing, learned START/END boundaries,
  log-space Viterbi and a training-frequency UNK bucket. Validation selected
  alpha 0.01 in A/B/C, rare-word cutoff 1 in A/B and 2 in C from the declared
  nine-candidate grid per mode. No validation data enters parameter estimation.
- D11, HMM: versioned JSON artifacts, explicit training/prediction IDs and an
  authenticated release manifest. CPython 3.13.5 / PyYAML 6.0.3 passed isolated
  reproduction. The checksummed handoff is on the team Drive
  (`Models/hmm-sponspeech-v2-20261029.zip`); teammate receipt is not recorded.
- D12, HMM measurement: common validation tokens, one model thread, 200 warmup
  sequences, three in-memory prediction repetitions and a separate decode
  latency pass. Hardware, timing scope and raw repetitions are preserved. No
  confidence intervals or statistical-significance claims are made.

See [HMM notes](../experiments/HMM_NOTES.md),
[configuration](../configs/hmm_sponspeech_v2.yaml) and
[verified results](../experiments/HMM_RESULTS.md). The shared scorer is pinned
at `3044a58e777200bcac64e6e3364922404be359bf` (Member 5 PR #2).

## Policy m1-v2 corrections (2026-10-29)

- D3: mixed ellipsis/other-mark boundaries are unresolved clusters even at the end of an utterance. The
  provisional label is retained only in the full/audit views; such records are excluded from primary use.
- D6: exact lowercased token-tuple overlap is checked in addition to original-text, video and six-gram overlap,
  including sequences shorter than six tokens. No silent short-phrase exemption is applied.
- Reproducibility: persisted paths use POSIX separators and generated text uses LF; raw hashes and view validation
  must pass before a manifest/checksum set can be published. Existing release IDs cannot be overwritten.
- The 4/8/12 vocabularies, B-to-A mapping, casing rules, utterance tokenization and original split membership are
  unchanged. Version 1 files/checksums are preserved; consumers must choose the version 2 release explicitly.
