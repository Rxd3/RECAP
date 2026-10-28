# Decision register

Record proposal, rationale, affected contracts, reviewers, resolution evidence,
policy version and PR link. Pending entries are not defaults. No deadlines are
assigned. Members must discuss shared interface changes before implementation.

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

Approved by the team lead on 2026-10-28 (relayed by Member 1). Policy version `m1-v1`, implemented in
`src/punct_restore/data/` and release `sponspeech-m1-v1`. **Sign-off by the listed reviewers happens in the
Member 1 pull request and is not yet recorded.** PR link: pending.

| ID | Resolution | Rationale | Affected contracts | Reviewers |
| --- | --- | --- | --- | --- |
| D1 | SponSpeech (Liu & Beigi 2024, arXiv:2409.11241), archive GCS generation `1716975130398953`, CC BY-NC 4.0; transcripts only (no audio); IWSLT sources not used | Spontaneous speech with cased, punctuated transcripts; license allows non-commercial derived data with attribution | `configs/datasets.yaml`, `docs/DATASET_PLAN.md` | 5, 6 |
| D2 | Contractions, numbers, URLs and abbreviations stay whole; meaningful internal punctuation is kept; target boundary punctuation is removed from tokens; casing is extracted before `str.lower()`; one sequence per utterance; identical tokens across A/B/C and all models | Keeps word alignment stable and avoids leaking targets into inputs | DATA_CONTRACT tokens/normalization | 2, 3, 4, 5, 6 |
| D3 | `--` and standalone spaced hyphens → DASH; word-internal hyphens and negative signs stay in tokens; ellipsis → O inside an utterance, PERIOD at the end (this loses some punctuation detail); unresolved records keep their reasons and source IDs, are excluded from default training and primary evaluation lists, and stay available for ambiguity analysis. Cases not covered (e.g. several marks in one gap) are flagged for review, not decided | Uses only the extended labels; no silent deletion | DATA_CONTRACT label modes, release audit files | 5, 6 |
| D4 | Casing precedence: (1) no cased letters → LOWER; (2) single uppercase letter → FIRST_UPPER; (3) several cased letters, all uppercase → ALL_UPPER; (4) otherwise first cased letter (upper → FIRST_UPPER, lower → LOWER). Step-4 approximations (`iPhone`) are masked per token position from primary casing/joint metrics; punctuation stays scored | Deterministic, no fourth casing class; masks keep sequences and positions intact | DATA_CONTRACT casing; mask interface to agree with Member 5 | 3, 4, 5, 6 |
| D5 | Canonical B → A projection as in `configs/label_sets.yaml` (EXCLAMATION, SEMICOLON → PERIOD; COLON, DASH → COMMA); canonical modes A and C use it. Native SponSpeech four-class labels are kept as a separate, versioned benchmark view; canonical and native scores are never compared as if their gold labels were the same | One consistent standard target; native view for original-benchmark comparison | `configs/label_sets.yaml`, EVALUATION_PROTOCOL cross-mode comparison | 2, 3, 5 |
| D6 | Original split membership kept; `dev` → `validation`; test-amb in its own file with `split: "test"` and `metadata.source_split`; overlap unit = source video ID; exact, near-duplicate (≥ 50% shared word 6-grams) and shared-video audits; leakage-controlled validation/test subsets frozen before model selection and used for primary evaluation; full official splits for supplementary reporting | Upstream splits share videos; primary results must not depend on leaked content | DATA_CONTRACT splits, `data/splits/` manifests | 5, 6 |

## Open decisions requiring agreement

| ID | Pending question | Lead and required reviewers | Blocks |
| --- | --- | --- | --- |
| D7 | CRF extractor injection, feature names, basic/improved presets and available context | 3 and 4; 6 | CRF feature implementation |
| D8 | HMM smoothing/OOV/state boundaries and CRF backend/tuning protocol | 2 and 3; 5 | Reproducible training |
| D9 | Separate casing versus joint study and compatible comparison target | 3 and 4; 1, 5 | Casing ablation design |
| D10 | Real ASR availability versus synthetic noise; alignment and reporting unit | 1 and 5; 6 | Noisy evaluation |
| D11 | Artifact distribution, environment locking and CLI flags (team lead prefers the shared Google Drive folder for data releases) | 6; 1, 2, 3, 5 | Reproducible integration |
| D12 | Optional confidence intervals, group-resampling unit and runtime measurement settings | 5; 1, 2, 3, 6 | Uncertainty/runtime report |

Do not add label classes to avoid an annotation decision. Preserve the agreed
4/8/12 target spaces; review and version any explicit mapping/exclusion policy.
