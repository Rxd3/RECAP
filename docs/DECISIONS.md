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

## Open decisions requiring agreement

| ID | Pending question | Lead and required reviewers | Blocks |
| --- | --- | --- | --- |
| D1 | Source revision/configuration/fields, license and supported download method | 1; 5, 6 | Data release |
| D2 | Tokenization, lowercasing proposal, contractions/numbers/abbreviations and window boundaries | 1; 2, 3, 4, 5, 6 | All aligned records |
| D3 | Consecutive marks, quotes, brackets, leading/terminal marks, Unicode dashes, unusual tokens | 1; 5, 6 | Punctuation extraction/decoding |
| D4 | One-letter uppercase precedence, mixed-case words and uncased tokens | 1; 3, 4, 5, 6 | Casing and C-mode release |
| D5 | Proposed COLON/DASH→COMMA, EXCLAMATION/SEMICOLON→PERIOD versus benchmark rules | 1; 2, 3, 5 | B→A/C projection and cross-mode comparison |
| D6 | Talk/document grouping, native split compatibility, duplicates across years/configs | 1; 5, 6 | Frozen splits and fair test |
| D7 | CRF extractor injection, feature names, basic/improved presets and available context | 3 and 4; 6 | CRF feature implementation |
| D8 | HMM smoothing/OOV/state boundaries and CRF backend/tuning protocol | 2 and 3; 5 | Reproducible training |
| D9 | Separate casing versus joint study and compatible comparison target | 3 and 4; 1, 5 | Casing ablation design |
| D10 | Real ASR availability versus synthetic noise; alignment and reporting unit | 1 and 5; 6 | Noisy evaluation |
| D11 | Artifact distribution, environment locking and CLI flags | 6; 1, 2, 3, 5 | Reproducible integration |
| D12 | Optional confidence intervals, group-resampling unit and runtime measurement settings | 5; 1, 2, 3, 6 | Uncertainty/runtime report |

Do not add label classes to avoid an annotation decision. Preserve the agreed
4/8/12 target spaces; review and version any explicit mapping/exclusion policy.
