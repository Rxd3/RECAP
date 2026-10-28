# Final technical presentation outline — planned

Owner: Member 6; every member supplies evidence and explains their methodology.
No final deck or results are created during scaffolding. Adapt slide count to
the course requirements once confirmed; no duration or deadline is assumed.

| Section | Content and evidence | Contributors |
| --- | --- | --- |
| Motivation/task | Unformatted English, boundary labels and truecasing; illustrative target clearly marked | 6, all |
| Project progression | Document classification in NALTRA versus sequence labeling here | 6 |
| Data and provenance | Verified sources/configs, terms, policies, cleaning and rejected examples | 1 |
| Labels and splits | A=4, B=8, C=4×3=12; ambiguity decisions, class support and leakage prevention | 1, 5 |
| HMM baseline | State/emission assumptions, Viterbi, smoothing and OOV limitations | 2 |
| Main CRF | Linear-chain model, basic features, selected settings and shared interface | 3 |
| Improvements | Improved features, separate/joint question, controlled ablations and negative findings | 4, 3 |
| Evaluation | Non-O F1, O/zero-division policy, joint/projection targets, held-out discipline | 5 |
| Actual findings | Verified tables, confusion matrices, rare classes, representative errors and runtime | 5, 2–4 |
| Robustness | Verified real ASR or explicitly synthetic condition; alignment and limitations | 1, 5 |
| Integrated demo | Same observable input across compatible models; label/text outputs and error handling | 6 |
| Reproduction/conclusion | Commands, versions/hashes, limitations, supported conclusions and future work | 6, all |

- [ ] Every figure/table names mode, split, class support and producing run.
- [ ] Illustrative targets are visually distinguished from actual predictions.
- [ ] Sources and preprocessing rules are cited accurately.
- [ ] Native eight-class and twelve-class scores are not presented as identical tasks.
- [ ] All technical owners review their claims; every member can explain the workflow.
- [ ] Demo has compatible artifacts, documented setup and a rehearsed fallback.
