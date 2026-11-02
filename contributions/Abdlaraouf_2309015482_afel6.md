# Abdlaraouf Alkabir (Abd Raouf) — Member 5: evaluation and error analysis; release integration

| Field | Value |
| --- | --- |
| Student ID | 2309015482 |
| GitHub | [@afel6](https://github.com/afel6) |
| Git author name(s) | `Raouf` |
| Branch | `feat/m5-evaluation`; integration branch `release/final` |
| Pull request | [#2](https://github.com/Rxd3/RECAP/pull/2) — feat(eval): evaluation pipeline, shortcut baseline and CLI (Member 5) (open on 2026-10-30; merged into `release/final` by 097c87b) |

## Contributions in this repository

- **Evaluation (Member 5):** evaluation pipeline, shortcut baseline and CLI (680438f); grouped bootstrap intervals, paired comparison, error examples and the manifest check (6356916); policy version in reports (ec37d19); fixes 9155a7e and 6340b59. Wrote `docs/EVALUATION_RESULTS.md` (51bd400) and `docs/EVALUATION_RESULTS_CRF.md` (255fc68, 5562e32, 44d81a0, fe9710b, 15e5ebf).
- **Release integration:** prepared `release/final`, merging the four member branches (097c87b, 0790467, 611f3f4, ddba805) and the final-push work branches (668e035, 052cdd6, 9248ce8, a10dc3a, ab5d784, 354d3fb, 72fd442, ba0d2b5).
- **Final push, code:** CRF marginals and prediction manifests (fbd8d45); threshold decoding (a930859); filler cue and the ablation runner (e7095cf); demo loading real HMM and CRF models (4304e4c); CLI fixes (01858ae, 3ca76d5, c575279, 31e59f9); clean-ups of scaffold stubs and a byte-order mark (5002994, 7c75d4e); exact dependency pins (54eed84) and the tested-environment lock file (d4ab315); the H1 projection script (6507152) and its mode B option (ca96025), `compare --class` per-class intervals (eb6afa7), leakage-subset builder (b973981), reporting scripts (4998b31, aad6c70) and a train_hmm.py pointer to the scorer (d3284b9).
- **Final push, results:** reproduced Member 3's six validation files byte for byte and scored the CRFs once on test and test_amb; ran the threshold, filler-cue, leakage and runtime studies (fe9710b), the per-class intervals and label-granularity study (9802964) and the settings sensitivity check (102393e); committed the evaluator outputs behind every reported number (d2953ce, fc68dbd) and Member 3's training manifests (15e5ebf).
- **Final push, documentation:** the commits authored as `Raouf` in this push also include the repository documentation (licence, contribution files, README, dataset card, related work, runbooks, notes), from 54a5666 to 72b9148 (licence, contribution files, README, dataset card, related work, model contract, data sample, runbooks, team board).
- **Comparison methods (exploratory, after the first test report):** gap classifier (cfc8b20), Stolcke-style hidden-event LM (e743928), prior-adjusted τ × γ decoding (b875ea3), question-lookback features (5373a82), casing-only CRF mode and factored pipeline (649bd53), demo loader for all four model families (c7934ef); results in `docs/EVALUATION_RESULTS_CRF.md` section 14 (1bc35eb, 6c8130c, 3f10555).
- **Robustness and data checks:** simulated ASR noise (910e552, 6c64226) and its v2 edit model with seed variance (7889149, dcfe950, 3814f38, efdd761, b2840dc); the gold-noise finder (3ec548b) and the section 12 rerun (a60428c); the controlled H1 check (a5c3ac8); CRF convergence (f4dee77) and length/position tables (596310b); τ files with input hashes (c0453c8, eafde4c).
- **Documentation after the regrades:** exploratory labels, related work and decision records (20c2a51, 4c7619b, 6d72f6d, f889d0f).

## Commits

| Commit | Date | Subject |
| --- | --- | --- |
| `008ef72` | 2026-11-02 | docs: date the fresh-checkout check to the 2026-11-02 CI run |
| `fe4e267` | 2026-11-02 | docs: gate-fix error wordings (score fields, fitted CRF, table commands, label audit) |
| `c169320` | 2026-11-02 | fix(review): by-design audit notes matched by their exact wordings, mode selection and leakage messages name their file and line |
| `a43827e` | 2026-11-02 | fix(review): refuse non-finite and out-of-range numbers, unfitted CRF artifacts and bad folders by name |
| `14d3042` | 2026-11-02 | refactor(review): lazy exports hidden from type checkers, one select-by-ID path, label-audit by-design check |
| `3ad578d` | 2026-11-02 | docs: CI lint, mypy and coverage gate, the remaining error lines, public-docstring test |
| `712d5a0` | 2026-11-02 | ci: pin mypy and coverage, type-check and gate coverage at 80 percent, Node 24 actions |
| `591c658` | 2026-11-02 | fix(messages): every error names its file or folder, one UTF-8 and YAML wording, per-row HMM counts, public docstrings |
| `08a32d9` | 2026-11-02 | docs: name the errors module by its import path |
| `38fe854` | 2026-11-02 | docs: errors and lazy-export modules, one record selector, recap result-stats, 37-entry HMM closure, new tests |
| `fe99422` | 2026-11-02 | feat(reporting): recap result-stats regenerates holm.json, clusters_summary.json and label_audit.json |
| `4c75a98` | 2026-11-02 | refactor(dedup): one record selector and joblib loader, errors module, file identity in file_io, lazy data/evaluation package inits |
| `1023ba2` | 2026-11-02 | docs: list test_non_text_inputs in tests/README |
| `a0c2f17` | 2026-11-02 | docs: list the error lines users see, the new file_io checks and the round-3 tests |
| `4fd586f` | 2026-11-02 | fix(errors): main's HMM label order, checked CRF artifacts and reporting inputs, release-folder and manifest messages, per-line CRF record checks |
| `d40744c` | 2026-11-02 | docs: list the pinned data-pipeline test in tests/README |
| `fbb5857` | 2026-11-02 | test: pin the data pipeline's output to main 72ebfd1 |
| `a89e0fa` | 2026-11-02 | docs: correct the error-helper name and HMM list history, drop verbal significance for the neural reference, explain the fingerprint lines in SHA256SUMS |
| `8b97fe8` | 2026-11-02 | test: make the deep-nesting and folder-listing tests platform-independent |
| `54a49f3` | 2026-11-01 | fix(review): non-string labels refused in every model, uniform <file>: <reason> messages, one YAML reader and folder check |
| `a6e62a9` | 2026-11-01 | fix(review): shape-first HMM config and study checks, one schema checker, require_file and iter_jsonl everywhere, named artifact errors |
| `62deed8` | 2026-11-01 | refactor(types): mypy-clean src, descriptive names, no getattr leftovers, help texts without team roles |
| `0760fbd` | 2026-11-01 | docs: name the package initializers and the threshold module without file-path forms |
| `1e7833e` | 2026-11-01 | docs: tau grid in models.threshold, runtime-closure HMM provenance, main's signatures, new error wording and tests |
| `3b4db1c` | 2026-11-01 | refactor(api): main's call signatures restored, runtime-closure HMM provenance, read-only HMM view, tau grid in models.threshold |
| `a00883c` | 2026-11-01 | refactor(dedup): file_io helpers (iter_jsonl, write_new_folder, JSON readers), one joblib loader, one pct, one file identity |
| `0b78f4a` | 2026-11-01 | fix(errors): shape-checked study files, named config and folder errors, TrainingFailed, typed label checks |
| `b0f7508` | 2026-11-01 | docs: ID-list rule, exact demo cleaning, public-name promise, cli wording; one copy of the old-study steps |
| `21b2cd0` | 2026-11-01 | docs: demote the neural reference, drop the unmeasured source-error figure, word leakage headlines descriptively |
| `8f37466` | 2026-11-01 | docs: list the HMM study and library-name tests in tests/README |
| `1c7cb9d` | 2026-11-01 | test: run the HMM study in-process and pin its tune, gate, finalize, verify and handoff steps |
| `831dbb3` | 2026-11-01 | fix(api): keep old library import paths (restructure spec §2) |
| `14f42d2` | 2026-11-01 | docs: restore the library import-path promise and list test_library_names |
| `81e07f8` | 2026-11-01 | docs: follow the fix-round module moves, error lines, CI matrix and new tests |
| `c6ca363` | 2026-11-01 | fix(errors): label config, titles and manifest files report one named error line; split the ablation tests |
| `a5e1796` | 2026-11-01 | fix(errors): every malformed input file is one error line naming it, with command-level regression tests |
| `3e0f05d` | 2026-11-01 | fix(review): config errors as one line, shared JSON-line and refuse-existing checks, silent HMM library, full HMM provenance closure |
| `2864c94` | 2026-11-01 | chore(cli): command summaries without team roles, run-ablations argument help, CI matrix 3.12/3.13 with lock constraints |
| `d51552c` | 2026-11-01 | fix(security): refuse archive entries that resolve outside the extraction folder |
| `7b22ef9` | 2026-11-01 | refactor(types): annotate every function, descriptive names, main(argv) signatures, one error prefix, decorators count in the limit |
| `904965e` | 2026-11-01 | refactor(structure): HMM study and CRF predict/inspect logic in the library, no cli-to-cli imports, no argument mutation |
| `465e3c8` | 2026-11-01 | refactor(dedup): one sha256 helper, one .ids reader, one JSON-line parser and new-file writer |
| `ffdb8ea` | 2026-11-01 | fix(errors): one error boundary, one-line errors with exit 1, no catch-all handlers |
| `061cf63` | 2026-11-01 | test: pin refactor outputs against main 72ebfd1 |
| `24a92d2` | 2026-11-01 | docs: address restructure grade findings (old HMM studies, neural reference, toy run, caveats) |
| `5d3551c` | 2026-11-01 | chore: keep YAML workflow files at LF line endings |
| `65c01ff` | 2026-11-01 | chore(lint): add ruff config, fix all findings, remove dead code, add CI |
| `bc7b337` | 2026-11-01 | docs: name load_records and read_ids as the release consumer API in ARCHITECTURE |
| `a296c4e` | 2026-11-01 | docs: switch commands to the recap CLI and describe the new layout |
| `6eb7be5` | 2026-11-01 | refactor(cli): split long command functions into library helpers and enforce size limits |
| `574a4de` | 2026-11-01 | refactor(models): split CRF, hidden-event LM and HMM code into short functions and modules |
| `d36c926` | 2026-11-01 | refactor(features): split CRF feature extraction into short helpers and lexicon/cue modules |
| `b6703d9` | 2026-11-01 | refactor(data): split release building and normalize into short functions and modules |
| `5b3683b` | 2026-11-01 | refactor(cli): move the command scripts into punct_restore.cli |
| `e8d66bb` | 2026-11-01 | refactor(reporting): move reporting commands into the package |
| `6738680` | 2026-11-01 | refactor(cli): add recap dispatcher and REPO_ROOT |
| `8d71b2d` | 2026-11-01 | docs(eval): attribute the neural training and selection claims to Member 2 |
| `a5d1c8b` | 2026-11-01 | docs(eval): score Member 2's BERT and FLAN-T5 taggers and add section 15 |
| `3d4985c` | 2026-11-01 | docs(contributions): regenerate the contribution files after Member 2's rename |
| `cc55140` | 2026-11-01 | docs(contributions): add Member 2's student ID |
| `e685607` | 2026-11-01 | docs(results): name the label-audit sheet among the committed excerpts |
| `ee38614` | 2026-11-01 | docs: name the label-audit sheet wherever committed dataset excerpts are listed |
| `5c5aa30` | 2026-11-01 | docs(data): add the 100-record label audit (hand-checked half and agent-only half reported separately) |
| `11bb5c3` | 2026-11-01 | docs(eval): add the word-cluster comparison as section 14.6 |
| `9864d80` | 2026-11-01 | docs(contributions): regenerate the contribution files after Member 4's rename |
| `c685d08` | 2026-11-01 | docs(contributions): add Member 4's student ID and GitHub handle |
| `d80a141` | 2026-11-01 | docs: add the word-cluster builder, switch and ablation variant to the file maps |
| `09bc390` | 2026-10-31 | feat(features): add an opt-in word-cluster feature switch and its ablation variant |
| `d3c86dc` | 2026-10-31 | feat(features): induce word clusters from default_train with PPMI, SVD and KMeans |
| `30514af` | 2026-10-31 | docs(contributions): regenerate the contribution files after the renames |
| `16f4b74` | 2026-10-31 | docs(contributions): add student IDs for Members 1, 3 and 6 |
| `2831c4c` | 2026-10-31 | docs(contributions): regenerate the contribution files at the current release/final head |
| `4b2c616` | 2026-10-31 | docs: map the ASR-noise report modules and refresh the section 13.1 seed-spread line |
| `e543704` | 2026-10-31 | docs(eval): regenerate ASR_NOISE_V2.md and its summary with the fixed generator |
| `597dd43` | 2026-10-31 | fix(reporting): compute the ASR-noise v2 findings from the data and round once |
| `84f10d5` | 2026-10-31 | fix(data): validate ASR-noise offsets and make the train-only confusion test able to fail |
| `a2df0db` | 2026-10-31 | docs(contributions): regenerate the contribution files at the current release/final head |
| `f889d0f` | 2026-10-31 | docs(eval): fold the ASR noise v2 study into section 13, README and D10, and fix the regrade doc items |
| `b2840dc` | 2026-10-30 | docs: add ASR_NOISE_V2 report and update documentation maps |
| `efdd761` | 2026-10-30 | test: add unit and safety tests for asr_noise_table reporting script |
| `3814f38` | 2026-10-30 | feat: add asr_noise_table reporting script and multi-seed ASR noise scores |
| `dcfe950` | 2026-10-30 | test: add v1 golden and v2 edit model tests for ASR noise simulation |
| `7889149` | 2026-10-30 | feat: add v2 simulated ASR edit model and --edit-model CLI option |
| `eafde4c` | 2026-10-30 | docs(eval): apply the regrade number check and regenerate tau files with input hashes |
| `6d72f6d` | 2026-10-30 | docs: document the gold-noise finder, CRF convergence and length/position tools and update the noise counts |
| `a60428c` | 2026-10-30 | docs(eval): rerun the gold-noise check with the committed finder and report CRF convergence |
| `c0453c8` | 2026-10-30 | feat(crf): record the split and input hashes in tau files |
| `596310b` | 2026-10-30 | feat(reporting): add length_position.py for the length-bucket and position tables |
| `f4dee77` | 2026-10-30 | feat(reporting): add crf_convergence.py for the CRF loss traces |
| `3ec548b` | 2026-10-30 | feat(data): add find_gold_noise.py with speaker-label, cue and auto-caption rules |
| `3f10555` | 2026-10-30 | docs(eval): per-class intervals for prior-adjusted decoding, H1 PERIOD intervals and exploratory labels |
| `4c7619b` | 2026-10-30 | docs: mark the test-informed methods as exploratory and fix the regrade wording |
| `20c2a51` | 2026-10-30 | docs: update related work for the exploratory methods and add prior-shift and hidden-event references |
| `345bc97` | 2026-10-30 | docs(contributions): add Member 5's name and student ID |
| `284c593` | 2026-10-30 | fix(results): commit the gold-noise excluded-ID lists and use true minus signs in section 14 |
| `a0e9c57` | 2026-10-30 | docs: list both HMM markers and the order of load_model's checks |
| `6c8130c` | 2026-10-30 | docs(eval): correct comparison-method claims found by the number check |
| `0eeffba` | 2026-10-30 | docs: document the demo loader's four model families and its errors |
| `c7934ef` | 2026-10-30 | feat(demo): load hidden-event LM and gap-classifier artifacts |
| `1bc35eb` | 2026-10-30 | docs(eval): add team comparison methods (gap classifier, hidden-event LM, prior-adjusted decoding, question lookback, factored casing) |
| `e377552` | 2026-10-30 | docs: document the comparison methods, robustness checks and their CLIs |
| `649bd53` | 2026-10-30 | feat(crf): add a casing-only CRF mode and a factored punctuation + casing pipeline |
| `6c64226` | 2026-10-30 | docs(eval): add the simulated ASR-noise robustness check |
| `910e552` | 2026-10-30 | feat(data): add simulate_asr_noise.py for a simulated ASR-noise robustness study |
| `5373a82` | 2026-10-30 | feat(features): add the question lookback and tag-question cues as an off-by-default extension |
| `b875ea3` | 2026-10-30 | feat(crf): add prior-adjusted per-class threshold decoding (tau x gamma) |
| `e743928` | 2026-10-30 | feat(models): add a Stolcke-style hidden-event language model |
| `cfc8b20` | 2026-10-30 | feat(models): add the gap classifier (CRF without transitions) and its ablation variant |
| `a5c3ac8` | 2026-10-30 | docs(eval): add the controlled H1 check on identical training IDs |
| `488a0dd` | 2026-10-30 | docs(eval): create output folders before the documented evaluator commands |
| `c3030b9` | 2026-10-30 | docs(eval): add the mode B gold-noise check |
| `55eb978` | 2026-10-30 | docs: map the score folders and folder guides, drop the folder markers and note the recorded Python version |
| `1da392b` | 2026-10-30 | docs: resolve the final gate's wording contradictions |
| `73618ee` | 2026-10-30 | docs: regenerate contribution files at the final commit |
| `72b9148` | 2026-10-30 | docs: document the final CLI options and checks |
| `102393e` | 2026-10-30 | docs(eval): add the CRF settings sensitivity check |
| `9802964` | 2026-10-30 | docs(eval): add per-class intervals for H2-H4 and the label-granularity study |
| `aad6c70` | 2026-10-30 | fix(reporting): refuse overwrites and derive report figures from the evaluator JSON |
| `45a3deb` | 2026-10-30 | docs(eval): link the original hypothesis text |
| `8c52025` | 2026-10-30 | docs: state what changed after the board freeze |
| `6e125d5` | 2026-10-30 | docs: freeze the team board at the final state |
| `d3284b9` | 2026-10-30 | fix(hmm): point train_hmm.py at evaluate_models.py score |
| `ca96025` | 2026-10-30 | feat(eval): project mode B predictions to mode A with project_joint.py --from B |
| `eb6afa7` | 2026-10-30 | feat(eval): add compare --class for per-class F1 difference intervals |
| `d4ab315` | 2026-10-30 | build(deps): add requirements-lock.txt with the tested environment's exact freeze |
| `f5a4721` | 2026-10-30 | docs: fix the testing-gate findings in results, runbooks and maps |
| `1c1ebd5` | 2026-10-30 | docs: commit a 10-utterance data sample and the release metadata the dataset card cites |
| `eacfa31` | 2026-10-30 | docs(eval): drop an uncommitted reference-CRF number |
| `8c65522` | 2026-10-30 | docs: add the model table by implementation, team-method related work and the pre-registered hypotheses |
| `f9082a7` | 2026-10-30 | docs: tidy the test guide, artifact locations, runbook placeholders and score index |
| `5c786a4` | 2026-10-30 | docs: remove the git workflow guide, historical sections and unused folder markers |
| `4d33521` | 2026-10-30 | docs: finalize the data contract, dataset plan and stale notes |
| `3948341` | 2026-10-30 | docs: rewrite the model contract to the implemented interfaces |
| `e00b3d1` | 2026-10-30 | docs(eval): add Holm check and leakage reproduction commands |
| `fc68dbd` | 2026-10-30 | docs(eval): record final protocol state and commit the remaining evaluator outputs |
| `45554af` | 2026-10-30 | docs: regenerate contribution files at the final commit |
| `d27de6f` | 2026-10-30 | docs: document the reporting and study scripts |
| `15e5ebf` | 2026-10-30 | docs(eval): add reproduce commands and the CRF training manifests |
| `4998b31` | 2026-10-30 | feat(reporting): port the result-table and timing scripts into scripts/reporting |
| `b973981` | 2026-10-30 | feat(data): add build_leakage_subsets.py to rebuild the leakage ID lists |
| `6507152` | 2026-10-30 | feat(eval): add project_joint.py for the joint-vs-punctuation-only study |
| `54eed84` | 2026-10-30 | build(deps): pin every runtime and dev dependency exactly |
| `9d26cb6` | 2026-10-30 | docs: name the docs index in the folder map |
| `472abc5` | 2026-10-30 | docs: add official-split overlap and the leakage cost by cause to the dataset card |
| `d418fc5` | 2026-10-30 | docs: separate course baselines from team additions and add classical related work |
| `3645bfb` | 2026-10-30 | docs: document the Drive layout, model locations, hardware and committed scores |
| `f97855a` | 2026-10-30 | docs: bring architecture, scope, decisions and CRF notes to the final state |
| `766aeb7` | 2026-10-30 | docs: remove scaffold-era files and fold the HMM completion audit into HMM_RESULTS |
| `7c75d4e` | 2026-10-30 | chore: strip the UTF-8 BOM from crf_features.py and its test |
| `31e59f9` | 2026-10-30 | fix(cli): turn remaining path errors into one-line messages |
| `d2953ce` | 2026-10-30 | docs(eval): commit the evaluator outputs behind every reported number |
| `71886ec` | 2026-10-30 | docs: complete the package map |
| `5002994` | 2026-10-30 | chore: remove scaffold TODO stubs and tighten the is_repeat test |
| `c575279` | 2026-10-30 | fix(cli): survive non-cp1252 stdout in run_demo and accept upper-case model suffixes |
| `3ca76d5` | 2026-10-30 | fix(cli): refuse colliding or unwritable outputs and leave no partial predictions |
| `bbf8367` | 2026-10-30 | docs: remove contributing stub and write tau files under experiments/runs |
| `b752c42` | 2026-10-30 | docs: sync notes and runbooks with the final CRF results |
| `fe9710b` | 2026-10-30 | docs(eval): add final CRF test results, threshold decoding, filler cue, leakage and runtime |
| `8fecfbd` | 2026-10-30 | docs: drop stale demo package path and install caveat |
| `01858ae` | 2026-10-30 | fix(crf): let train_crf.py run from a checkout without installation |
| `9fe6f00` | 2026-10-30 | docs: sync docs with the merged backend |
| `4304e4c` | 2026-10-30 | fix(demo): load real HMM and CRF artifacts and validate labels |
| `e7095cf` | 2026-10-30 | feat(features): add filler cues and a reproducible ablation runner |
| `a930859` | 2026-10-30 | feat(crf): add validation-tuned threshold decoding |
| `fbd8d45` | 2026-10-30 | feat(crf): add predict_marginals and prediction manifests; declare CRF dependencies |
| `a7ab470` | 2026-10-30 | docs: refresh stale notes and team board with evidence |
| `b6a3799` | 2026-10-30 | docs: add dataset card and related work |
| `54a5666` | 2026-10-30 | docs: add license, contributions, dataset licence and rewrite README and runbooks |
| `44d81a0` | 2026-10-30 | docs(eval): correct native-label difference rounding |
| `5562e32` | 2026-10-30 | docs(eval): add leakage and native-benchmark supplementary rows |
| `255fc68` | 2026-10-30 | docs(eval): add CRF validation results, feature ablations and hypothesis outcomes |
| `51bd400` | 2026-10-29 | docs(eval): add HMM evaluation results with confusion matrices, error analysis and runtime |
| `6340b59` | 2026-10-29 | fix(eval): round Markdown percentages half-up from the stored 6-decimal value |
| `9155a7e` | 2026-10-29 | fix(eval): accept the HMM bundle's nested inputs.input in the manifest check |
| `ec37d19` | 2026-10-29 | feat(eval): record the gold data policy version in reports |
| `6356916` | 2026-10-29 | feat(eval): add grouped bootstrap CIs, paired compare, error examples and manifest check |
| `680438f` | 2026-10-29 | feat(eval): add evaluation pipeline, shortcut baseline and CLI |

Merge commits:

| Commit | Date | Subject |
| --- | --- | --- |
| `a0575d3` | 2026-11-02 | merge: final round docs |
| `9a30be2` | 2026-11-02 | merge: data pipeline pinned to main 72ebfd1 |
| `4a85cc2` | 2026-11-02 | merge: final round backend (label order, CRF artifact checks, reporting errors, result-stats, CI checks) |
| `b992e29` | 2026-11-02 | Merge pull request #9 from Rxd3/refactor/package-cli |
| `2d6b4c9` | 2026-11-01 | merge: fix round 2 docs |
| `7c37d30` | 2026-11-01 | merge: fix round 2 backend (errors, dedup, API signatures, types, review fixes) |
| `8950c71` | 2026-11-01 | merge: in-process HMM study tests (fix item 24) |
| `0c7085e` | 2026-11-01 | merge: docs follow-up for round 1 module moves and import-path promise |
| `3dd35b3` | 2026-11-01 | merge: refactor equivalence tests pinned to main 72ebfd1 |
| `7a30f5e` | 2026-11-01 | merge: docs fix round 1 (items 26-34) |
| `0760384` | 2026-11-01 | Merge branch 'wip/restructure-docs' into refactor/package-cli |
| `72ebfd1` | 2026-11-01 | Merge pull request #7 from Rxd3/release/final |
| `302536b` | 2026-11-01 | chore: merge word-cluster features into release/final |
| `3053cbb` | 2026-10-31 | chore: merge ASR-noise v2 report fixes into release/final |
| `1dbc843` | 2026-10-31 | chore: merge simulated-ASR v2 edit model and seed variance into release/final |
| `6ffed31` | 2026-10-30 | chore: merge gold-noise and new-tool docs into release/final |
| `e574e4b` | 2026-10-30 | chore: merge gold-noise finder, convergence and length/position reports into release/final |
| `9a68a99` | 2026-10-30 | chore: merge regrade wording fixes into release/final |
| `ac4966f` | 2026-10-30 | chore: merge model-contract loader-order fix into release/final |
| `2a42c2d` | 2026-10-30 | chore: merge model-contract docs for the demo loader into release/final |
| `d33d1ca` | 2026-10-30 | chore: merge demo loader for hidden-event LM and gap classifier into release/final |
| `4c5e4ad` | 2026-10-30 | chore: merge comparison-method docs into release/final |
| `c4f9a40` | 2026-10-30 | chore: merge factored casing pipeline into release/final |
| `b69130c` | 2026-10-30 | chore: merge simulated ASR-noise tool into release/final |
| `09b8845` | 2026-10-30 | chore: merge per-class thresholds and question-lookback features into release/final |
| `8b08845` | 2026-10-30 | chore: merge gap classifier and hidden-event LM into release/final |
| `c1c94c8` | 2026-10-30 | chore: merge wip/docs grader fixes into release/final |
| `ba0d2b5` | 2026-10-30 | chore: merge wip/backend final round into release/final |
| `72fd442` | 2026-10-30 | chore: merge wip/docs final round into release/final |
| `fe95b44` | 2026-10-30 | chore: merge release/final into wip/docs |
| `354d3fb` | 2026-10-30 | chore: merge wip/backend re-grade work into release/final |
| `ab5d784` | 2026-10-30 | chore: merge wip/backend fix round into release/final |
| `a10dc3a` | 2026-10-30 | chore: merge wip/docs follow-up into release/final |
| `9248ce8` | 2026-10-30 | chore: merge wip/backend follow-up into release/final |
| `052cdd6` | 2026-10-30 | chore: merge wip/docs (tasks 6-7) into release/final |
| `668e035` | 2026-10-30 | chore: merge wip/backend (tasks 2-5) into release/final |
| `ddba805` | 2026-10-30 | chore: merge feat/m6-integration-presentation into release/final |
| `611f3f4` | 2026-10-30 | chore: merge feat/m3-crf-model into release/final |
| `0790467` | 2026-10-30 | chore: merge feat/m2-hmm-baseline into release/final |
| `097c87b` | 2026-10-30 | chore: merge feat/m5-evaluation into release/final |
| `aca40c6` | 2026-10-29 | chore: merge feat/m5-evaluation (main with m1-v2 data) into feat/m5-eval-uncertainty |
| `3044a58` | 2026-10-29 | chore: merge main into feat/m5-evaluation |

## Files added or changed

- repository root: 9 files (2 added); commits 0760fbd, 08a32d9, 0eeffba, 11bb5c3, 14f42d2, 16f4b74, 1bc35eb, 1c1ebd5, 1da392b, 1e7833e, 21b2cd0, 24a92d2, 284c593, 345bc97, 3645bfb, 38fe854, 3ad578d, 4b2c616, 4c7619b, 4d33521, 5002994, 54a5666, 54eed84, 55eb978, 5c5aa30, 5c786a4, 5d3551c, 65c01ff, 6738680, 6c8130c, 6d72f6d, 6e125d5, 712d5a0, 71886ec, 72b9148, 766aeb7, 81e07f8, 8c52025, 8c65522, 8fecfbd, 9802964, 9d26cb6, 9fe6f00, a0c2f17, a296c4e, a5d1c8b, a7ab470, a89e0fa, b0f7508, b2840dc, b6a3799, bbf8367, c3030b9, c685d08, cc55140, cfc8b20, d27de6f, d2953ce, d418fc5, d4ab315, d80a141, e377552, eafde4c, ee38614, f5a4721, f889d0f, f9082a7, f97855a, fbd8d45, fe9710b
- `.github/workflows/`: `ci.yml` (A in 65c01ff; M in 712d5a0, 2864c94)
- `artifacts/`: `README.md` (M in f9082a7, 5c786a4, 766aeb7)
- `artifacts/models/`: `.gitkeep` (D in 5c786a4)
- `configs/`: `README.md` (M in f97855a, 766aeb7), `datasets.yaml` (M in 3645bfb), `experiment.example.yaml` (D in 766aeb7)
- `contributions/`: 12 files (6 added); commits 16f4b74, 2831c4c, 30514af, 345bc97, 3d4985c, 45554af, 54a5666, 73618ee, 9864d80, a2df0db, a7ab470, b752c42, c685d08, cc55140
- `data/`: `README.md` (M in a0c2f17, a89e0fa, 1e7833e, a296c4e, 55eb978, 1c1ebd5, 3645bfb)
- `data/processed/`: `.gitkeep` (D in 55eb978)
- `data/raw/`: `.gitkeep` (D in 55eb978)
- `data/release_metadata/`: `README.md` (A in 1c1ebd5)
- `data/release_metadata/sponspeech-m1-v2/processed/`: `RELEASE_NOTES.md` (A in 1c1ebd5), `SHA256SUMS` (A in 1c1ebd5), `stats.json` (A in 1c1ebd5)
- `data/release_metadata/sponspeech-m1-v2/splits/`: `SHA256SUMS` (A in 1c1ebd5)
- `data/release_metadata/sponspeech-m1-v2/splits/leakage_controlled/`: `report.json` (A in 1c1ebd5)
- `data/sample/`: `README.md` (A in 1c1ebd5), `case_joint_mask_validation.jsonl` (A in 1c1ebd5), `mode_A_validation.jsonl` (A in 1c1ebd5), `mode_B_validation.jsonl` (A in 1c1ebd5), `mode_C_validation.jsonl` (A in 1c1ebd5), `sample.ids` (A in 1c1ebd5)
- `data/splits/`: `.gitkeep` (D in 55eb978)
- `docs/`: 23 files (7 added); commits 008ef72, 0eeffba, 102393e, 11bb5c3, 14f42d2, 15e5ebf, 1bc35eb, 1c1ebd5, 1da392b, 1e7833e, 20c2a51, 21b2cd0, 24a92d2, 255fc68, 284c593, 3645bfb, 38fe854, 3948341, 3ad578d, 3f10555, 44d81a0, 45a3deb, 472abc5, 488a0dd, 4b2c616, 4c7619b, 4d33521, 51bd400, 54a5666, 5562e32, 55eb978, 5c5aa30, 5c786a4, 6356916, 680438f, 6c64226, 6c8130c, 6d72f6d, 6e125d5, 72b9148, 766aeb7, 81e07f8, 8c65522, 8d71b2d, 9802964, 9fe6f00, a0c2f17, a0e9c57, a296c4e, a5c3ac8, a5d1c8b, a60428c, a89e0fa, b0f7508, b2840dc, b6a3799, b752c42, bbf8367, bc7b337, c3030b9, c685d08, d27de6f, d2953ce, d418fc5, d80a141, e00b3d1, e377552, e543704, eacfa31, eafde4c, ec37d19, ee38614, f5a4721, f889d0f, f9082a7, f97855a, fc68dbd, fe9710b
- `experiments/`: `CRF_NOTES.md` (M in a296c4e, 4c7619b, 0eeffba, 1da392b, 5c786a4, 4d33521, f97855a, b752c42, fe9710b, a7ab470, 54a5666), `FEATURE_NOTES.md` (M in a296c4e, c685d08, eafde4c, 1da392b, f5a4721, b752c42, fe9710b, a7ab470), `HMM_RESULTS.md` (M in a296c4e, 1da392b, f5a4721, 766aeb7), `HMM_RUNBOOK.md` (M in 38fe854, a89e0fa, 1e7833e, b0f7508, 81e07f8, 24a92d2, a296c4e, f5a4721), `README.md` (M in 4d33521, 766aeb7, b752c42, a7ab470)
- `experiments/runs/`: `.gitkeep` (D in 55eb978)
- `experiments/runs/recap_m3_handoff_eval/`: `README.md` (D in 766aeb7)
- `presentation/`: `README.md` (M in 5c786a4, b752c42, a7ab470)
- `presentation/assets/`: `.gitkeep` (D in 5c786a4)
- `results/`: `README.md` (M in a296c4e, a5d1c8b, e685607, 11bb5c3, f889d0f, 55eb978, 72b9148, 1c1ebd5, d27de6f, 3645bfb, b752c42, a7ab470, 680438f)
- `results/figures/`: `.gitkeep` (D in 5c786a4)
- `results/predictions/`: `.gitkeep` (D in 55eb978)
- `results/scores/`: `README.md` (A in d2953ce; M in 38fe854, a296c4e, a5d1c8b, ee38614, 5c5aa30, 11bb5c3, f889d0f, b2840dc, eafde4c, 6d72f6d, 4c7619b, 284c593, 6c8130c, 1bc35eb, 6c64226, c3030b9, 102393e, 9802964, 1c1ebd5, f9082a7, 15e5ebf), `holm.json` (A in fe99422)
- `results/scores/ablations/mode_A/`: 13 files (13 added); commits d2953ce
- `results/scores/ablations/mode_B/`: 13 files (13 added); commits d2953ce
- `results/scores/ablations/mode_C/`: 13 files (13 added); commits d2953ce
- `results/scores/asr_noise/wer0.05/`: 10 files (10 added); commits 6c64226
- `results/scores/asr_noise/wer0.10/`: 10 files (10 added); commits 6c64226
- `results/scores/asr_noise/wer0.15/`: 10 files (10 added); commits 6c64226
- `results/scores/asr_noise_v2/`: `summary.json` (A in 3814f38; M in e543704)
- `results/scores/asr_noise_v2/v1/wer0.15/seed2027/`: 10 files (10 added); commits 3814f38
- `results/scores/asr_noise_v2/v1/wer0.15/seed2028/`: 10 files (10 added); commits 3814f38
- `results/scores/asr_noise_v2/v2/wer0.05/seed2026/`: 10 files (10 added); commits 3814f38
- `results/scores/asr_noise_v2/v2/wer0.05/seed2027/`: 10 files (10 added); commits 3814f38
- `results/scores/asr_noise_v2/v2/wer0.05/seed2028/`: 10 files (10 added); commits 3814f38
- `results/scores/asr_noise_v2/v2/wer0.10/seed2026/`: 10 files (10 added); commits 3814f38
- `results/scores/asr_noise_v2/v2/wer0.10/seed2027/`: 10 files (10 added); commits 3814f38
- `results/scores/asr_noise_v2/v2/wer0.10/seed2028/`: 10 files (10 added); commits 3814f38
- `results/scores/asr_noise_v2/v2/wer0.15/seed2026/`: 10 files (10 added); commits 3814f38
- `results/scores/asr_noise_v2/v2/wer0.15/seed2027/`: 10 files (10 added); commits 3814f38
- `results/scores/asr_noise_v2/v2/wer0.15/seed2028/`: 10 files (10 added); commits 3814f38
- `results/scores/b_to_a/`: `crf_basic_B_vs_A.json` (A in 9802964), `crf_improved_B_vs_A.json` (A in 9802964), `hmm_B_vs_A.json` (A in 9802964)
- `results/scores/clusters/`: `clusters_summary.json` (A in 11bb5c3), `compare_A_k100.json` (A in 11bb5c3), `compare_A_k300.json` (A in 11bb5c3), `compare_B_k100.json` (A in 11bb5c3), `compare_B_k300.json` (A in 11bb5c3), `compare_C_k100.json` (A in 11bb5c3), `compare_C_k300.json` (A in 11bb5c3)
- `results/scores/crf_convergence/`: `convergence.json` (A in a60428c)
- `results/scores/crf_test/`: 33 files (33 added); commits d2953ce
- `results/scores/crf_training_manifests/`: `crf_mode_a_basic_full_01.manifest.json` (A in 15e5ebf), `crf_mode_a_improved_full_01.manifest.json` (A in 15e5ebf), `crf_mode_b_basic_full_01.manifest.json` (A in 15e5ebf), `crf_mode_b_improved_full_01.manifest.json` (A in 15e5ebf), `crf_mode_c_basic_full_01.manifest.json` (A in 15e5ebf), `crf_mode_c_improved_full_01.manifest.json` (A in 15e5ebf)
- `results/scores/crf_validation/`: 18 files (18 added); commits d2953ce
- `results/scores/factored/`: `factored_C.json` (A in 1bc35eb), `factored_vs_joint_C.json` (A in 1bc35eb), `improved_C.json` (A in 1bc35eb)
- `results/scores/filler/mode_A/`: `full_vs_improved_plus_filler.json` (A in d2953ce), `improved.json` (A in d2953ce), `improved_plus_filler.json` (A in d2953ce)
- `results/scores/filler/mode_B/`: `full_vs_improved_plus_filler.json` (A in d2953ce), `improved.json` (A in d2953ce), `improved_plus_filler.json` (A in d2953ce)
- `results/scores/filler/mode_C/`: `full_vs_improved_plus_filler.json` (A in d2953ce), `improved.json` (A in d2953ce), `improved_plus_filler.json` (A in d2953ce)
- `results/scores/gap/`: `gap_improved_A.json` (A in 1bc35eb), `gap_improved_B.json` (A in 1bc35eb), `gap_improved_C.json` (A in 1bc35eb), `gap_vs_crf_improved_A.json` (A in 1bc35eb), `gap_vs_crf_improved_B.json` (A in 1bc35eb), `gap_vs_crf_improved_C.json` (A in 1bc35eb)
- `results/scores/gold_noise/`: 20 files (20 added); commits 284c593, 5c5aa30, a60428c, c3030b9
- `results/scores/h1_joint/`: 11 files (11 added); commits 3f10555, a5c3ac8, d2953ce, fc68dbd
- `results/scores/hidden_event_lm/`: 45 files (45 added); commits 1bc35eb
- `results/scores/hmm/`: 26 files (26 added); commits a60428c, d2953ce, fc68dbd
- `results/scores/leakage/`: 14 files (14 added); commits d2953ce
- `results/scores/neural/`: 35 files (35 added); commits a5d1c8b
- `results/scores/per_class/`: 12 files (12 added); commits 9802964
- `results/scores/question2/`: `question2_vs_improved_A.json` (A in 1bc35eb), `question2_vs_improved_B.json` (A in 1bc35eb), `question2_vs_improved_C.json` (A in 1bc35eb)
- `results/scores/runtime/`: 9 files (9 added); commits d2953ce, fc68dbd
- `results/scores/sensitivity/`: 18 files (18 added); commits 102393e
- `results/scores/threshold/`: 15 files (15 added); commits d2953ce, eafde4c
- `results/scores/threshold_prior/`: 21 files (21 added); commits 1bc35eb, 3f10555, eafde4c
- `results/tables/`: `.gitkeep` (D in 55eb978)
- `scripts/`: 19 files (8 added); commits 01858ae, 09bc390, 0eeffba, 1da392b, 31e59f9, 38fe854, 3ad578d, 3ca76d5, 3ec548b, 4304e4c, 4b2c616, 5373a82, 54a5666, 5b3683b, 6356916, 649bd53, 6507152, 680438f, 6d72f6d, 71886ec, 72b9148, 7889149, 84f10d5, 8fecfbd, 910e552, 9fe6f00, a0c2f17, a296c4e, a930859, b0f7508, b2840dc, b875ea3, b973981, bbf8367, c0453c8, c575279, ca96025, cfc8b20, d27de6f, d3284b9, d3c86dc, d80a141, e377552, e7095cf, e743928, eafde4c, eb6afa7, ec37d19, f889d0f, fbd8d45, fe4e267
- `scripts/reporting/`: 13 files (13 added); commits 1da392b, 3814f38, 4998b31, 596310b, 597dd43, aad6c70, e8d66bb, f4dee77, f889d0f, fe99422
- `src/punct_restore/`: 9 files (7 added); commits 0b78f4a, 31e59f9, 3b4db1c, 3ca76d5, 3e0f05d, 4304e4c, 465e3c8, 4c75a98, 4fd586f, 5002994, 54a49f3, 591c658, 62deed8, 649bd53, 65c01ff, 6738680, 7b22ef9, 831dbb3, a00883c, a43827e, a5e1796, a6e62a9, c575279, c7934ef, ffdb8ea
- `src/punct_restore/cli/`: 21 files (21 added); commits 0b78f4a, 14d3042, 2864c94, 3b4db1c, 3e0f05d, 465e3c8, 4c75a98, 4fd586f, 54a49f3, 591c658, 5b3683b, 62deed8, 65c01ff, 6738680, 6eb7be5, 7b22ef9, 904965e, a00883c, a43827e, a5e1796, a6e62a9, c169320, fe99422, ffdb8ea
- `src/punct_restore/data/`: 15 files (8 added); commits 0b78f4a, 14d3042, 3b4db1c, 3e0f05d, 3ec548b, 465e3c8, 4c75a98, 4fd586f, 54a49f3, 591c658, 5b3683b, 62deed8, 65c01ff, 7889149, 7b22ef9, 831dbb3, 84f10d5, 904965e, 910e552, a00883c, a43827e, a6e62a9, b6703d9, c6ca363, d51552c, ffdb8ea
- `src/punct_restore/demo/`: `__init__.py` (D in 4304e4c), `inference.py` (D in 4304e4c)
- `src/punct_restore/evaluation/`: 10 files (7 added); commits 14d3042, 465e3c8, 4c75a98, 54a49f3, 591c658, 5b3683b, 62deed8, 6340b59, 6356916, 65c01ff, 680438f, 9155a7e, a00883c, a43827e, a5e1796, eb6afa7, ec37d19
- `src/punct_restore/features/`: `__init__.py` (M in 5002994), `crf_cues.py` (A in d36c926), `crf_features.py` (M in 62deed8, 831dbb3, 904965e, 65c01ff, d36c926, 09bc390, 5373a82, 7c75d4e, e7095cf), `crf_lexicons.py` (A in d36c926), `word_clusters.py` (A in d3c86dc; M in 4c75a98, a5e1796, 904965e, 65c01ff)
- `src/punct_restore/models/`: 31 files (27 added); commits 09bc390, 0b78f4a, 14d3042, 3b4db1c, 3e0f05d, 465e3c8, 4c75a98, 4fd586f, 5002994, 5373a82, 54a49f3, 574a4de, 591c658, 5b3683b, 62deed8, 649bd53, 65c01ff, 6eb7be5, 7b22ef9, 831dbb3, 904965e, a00883c, a43827e, a5e1796, a6e62a9, a930859, b875ea3, c169320, c6ca363, cfc8b20, e743928, eafde4c, fbd8d45, ffdb8ea
- `src/punct_restore/reporting/`: 17 files (12 added); commits 14d3042, 2864c94, 465e3c8, 4c75a98, 4fd586f, 54a49f3, 62deed8, 65c01ff, 6738680, 904965e, a00883c, a43827e, a5e1796, a6e62a9, c169320, c6ca363, e8d66bb, fe99422, ffdb8ea
- `tests/`: 131 files (119 added); commits 01858ae, 061cf63, 0760fbd, 08a32d9, 09bc390, 0b78f4a, 1023ba2, 14d3042, 14f42d2, 1c7cb9d, 1da392b, 1e7833e, 2864c94, 31e59f9, 38fe854, 3ad578d, 3b4db1c, 3ca76d5, 3e0f05d, 3ec548b, 4304e4c, 465e3c8, 4998b31, 4b2c616, 4c75a98, 4fd586f, 5002994, 5373a82, 54a49f3, 54eed84, 591c658, 596310b, 597dd43, 5b3683b, 62deed8, 6340b59, 6356916, 649bd53, 6507152, 65c01ff, 6738680, 680438f, 6d72f6d, 6eb7be5, 766aeb7, 7b22ef9, 7c75d4e, 81e07f8, 831dbb3, 84f10d5, 8b97fe8, 8f37466, 904965e, 910e552, 9155a7e, a00883c, a0c2f17, a43827e, a5e1796, a6e62a9, a930859, aad6c70, b2840dc, b875ea3, b973981, c0453c8, c169320, c575279, c6ca363, c7934ef, ca96025, cfc8b20, d3284b9, d3c86dc, d40744c, d4ab315, d51552c, d80a141, dcfe950, e377552, e7095cf, e743928, e8d66bb, eb6afa7, ec37d19, efdd761, f4dee77, f9082a7, fbb5857, fbd8d45, fe99422, ffdb8ea
- `tests/fixtures/`: `library_names_72ebfd1.json` (A in 831dbb3), `sigs_main.json` (A in 3b4db1c)

Generated from `git log HEAD --author="Raouf"` at commit `008ef72` of `refactor/final-fixes` on 2026-11-02. Only the member's own commits are listed; later changes by others are not shown.
