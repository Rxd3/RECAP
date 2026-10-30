# Six-member project board

**Status (2026-10-30):** the data release and the HMM baseline are verified
([release validation](docs/DATA_RELEASE_VALIDATION.md), [HMM results](experiments/HMM_RESULTS.md)).
The six CRF models are trained and scored once on test and test_amb
([CRF results](docs/EVALUATION_RESULTS_CRF.md)). The course-baseline settings (HMM smoothing, CRF settings, the plain threshold τ) were fixed or chosen on validation before any test result. The comparison methods in sections 13–14 of EVALUATION_RESULTS_CRF.md (simulated ASR noise, gap classifier, hidden-event LM, prior-adjusted decoding, question lookback, factored casing) were added on 2026-10-30, after the first test report, in response to a review that cited test results. They are exploratory; their settings were still chosen on validation only, and each test list was predicted once. A ticked item
names its evidence in the repository; an unticked one is not done or has no such evidence. Every
member must understand their component's assumptions, label schemes, leakage
risks, evaluation choices and the overall methodology.

Primary ownership identifies the first reviewer, not permission to change shared
contracts alone. Discuss shared changes before coding, then obtain affected
owners' review. See [architecture](docs/ARCHITECTURE.md) and
[decision register](docs/DECISIONS.md).

## Member 1 — Dataset and Preprocessing (Mutasim)

**Branch:** `feat/m1-data-preprocessing`

**Primary ownership:** `src/punct_restore/data/`, `scripts/prepare_data.py`,
`data/README.md`, `docs/DATASET_PLAN.md`, `docs/DATA_CONTRACT.md`, dataset settings,
and `tests/test_data_contract.py`. Label policy changes require Member 5 review.

**Tasks:**

- [x] Inspect source configurations, fields, revision, access, annotations and license. (evidence: [DATASET_PLAN.md](docs/DATASET_PLAN.md), [configs/datasets.yaml](configs/datasets.yaml))
- [x] Agree and document cleaning, tokenization, lowercasing and edge-case policies. (evidence: [DECISIONS.md](docs/DECISIONS.md) D2–D4, [DATASET_PLAN.md](docs/DATASET_PLAN.md))
- [x] Extract punctuation boundaries and casing from original English text. (evidence: [labeling.py](src/punct_restore/data/labeling.py), [tests/test_data_contract.py](tests/test_data_contract.py))
- [x] Prepare aligned A (4), B (8) and C (4×3=12) representations without pipeline copies. (evidence: [DATA_CONTRACT.md](docs/DATA_CONTRACT.md), [data/README.md](data/README.md))
- [x] Verify the approved B→A mapping and retain a separate native benchmark view. (evidence: [DATA_RELEASE_VALIDATION.md](docs/DATA_RELEASE_VALIDATION.md), [tests/test_data_contract.py](tests/test_data_contract.py))
- [x] Audit duplicate English text across source configs/years and talk/document groups. (evidence: [DATASET_PLAN.md](docs/DATASET_PLAN.md) leakage tables, [splits.py](src/punct_restore/data/splits.py))
- [x] Freeze reproducible training/validation/test assignments and checksums. (evidence: [DATA_RELEASE_VALIDATION.md](docs/DATA_RELEASE_VALIDATION.md) fingerprints)
- [x] Calculate actual data/length statistics, exclusions and class distributions. (evidence: [DATASET_PLAN.md](docs/DATASET_PLAN.md))
- [x] Validate every release against the shared contract and add tiny fixture tests. (evidence: [validation.py](src/punct_restore/data/validation.py), [tests/test_data_contract.py](tests/test_data_contract.py), [tests/test_data_release_regressions.py](tests/test_data_release_regressions.py))
- [x] Verify clean/ASR availability and alignment without labeling synthetic noise as ASR. (evidence: [DATASET_PLAN.md](docs/DATASET_PLAN.md) Source and provenance)

**Deliverables:** working preprocessing pipeline; reproducible processed-data
release and split manifest; annotation policy; provenance/download guide;
validation and leakage reports; label-distribution report and data tests.

**Dependencies:** team decisions D1–D6; Member 5 for evaluation compatibility;
Member 6 for portable handoff and artifact distribution. No model is required.

**Definition of Done:**

- [x] Approved policies cover ambiguous punctuation/casing and rejected records. (evidence: [DECISIONS.md](docs/DECISIONS.md) D3–D4, [DATASET_PLAN.md](docs/DATASET_PLAN.md))
- [x] IDs, lengths, labels and splits validate; primary subsets pass video, raw/normalized exact and six-gram overlap checks against their reference splits. (evidence: [DATA_RELEASE_VALIDATION.md](docs/DATA_RELEASE_VALIDATION.md))
- [x] Regeneration command, source revision, environment and hashes are recorded. (evidence: [data/README.md](data/README.md), [DATA_RELEASE_VALIDATION.md](docs/DATA_RELEASE_VALIDATION.md))
- [ ] Members 2, 3, 5 and 6 acknowledge the same distributed release without retokenizing/resplitting. — not done: no member acknowledgment is recorded

**Handoff:** [PR #1](https://github.com/Rxd3/RECAP/pull/1) contains the pipeline,
contracts and regression tests. The corrected `sponspeech-m1-v2` bundle is on the team Drive
(`dataset/sponspeech-m1-v2-handoff.zip`); the old `sponspeech-m1-v1-rev2` bundle is superseded.
Individual member acknowledgments are not recorded. The team lead authorized this review.

## Member 2 — HMM Baseline (Raşit)

**Branch:** `feat/m2-hmm-baseline`

**Primary ownership:** `src/punct_restore/models/hmm.py`, `scripts/train_hmm.py`,
`experiments/HMM_NOTES.md`; HMM-specific tests added later.

**Tasks:**

- [x] Specify HMM assumptions, states per mode and sequence start/end handling. (evidence: [HMM_NOTES.md](experiments/HMM_NOTES.md))
- [x] Implement initial, transition and emission estimation from training only; approved real-data counts verified. (evidence: [hmm.py](src/punct_restore/models/hmm.py), [tests/test_hmm.py](tests/test_hmm.py), [HMM_RESULTS.md](experiments/HMM_RESULTS.md))
- [x] Implement numerically stable Viterbi decoding and an unseen-word strategy. (evidence: [hmm.py](src/punct_restore/models/hmm.py), [tests/test_hmm.py](tests/test_hmm.py))
- [x] Select smoothing/settings on shared validation data and record them (27 candidates; all modes frozen before test). (evidence: [HMM_RESULTS.md](experiments/HMM_RESULTS.md), [configs/hmm_sponspeech_v2.yaml](configs/hmm_sponspeech_v2.yaml))
- [x] Support agreed label modes through configuration and the common model interface. (evidence: [tests/test_hmm.py](tests/test_hmm.py))
- [x] Train on the shared release; save reproducible metadata and artifacts (all three independent refits byte-identical). (evidence: [HMM_RESULTS.md](experiments/HMM_RESULTS.md))
- [x] Implement prediction export preserving IDs, order and one label per token; all full-split predictions reproduced. (evidence: [predict_hmm.py](scripts/predict_hmm.py), [tests/test_hmm_cli.py](tests/test_hmm_cli.py), [HMM_RESULTS.md](experiments/HMM_RESULTS.md))
- [x] Document baseline limitations and observed behavior after actual experiments. (evidence: [HMM_RESULTS.md](experiments/HMM_RESULTS.md), [EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md))

**Deliverables:** trained HMM baseline; reproducible training command/config;
standard prediction bundle; model-specific notes and meaningful tests.

**Dependencies:** Member 1's approved policies and release; Member 5's scoring
protocol; Member 6's artifact/inference adapter contract. The completed HMM follows
the shared record contract; integrated demo acceptance remains Member 6's work.

**Definition of Done:**

- [x] Tiny hand-checkable sequences verify probability/Viterbi and OOV behavior. (evidence: [tests/test_hmm.py](tests/test_hmm.py))
- [x] HMM input/output contract, gold-invariance and save/load hold on synthetic tests and real data; the demo loads the selected HMM JSON. (evidence: [tests/test_hmm.py](tests/test_hmm.py), [tests/test_hmm_workflow.py](tests/test_hmm_workflow.py), [HMM_RESULTS.md](experiments/HMM_RESULTS.md))
- [x] Artifact, data/mode/configuration hashes and actual commands are recorded. (evidence: [HMM_RUNBOOK.md](experiments/HMM_RUNBOOK.md), [HMM_RESULTS.md](experiments/HMM_RESULTS.md))
- [x] Member 5's pinned shared scorer validates prediction alignment and computes the baseline scores; an independent counter verifies them. (evidence: [HMM_RESULTS.md](experiments/HMM_RESULTS.md), [EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md))

**Handoff:** the leader receives the local, checksummed bundle
`artifacts/hmm/hmm-sponspeech-v2-20261029.zip`, containing selected models, all
candidate evidence, predictions, scores, configuration, runtime and reproduction
proof. See [results](experiments/HMM_RESULTS.md) and
[runbook](experiments/HMM_RUNBOOK.md). Distribute it through the team Drive folder
to Members 3, 5 and 6; it is on the team Drive (`Models/`). Individual acknowledgments are not
recorded; the integrated demo exists (`scripts/run_demo.py`, `tests/test_pipeline.py`). Running the shared scorer records technical
validation, not a human Member 5 sign-off.

## Member 3 — CRF Main Model (Omar)

**Branch:** `feat/m3-crf-model`

**Primary ownership:** `src/punct_restore/models/crf.py`, `scripts/train_crf.py`,
`experiments/CRF_NOTES.md`; CRF-specific tests added later. The basic preset in
Member 4's feature module is a coordinated contribution, not overlapping ownership.

**Tasks:**

- [ ] Agree the injected feature interface with Member 4 before implementing it. — not done: Member 4 has no commits; Member 3 wrote the interface alone (`aa232be`)
- [x] Choose a linear-chain CRF backend and document assumptions/optimizer settings. (evidence: [CRF_EXPERIMENTS.md](docs/CRF_EXPERIMENTS.md) §2, §6)
- [ ] Define a basic token-feature preset with Member 4 through one reviewed PR. — basic preset written by Member 3 alone in `aa232be`; no PR with Member 4
- [x] Implement fit/predict/save/load under the common contract. (evidence: [crf.py](src/punct_restore/models/crf.py), [tests/test_crf_model.py](tests/test_crf_model.py))
- [x] Support configurable A/B/C targets with explicit artifact compatibility. (evidence: [tests/test_train_crf_cli.py](tests/test_train_crf_cli.py) (modes A, B, C; mode mismatch rejected))
- [ ] Train and select settings on the fixed training/validation release. — six models trained ([CRF_EXPERIMENTS.md](docs/CRF_EXPERIMENTS.md) §7); settings not tuned
- [x] Export standard predictions and complete configuration/run records. (evidence: Member 3's training manifests per model; test and test_amb predictions with prediction manifests produced by Member 5 from Member 3's frozen models, [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) Provenance)
- [x] Compare against the HMM within the same mode using Member 5's evaluation. (evidence: [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §1–2 test, §5 validation)

**Deliverables:** working basic CRF; reproducible training procedure;
common-format prediction/artifact bundles; experiment records and contract tests.

**Dependencies:** Member 1's data; agreement with Member 4 on feature injection;
Member 5's metrics; Member 2's evaluated baseline for later comparison; Member 6
for integration. CRF implementation need not wait for HMM training.

**Definition of Done:**

- [x] Feature/token alignment and model persistence pass meaningful tiny tests. (evidence: [tests/test_crf_features.py](tests/test_crf_features.py), [tests/test_crf_model.py](tests/test_crf_model.py))
- [x] One CRF implementation supports both basic and improved feature injection. (evidence: [crf.py](src/punct_restore/models/crf.py), [CRF_EXPERIMENTS.md](docs/CRF_EXPERIMENTS.md) §3, §7)
- [x] No gold casing/punctuation or test-derived information enters features. (evidence: [tests/test_crf_features.py](tests/test_crf_features.py) `test_gold_free_integrity`)
- [ ] Selected validation settings and actual evidence are reproducible. — all six validation prediction files reproduced byte for byte from the delivered models ([EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) Provenance); settings were fixed, not selected

**Handoff:** give Member 4 the stable CRF adapter/extractor signature and basic
preset; give Members 5 and 6 compatible artifacts, predictions and provenance.

## Member 4 — Feature Engineering and Model Improvement (Mohamed Kamaleldin)

**Branch:** `feat/m4-feature-engineering`

**Primary ownership:** `src/punct_restore/features/`, `scripts/run_ablations.py`,
`experiments/FEATURE_NOTES.md`; feature-specific tests added later.

**Tasks:**

- [ ] Agree a feature interface with Member 3 and version its configuration. — implemented by Member 3 (see [contributions](contributions/Omar_STUDENT_ID_Devinci101.md))
- [ ] Develop improved context-window, neighbor, prefix and suffix features. — implemented by Member 3 (see [contributions](contributions/Omar_STUDENT_ID_Devinci101.md))
- [ ] Investigate question-detection patterns and observable capitalization cues. — implemented by Member 3 (see [contributions](contributions/Omar_STUDENT_ID_Devinci101.md))
- [ ] Explicitly account for lowercased input; never use hidden gold casing. — implemented by Member 3 (see [contributions](contributions/Omar_STUDENT_ID_Devinci101.md))
- [ ] Compare basic versus improved features through Member 3's CRF implementation. — done by Member 5 ([EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §1)
- [ ] Design separate punctuation/casing versus joint experiments with Members 3 and 5. — joint vs punctuation-only comparison done by Member 5 ([EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md), H1)
- [ ] Run controlled one-change ablations with fixed splits and selection rules. — done by Member 5 ([EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §6, validation)
- [ ] Document useful, ineffective and uncertain features based on actual evidence. — documented by Member 5 ([EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §6) and in [FEATURE_NOTES.md](experiments/FEATURE_NOTES.md)

**Deliverables:** improved extractors; reproducible ablation configuration and
run records; feature comparison tables from real evaluation; technical explanation.

**Dependencies:** Member 3's agreed adapter and basic preset; Member 1's input
policy; Member 5's compatible metrics; Member 6's serialization/inference needs.
Do not independently rewrite or replace the CRF model.

**Definition of Done:**

- [ ] Extractors return one feature dictionary per input token with no gold access. — implemented by Member 3 (see [contributions](contributions/Omar_STUDENT_ID_Devinci101.md))
- [ ] Basic and improved variants share model/data/evaluation code. — implemented by Member 3 (see [contributions](contributions/Omar_STUDENT_ID_Devinci101.md))
- [ ] Ablations change declared factors only and select on validation. — done by Member 5 ([EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §6)
- [ ] Gains or negative findings have reproducible evidence and support counts. — done by Member 5 ([EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §6 ablations, §4 class supports)

**Handoff:** provide Member 3 the versioned extractor/configuration, Member 5 the
paired run bundles and hypothesis list, and Member 6 feature-loading requirements.

## Member 5 — Evaluation and Error Analysis (Abdlaraouf Alkabir)

**Branch:** `feat/m5-evaluation`

**Primary ownership:** `src/punct_restore/evaluation/`,
`scripts/evaluate_models.py`, `docs/EVALUATION_PROTOCOL.md`, `results/`,
`tests/test_evaluation_*.py`.

**Tasks:**

- [x] Agree complete vocabularies, target projections and split checks with Member 1. (evidence: [DECISIONS.md](docs/DECISIONS.md) D3–D6 (Member 5 reviewer), [EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md))
- [x] Validate prediction IDs, mode, token alignment and provenance before scoring. (evidence: [inputs.py](src/punct_restore/evaluation/inputs.py), [manifest.py](src/punct_restore/evaluation/manifest.py), [tests/test_evaluation_inputs.py](tests/test_evaluation_inputs.py))
- [x] Implement token precision/recall/F1 and class-wise support/predicted counts. (evidence: [metrics.py](src/punct_restore/evaluation/metrics.py), [tests/test_evaluation_metrics.py](tests/test_evaluation_metrics.py))
- [x] Report non-O macro F1 as primary; label inclusion/exclusion of O everywhere. (evidence: [EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md), [EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md))
- [x] Use zero_division=0; keep accuracy secondary and absent-class support visible. (evidence: [EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md))
- [x] Evaluate C jointly and after punctuation/casing projections; compare tasks fairly. (evidence: [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §3, H1, [tests/test_evaluation_views.py](tests/test_evaluation_views.py))
- [x] Produce confusion matrices, rare-label/class-imbalance and sequence error analysis. (evidence: [EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md) §3–6, [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md))
- [x] Compare runtime with common hardware/input/measurement conditions. (evidence: HMM [EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md) §7; CRF [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §10; prediction speed on the same machine and procedure, CRF fit times from Member 3's machine)
- [x] Support group-aware paired confidence intervals if practical and documented. (evidence: [uncertainty.py](src/punct_restore/evaluation/uncertainty.py), [comparison.py](src/punct_restore/evaluation/comparison.py), [EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md))
- [x] Reserve validation for selection and test for final reporting; never fabricate results. (evidence: [HMM_RUNBOOK.md](experiments/HMM_RUNBOOK.md) (selection froze before test access), [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) (validation selection in §5; each frozen model predicted test once))

**Deliverables:** evaluation pipeline; verified comparison tables/visualizations;
confusion matrices; error-analysis report; runtime comparison and optional intervals.

**Dependencies:** Member 1's immutable release and alignment policy; Members 2–4
prediction bundles; Member 6's manifest/runtime conventions. Metric definitions
and tiny fixture tests can be prepared before trained models exist.

**Definition of Done:**

- [x] Hand-checkable metric fixtures cover O mistakes, absent classes and projections. (evidence: [tests/test_evaluation_metrics.py](tests/test_evaluation_metrics.py), [tests/test_evaluation_views.py](tests/test_evaluation_views.py))
- [x] Incompatible/misaligned predictions fail explicitly; no examples are silently lost. (evidence: [tests/test_evaluation_inputs.py](tests/test_evaluation_inputs.py))
- [x] Every score identifies data, mode, support, averaging and O policy. (evidence: [report.py](src/punct_restore/evaluation/report.py), [EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md))
- [x] Reported numbers/figures derive from real runs and traceable artifacts. (evidence: [EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md) Provenance, [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) Provenance)

**Handoff:** provide Member 6 verified tables/figures with source hashes,
representative errors, selection/test separation and limitations; return failure
cases to Members 1–4 for analysis without silently changing the frozen test.

## Member 6 — Integration, Reproducibility and Presentation (Abdulla Jamal)

**Branch:** `feat/m6-integration-presentation`

**Primary ownership:** `src/punct_restore/pipeline.py`,
`scripts/run_demo.py`, `docs/REPRODUCIBILITY.md`,
`docs/PRESENTATION_OUTLINE.md`, `presentation/`, integration docs and
`tests/test_model_contract.py` with model-owner review.

**Tasks:**

- [ ] Coordinate interface versions, compatibility checks and integration PR order. — not done by Member 6: Member 5 merged the member branches into `release/final` (097c87b, 0790467, 611f3f4, ddba805)
- [x] Define CLI prediction inputs/flags and artifact-loading/model selection adapters. (evidence: `load_model` in [pipeline.py](src/punct_restore/pipeline.py), [scripts/README.md](scripts/README.md), [tests/test_pipeline.py](tests/test_pipeline.py); completed during release integration)
- [x] Route identical gold-free token sequences to each trained model. (evidence: `ModelAdapter` in [pipeline.py](src/punct_restore/pipeline.py) sends the same gold-free record to an HMM or a CRF; [tests/test_model_contract.py](tests/test_model_contract.py), [tests/test_pipeline.py](tests/test_pipeline.py))
- [x] Integrate approved punctuation/casing reconstruction and error handling. (evidence: `RestorationPipeline` in [pipeline.py](src/punct_restore/pipeline.py), [tests/test_pipeline.py](tests/test_pipeline.py); completed during release integration)
- [x] Build a command-line demonstration; add interactive UI only if later agreed. (evidence: [scripts/run_demo.py](scripts/run_demo.py), [tests/test_pipeline.py](tests/test_pipeline.py); completed during release integration)
- [x] Document exact preparation, training, prediction and evaluation commands as built. (evidence: [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md))
- [x] Record dependency locks, seeds, code/data/artifact hashes and hardware needs. (evidence: exact pins in [pyproject.toml](pyproject.toml) checked by [tests/test_dependencies.py](tests/test_dependencies.py); seed 2026, data and model hashes and hardware in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md))
- [x] Run a tiny end-to-end reproduction in a clean environment. (evidence: the fixture suite in `tests/` passes in a fresh clone ([REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) §1), and the demo command there fits and runs a small HMM)
- [x] Collect verified evidence and limitations from every member for the presentation. (evidence: [PRESENTATION_OUTLINE.md](docs/PRESENTATION_OUTLINE.md))
- [ ] Rehearse the demo and verify the final integrated release with all owners. — not done: no rehearsal or owner sign-off is recorded

**Deliverables:** integrated inference workflow and demo; reproducibility runbook;
integration/contract checks; final presentation backed by real findings.

**Dependencies:** all five members' contracts and versioned handoffs; Member 1's
decoding policies; Members 2–3 model persistence; Member 4 feature compatibility;
Member 5 verified outputs. CLI/manifest planning starts before models are trained.

**Definition of Done:**

- [x] Compatible artifacts load and incompatible ones fail with useful messages. (evidence: `src/punct_restore/pipeline.py` `load_model`, [tests/test_model_contract.py](tests/test_model_contract.py); completed during release integration)
- [x] Shared inputs, IDs and output alignment survive every adapter; no gold leakage. (evidence: label-count and vocabulary checks in [pipeline.py](src/punct_restore/pipeline.py), [tests/test_pipeline.py](tests/test_pipeline.py); gold-field invariance in [tests/test_hmm.py](tests/test_hmm.py); gold-free CLI inputs in [prediction_io.py](src/punct_restore/prediction_io.py))
- [x] Documented commands reproduce a small complete workflow from a clean setup. (evidence: [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) §1 (fresh clone, tests) and §10 (fit a small HMM, run the demo))
- [ ] Demo and slides accurately distinguish trained behavior, targets and limitations. — not done: the slide deck is made after the repository freeze

**Handoff:** publish a reviewed integration/reproduction checklist for the team,
collect sign-off on each technical claim, and prepare the final demonstration.
This is a technical integration role, not only slide preparation.

## Shared milestones and integration checkpoints

- [ ] M0 — Everyone reads the scope/contracts and checks out their member branch. — not recorded: no evidence of each member's reading
- [ ] M1 — Member 1 source/policy PR and Member 5 metric/projection review accepted;
  Members 3–4 feature-interface PR and Member 6 artifact/CLI plan accepted. — not done: no Members 3–4 feature-interface PR (Member 4 has no commits)
- [ ] M2 — Member 1 data release validated; all consumers agree identical IDs/splits. — partly: the release is validated ([DATA_RELEASE_VALIDATION.md](docs/DATA_RELEASE_VALIDATION.md)); consumer agreement is not recorded
- [x] M3 — HMM and basic CRF contract tests pass; Member 5 scores validation bundles;
  Member 6 verifies shared-input inference and save/load compatibility. (evidence: [tests/test_model_contract.py](tests/test_model_contract.py), [tests/test_pipeline.py](tests/test_pipeline.py), [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §5; the shared-input part was completed during release integration)
- [x] M4 — Improved CRF and ablations reviewed on fixed validation settings. (evidence: [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) §5, §6)
- [x] M5 — Freeze settings; Member 5 runs final held-out evaluation and error analysis. (evidence: [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md), [EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md))
- [ ] M6 — Member 6 verifies clean-environment reproduction, demo and presentation;
  all members review assumptions and evidence before the final release. — partly: clean-clone tests and the demo are recorded; member review is not

No dates are assigned. HMM and CRF can develop in parallel after interface
agreement; meaningful model comparison depends on the data release and common
evaluation. Improved features depend on the CRF interface, not a replacement CRF.

## Decisions requiring team agreement

Track [D1–D12](docs/DECISIONS.md): sources/license/download; tokenization and
lowercasing; punctuation/casing ambiguities; B→A mapping; leakage and splits;
feature injection; HMM/CRF settings; separate versus joint casing; real ASR versus
synthetic noise/alignment; artifacts/CLI/environment; uncertainty/runtime policy.

## Final project readiness

- [ ] Every member can explain their assumptions and the shared methodology. — not recorded: no evidence in the repository
- [x] All data sources/policies, labels and split manifests are verified/versioned. (evidence: [DATA_RELEASE_VALIDATION.md](docs/DATA_RELEASE_VALIDATION.md), [DATASET_PLAN.md](docs/DATASET_PLAN.md))
- [x] All models use common contracts, frozen splits and gold-free inference. (evidence: [MODEL_CONTRACT.md](docs/MODEL_CONTRACT.md), [scripts/train_crf.py](scripts/train_crf.py) predict passes only sample_id and tokens, [HMM_RESULTS.md](experiments/HMM_RESULTS.md), [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) Provenance)
- [x] Results and projections are compatible, supported and reproducible. (evidence: evaluator JSON in `results/scores/`, table builders in `scripts/reporting/`, reproduction levels in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md))
- [x] O policy, zero division, class imbalance and limitations are reported. (evidence: [EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md), [EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md) §3 and Limitations)
- [x] Test data was held out for final reporting; selection used validation. (evidence: [HMM_RUNBOOK.md](experiments/HMM_RUNBOOK.md), [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md))
- [x] Models/predictions are shared as checksummed artifacts, not large Git files. (evidence: [HMM_RUNBOOK.md](experiments/HMM_RUNBOOK.md), [EVALUATION_RESULTS_CRF.md](docs/EVALUATION_RESULTS_CRF.md) Provenance, [.gitignore](.gitignore))
- [x] No credentials or personal absolute paths are committed. (evidence: `git grep` for absolute user paths finds none, 2026-10-30)
- [x] Integration tests, documented workflow and demonstration work on a clean setup. (evidence: [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) §1 (fresh clone, the full suite) and §10 (demo))
- [ ] Presentation claims/figures trace to real experiments and all owners review them. — partly: the outline's numbers trace to `results/scores/` ([PRESENTATION_OUTLINE.md](docs/PRESENTATION_OUTLINE.md)); owner review is not recorded

## Final state

Final state: the head of `release/final`, 2026-10-30. A ticked item names its evidence; every
unticked item gives its reason.
