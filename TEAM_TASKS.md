# Six-member project board

**Initial Scaffolding / Work Not Yet Implemented.** Replace Member placeholders
with names later. No implementation checkboxes are complete and no deadlines are
assumed. Every member must understand their component's assumptions, the label
schemes, leakage risks, evaluation choices and the overall methodology.

Primary ownership identifies the first reviewer, not permission to change shared
contracts alone. Discuss shared changes before coding, then obtain affected
owners' review. See [architecture](docs/ARCHITECTURE.md) and
[decision register](docs/DECISIONS.md).

## Member 1 — Dataset and Preprocessing Mutasim  member 1

**Branch:** `feat/m1-data-preprocessing`

**Primary ownership:** `src/punct_restore/data/`, `scripts/prepare_data.py`,
`data/README.md`, `docs/DATASET_PLAN.md`, `docs/DATA_CONTRACT.md`, dataset settings,
and `tests/test_data_contract.py`. Label policy changes require Member 5 review.

**Tasks:**

- [ ] Inspect source configurations, fields, revision, access, annotations and license.
- [ ] Agree and document cleaning, tokenization, lowercasing and edge-case policies.
- [ ] Extract punctuation boundaries and casing from original English text.
- [ ] Prepare aligned A (4), B (8) and C (4×3=12) representations without pipeline copies.
- [ ] Verify the proposed B→A mapping against benchmark rules with Member 5.
- [ ] Audit duplicate English text across source configs/years and talk/document groups.
- [ ] Freeze reproducible training/validation/test assignments and checksums.
- [ ] Calculate actual data/length statistics, exclusions and class distributions.
- [ ] Validate every release against the shared contract and add tiny fixture tests.
- [ ] Verify clean/ASR availability and alignment without labeling synthetic noise as ASR.

**Deliverables:** working preprocessing pipeline; reproducible processed-data
release and split manifest; annotation policy; provenance/download guide;
validation and leakage reports; label-distribution report and data tests.

**Dependencies:** team decisions D1–D6; Member 5 for evaluation compatibility;
Member 6 for portable handoff and artifact distribution. No model is required.

**Definition of Done:**

- [ ] Approved policies cover ambiguous punctuation/casing and rejected records.
- [ ] IDs, lengths, labels and splits validate; no known cross-split duplicate leakage.
- [ ] Regeneration command, source revision, environment and hashes are recorded.
- [ ] Members 2, 3, 5 and 6 can load the same release without retokenizing/resplitting.

**Handoff:** first submit source/policy inspection PR, then a data pipeline and
contract-test PR. Provide versioned data/manifests separately from Git. Notify
Members 2–6 of release ID, paths, checksums, distributions and known limitations.

## Member 2 — HMM Baseline Raşit  

**Branch:** `feat/m2-hmm-baseline`

**Primary ownership:** `src/punct_restore/models/hmm.py`, `scripts/train_hmm.py`,
`experiments/HMM_NOTES.md`; HMM-specific tests added later.

**Tasks:**

- [ ] Specify HMM assumptions, states per mode and sequence start/end handling.
- [ ] Estimate initial, transition and emission probabilities from training only.
- [ ] Implement numerically stable Viterbi decoding and an unseen-word strategy.
- [ ] Select smoothing/settings on shared validation data and record them.
- [ ] Support agreed label modes through configuration and the common model interface.
- [ ] Train on the shared release; save reproducible metadata and artifacts.
- [ ] Export predictions preserving IDs, order and one label per token.
- [ ] Document baseline limitations and observed behavior after actual experiments.

**Deliverables:** trained HMM baseline; reproducible training command/config;
standard prediction bundle; model-specific notes and meaningful tests.

**Dependencies:** Member 1's approved policies and release; Member 5's scoring
protocol; Member 6's artifact/inference adapter contract. Interface planning can
begin on illustrative tiny fixtures while the release is pending.

**Definition of Done:**

- [ ] Tiny hand-checkable sequences verify probability/Viterbi and OOV behavior.
- [ ] Model contract holds, no gold leakage occurs, and save/load preserves predictions.
- [ ] Artifact, data/mode/configuration hashes and actual commands are recorded.
- [ ] Member 5 validates prediction alignment and computes the baseline scores.

**Handoff:** give Members 3, 5 and 6 the frozen artifact/prediction bundle,
selected configuration, runtime conditions and limitations; do not write scores
before evaluation runs.

## Member 3 — CRF Main Model Omar

**Branch:** `feat/m3-crf-model`

**Primary ownership:** `src/punct_restore/models/crf.py`, `scripts/train_crf.py`,
`experiments/CRF_NOTES.md`; CRF-specific tests added later. The basic preset in
Member 4's feature module is a coordinated contribution, not overlapping ownership.

**Tasks:**

- [ ] Agree the injected feature interface with Member 4 before implementing it.
- [ ] Choose a linear-chain CRF backend and document assumptions/optimizer settings.
- [ ] Define a basic token-feature preset with Member 4 through one reviewed PR.
- [ ] Implement fit/predict/save/load under the common contract.
- [ ] Support configurable A/B/C targets with explicit artifact compatibility.
- [ ] Train and select settings on the fixed training/validation release.
- [ ] Export standard predictions and complete configuration/run records.
- [ ] Compare against the HMM within the same mode using Member 5's evaluation.

**Deliverables:** working basic CRF; reproducible training procedure;
common-format prediction/artifact bundles; experiment records and contract tests.

**Dependencies:** Member 1's data; agreement with Member 4 on feature injection;
Member 5's metrics; Member 2's evaluated baseline for later comparison; Member 6
for integration. CRF implementation need not wait for HMM training.

**Definition of Done:**

- [ ] Feature/token alignment and model persistence pass meaningful tiny tests.
- [ ] One CRF implementation supports both basic and improved feature injection.
- [ ] No gold casing/punctuation or test-derived information enters features.
- [ ] Selected validation settings and actual evidence are reproducible.

**Handoff:** give Member 4 the stable CRF adapter/extractor signature and basic
preset; give Members 5 and 6 compatible artifacts, predictions and provenance.

## Member 4 — Feature Engineering and Model Improvement Mohamed Kamaleldin

**Branch:** `feat/m4-feature-engineering`

**Primary ownership:** `src/punct_restore/features/`, `scripts/run_ablations.py`,
`experiments/FEATURE_NOTES.md`; feature-specific tests added later.

**Tasks:**

- [ ] Agree a feature interface with Member 3 and version its configuration.
- [ ] Develop improved context-window, neighbor, prefix and suffix features.
- [ ] Investigate question-detection patterns and observable capitalization cues.
- [ ] Explicitly account for lowercased input; never use hidden gold casing.
- [ ] Compare basic versus improved features through Member 3's CRF implementation.
- [ ] Design separate punctuation/casing versus joint experiments with Members 3 and 5.
- [ ] Run controlled one-change ablations with fixed splits and selection rules.
- [ ] Document useful, ineffective and uncertain features based on actual evidence.

**Deliverables:** improved extractors; reproducible ablation configuration and
run records; feature comparison tables from real evaluation; technical explanation.

**Dependencies:** Member 3's agreed adapter and basic preset; Member 1's input
policy; Member 5's compatible metrics; Member 6's serialization/inference needs.
Do not independently rewrite or replace the CRF model.

**Definition of Done:**

- [ ] Extractors return one feature dictionary per input token with no gold access.
- [ ] Basic and improved variants share model/data/evaluation code.
- [ ] Ablations change declared factors only and select on validation.
- [ ] Gains or negative findings have reproducible evidence and support counts.

**Handoff:** provide Member 3 the versioned extractor/configuration, Member 5 the
paired run bundles and hypothesis list, and Member 6 feature-loading requirements.

## Member 5 — Evaluation and Error Analysis Abd Raouf

**Branch:** `feat/m5-evaluation`

**Primary ownership:** `src/punct_restore/evaluation/`,
`scripts/evaluate_models.py`, `docs/EVALUATION_PROTOCOL.md`, `results/`,
`tests/test_evaluation_contract.py`.

**Tasks:**

- [ ] Agree complete vocabularies, target projections and split checks with Member 1.
- [ ] Validate prediction IDs, mode, token alignment and provenance before scoring.
- [ ] Implement token precision/recall/F1 and class-wise support/predicted counts.
- [ ] Report non-O macro F1 as primary; label inclusion/exclusion of O everywhere.
- [ ] Use zero_division=0; keep accuracy secondary and absent-class support visible.
- [ ] Evaluate C jointly and after punctuation/casing projections; compare tasks fairly.
- [ ] Produce confusion matrices, rare-label/class-imbalance and sequence error analysis.
- [ ] Compare runtime with common hardware/input/measurement conditions.
- [ ] Support group-aware paired confidence intervals if practical and documented.
- [ ] Reserve validation for selection and test for final reporting; never fabricate results.

**Deliverables:** evaluation pipeline; verified comparison tables/visualizations;
confusion matrices; error-analysis report; runtime comparison and optional intervals.

**Dependencies:** Member 1's immutable release and alignment policy; Members 2–4
prediction bundles; Member 6's manifest/runtime conventions. Metric definitions
and tiny fixture tests can be prepared before trained models exist.

**Definition of Done:**

- [ ] Hand-checkable metric fixtures cover O mistakes, absent classes and projections.
- [ ] Incompatible/misaligned predictions fail explicitly; no examples are silently lost.
- [ ] Every score identifies data, mode, support, averaging and O policy.
- [ ] Reported numbers/figures derive from real runs and traceable artifacts.

**Handoff:** provide Member 6 verified tables/figures with source hashes,
representative errors, selection/test separation and limitations; return failure
cases to Members 1–4 for analysis without silently changing the frozen test.

## Member 6 — Integration, Reproducibility and Presentation Abdulla Jamal

**Branch:** `feat/m6-integration-presentation`

**Primary ownership:** `src/punct_restore/pipeline.py`, `src/punct_restore/demo/`,
`scripts/run_demo.py`, `docs/REPRODUCIBILITY.md`,
`docs/PRESENTATION_OUTLINE.md`, `presentation/`, integration docs and
`tests/test_model_contract.py` with model-owner review.

**Tasks:**

- [ ] Coordinate interface versions, compatibility checks and integration PR order.
- [ ] Define CLI prediction inputs/flags and artifact-loading/model selection adapters.
- [ ] Route identical gold-free token sequences to each trained model.
- [ ] Integrate approved punctuation/casing reconstruction and error handling.
- [ ] Build a command-line demonstration; add interactive UI only if later agreed.
- [ ] Document exact preparation, training, prediction and evaluation commands as built.
- [ ] Record dependency locks, seeds, code/data/artifact hashes and hardware needs.
- [ ] Run a tiny end-to-end reproduction in a clean environment.
- [ ] Collect verified evidence and limitations from every member for the presentation.
- [ ] Rehearse the demo and verify the final integrated release with all owners.

**Deliverables:** integrated inference workflow and demo; reproducibility runbook;
integration/contract checks; final presentation backed by real findings.

**Dependencies:** all five members' contracts and versioned handoffs; Member 1's
decoding policies; Members 2–3 model persistence; Member 4 feature compatibility;
Member 5 verified outputs. CLI/manifest planning starts before models are trained.

**Definition of Done:**

- [ ] Compatible artifacts load and incompatible ones fail with useful messages.
- [ ] Shared inputs, IDs and output alignment survive every adapter; no gold leakage.
- [ ] Documented commands reproduce a small complete workflow from a clean setup.
- [ ] Demo and slides accurately distinguish trained behavior, targets and limitations.

**Handoff:** publish a reviewed integration/reproduction checklist for the team,
collect sign-off on each technical claim, and prepare the final demonstration.
This is a technical integration role, not only slide preparation.

## Shared milestones and integration checkpoints

- [ ] M0 — Everyone reads the scope/contracts and checks out their member branch.
- [ ] M1 — Member 1 source/policy PR and Member 5 metric/projection review accepted;
  Members 3–4 feature-interface PR and Member 6 artifact/CLI plan accepted.
- [ ] M2 — Member 1 data release validated; all consumers agree identical IDs/splits.
- [ ] M3 — HMM and basic CRF contract tests pass; Member 5 scores validation bundles;
  Member 6 verifies shared-input inference and save/load compatibility.
- [ ] M4 — Improved CRF and ablations reviewed on fixed validation settings.
- [ ] M5 — Freeze settings; Member 5 runs final held-out evaluation and error analysis.
- [ ] M6 — Member 6 verifies clean-environment reproduction, demo and presentation;
  all members review assumptions and evidence before the final release.

No dates are assigned. HMM and CRF can develop in parallel after interface
agreement; meaningful model comparison depends on the data release and common
evaluation. Improved features depend on the CRF interface, not a replacement CRF.

## Decisions requiring team agreement

Track [D1–D12](docs/DECISIONS.md): sources/license/download; tokenization and
lowercasing; punctuation/casing ambiguities; B→A mapping; leakage and splits;
feature injection; HMM/CRF settings; separate versus joint casing; real ASR versus
synthetic noise/alignment; artifacts/CLI/environment; uncertainty/runtime policy.

## Final project readiness

- [ ] Every member can explain their assumptions and the shared methodology.
- [ ] All data sources/policies, labels and split manifests are verified/versioned.
- [ ] All models use common contracts, frozen splits and gold-free inference.
- [ ] Results and projections are compatible, supported and reproducible.
- [ ] O policy, zero division, class imbalance and limitations are reported.
- [ ] Test data was held out for final reporting; selection used validation.
- [ ] Models/predictions are shared as checksummed artifacts, not large Git files.
- [ ] No credentials or personal absolute paths are committed.
- [ ] Integration tests, documented workflow and demonstration work on a clean setup.
- [ ] Presentation claims/figures trace to real experiments and all owners review them.
