# RECAP

**Restoring English Casing and Punctuation**

**COE025 — Natural Language Processing | Group Project 2 — Sequence Labeling**

**Status: Initial Scaffolding / Work Not Yet Implemented.** No data pipeline,
trained model, inference command, or evaluation calculation is implemented yet.

## Overview

Restore punctuation and capitalization in unformatted English text, including a
future evaluation on speech-recognition transcripts. Each word receives a label
for the punctuation boundary **after** it and a casing label for the word itself.

Repository: [Rxd3/RECAP](https://github.com/Rxd3/RECAP) (private).

Project 1, [NALTRA](https://github.com/Rxd3/NALTRA/), classified whole documents.
Project 2 predicts aligned token sequences in a separate repository. We retain
its `src/` layout, YAML configuration, script entry points, common evaluation
contracts, ownership documentation, and pull-request workflow. See the
[reference review](docs/REFERENCE_REVIEW.md) for what was inspected.

## Illustrative task

Input:

```text
hello everyone welcome to our presentation today we will discuss natural language processing
```

Target:

```text
Hello everyone, welcome to our presentation. Today we will discuss natural language processing.
```

This is an illustrative target, not a model prediction. Punctuation-only modes
do not themselves predict capitalization; the full example requires casing.

## Planned label configurations

| Mode | Target | Labels |
| --- | --- | --- |
| A | Standard punctuation | O, COMMA, PERIOD, QUESTION (4) |
| B | Extended punctuation | Mode A plus EXCLAMATION, COLON, SEMICOLON, DASH (8) |
| C | Joint standard punctuation and casing | Mode A × LOWER, FIRST_UPPER, ALL_UPPER (12) |

Mode C uses **4 × 3 = 12**, never 8 × 3. Its label form is
`PERIOD|FIRST_UPPER`: punctuation after this word, casing of this word.
[Label configuration](configs/label_sets.yaml) is the vocabulary source of truth;
[data rules](docs/DATA_CONTRACT.md) define alignment and proposed projections.
Ambiguous annotation policies remain open in [decisions](docs/DECISIONS.md).

## Proposed datasets

- [IWSLT/iwslt2017](https://huggingface.co/datasets/IWSLT/iwslt2017): proposed
  primary English text source for deriving extended punctuation and casing.
- [cdminix/iwslt2011](https://huggingface.co/datasets/cdminix/iwslt2011): proposed
  standard-punctuation reference and potential ASR evaluation source.

Neither source has been downloaded or approved for this project. Exact source
revision, English configuration/fields, access method, license conditions,
annotation conventions, and compatible splits need verification. IWSLT 2017 is
not assumed to supply ready-made eight-class annotations. See the
[dataset plan](docs/DATASET_PLAN.md) and [storage guide](data/README.md).

## Planned models and evaluation

1. HMM with Viterbi decoding: baseline.
2. Linear-chain CRF with basic features: main model.
3. The same CRF with improved features: controlled improvement and ablations.

Future architectures such as BiLSTM-CRF may implement the same
[model contract](docs/MODEL_CONTRACT.md); neural work is outside this scaffold.
All models will use the same versioned records and fixed splits.

Primary reporting will include non-O punctuation macro F1, per-class precision,
recall and F1, confusion matrices, class support, and latency. Accuracy is
secondary. Mode C also requires joint and projected punctuation/casing scores.
Modes B and C have different tasks; compare only justified compatible targets.
Validation selects settings; test data is reserved for final reporting. The
[evaluation protocol](docs/EVALUATION_PROTOCOL.md) specifies denominators and
zero-division policy. No results exist yet.

## Repository structure

```text
configs/                 Label vocabularies, source planning, experiment template
data/{raw,processed,splits}/  Local data and frozen split manifests (ignored)
src/punct_restore/
  data/                  Member 1: cleaning and label derivation
  models/                Members 2–3: HMM and CRF placeholders
  features/              Member 4: shared CRF feature interface
  evaluation/            Member 5: scoring and analysis placeholders
  demo/                  Member 6: inference demonstration placeholder
  pipeline.py            Member 6: integration placeholder
scripts/                 Future command entry points; currently stop explicitly
experiments/             Model notes and future run-record instructions
results/{predictions,tables,figures}/  Empty output locations
artifacts/models/        Empty, ignored model storage
tests/                   Contract test plans; no executable tests yet
docs/                    Scope, contracts, decisions, workflow, reproduction
presentation/assets/    Presentation outline and future evidence
```

## Team responsibilities

| Member | Responsibility | Branch |
| --- | --- | --- |
| 1 | Dataset and preprocessing | `feat/m1-data-preprocessing` |
| 2 | HMM baseline | `feat/m2-hmm-baseline` |
| 3 | CRF main model | `feat/m3-crf-model` |
| 4 | Feature engineering and improvement | `feat/m4-feature-engineering` |
| 5 | Evaluation and error analysis | `feat/m5-evaluation` |
| 6 | Integration, reproducibility and presentation | `feat/m6-integration-presentation` |

[TEAM_TASKS.md](TEAM_TASKS.md) is the detailed board, including deliverables,
dependencies, acceptance criteria, shared milestones, and handoffs.

## Installation and environment preparation

Use Python 3.11+ from the repository root. Python 3.13 was available during
scaffolding; experiment dependency compatibility has not yet been established.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On macOS/Linux activate with `source .venv/bin/activate`. Alternatively,
`python -m pip install -r requirements.txt` installs the same development extra.
These are preparation instructions; setup did not install packages. Runtime
dependencies are intentionally empty. NumPy, pandas, scikit-learn,
sklearn-crfsuite, Datasets, and matplotlib are candidates to review when needed.
PyTorch is a future extension only. No GPU framework or pretrained model is
required. [Reproducibility notes](docs/REPRODUCIBILITY.md) describe future locking.

The six task scripts are explicit placeholders that exit with an explanatory
message. Do not expect training, prediction, or `--help` to work yet.
The test modules contain TODO plans only; a pytest run currently collects no
tests. Structural verification is recorded in [initial verification](docs/INITIAL_VERIFICATION.md).

## Git collaboration

Use `main` for reviewed integration and each member's branch for development.
All six member branches start at the same scaffolding commit. Open focused PRs
into `main`; discuss shared interface changes with affected owners first.
Never force-push another member's work. The Project 2 remote is
`https://github.com/Rxd3/RECAP.git`; `main` and all six member branches are
published there. Team members need collaborator access to this private repository.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the daily checklist and
[Git workflow](docs/GIT_WORKFLOW.md) for clone, branch, commit, push, PR, and
conflict commands using the Project 2 remote.

## Roadmap and current status

- [x] Create package structure, configuration templates, ownership and contracts.
- [x] Prepare independent local Git history and six member branches.
- [ ] Agree annotation policies, source configuration, and evaluation projections.
- [ ] Release reproducible data and immutable split manifests.
- [ ] Implement and validate the HMM baseline and basic CRF.
- [ ] Implement improved CRF features and controlled ablations.
- [ ] Evaluate frozen settings on held-out data; analyze errors and runtime.
- [ ] Integrate inference, rehearse the demo, and prepare evidence-based slides.

No deadlines, dataset counts, model scores, or finished experiments are assumed.
Record seeds, source revisions, preprocessing versions, split hashes, code
commits, environments, and commands for every future run.
