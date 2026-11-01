# HMM experiments and reproduction

Owner: Member 2. Use release `sponspeech-m1-v2` and policy `m1-v2`.
The real-data study completed and passed independent verification on 2026-10-29.
See [results](HMM_RESULTS.md) for measurements and [HMM notes](HMM_NOTES.md)
for the math. Training implementation commit:
`e4c31834799bb261e78218c67dd85e148ad8d493`.

## Environment and inputs

Use CPython 3.13.5 with [requirements-hmm.txt](../requirements-hmm.txt).
The estimator otherwise uses the standard library. No GPU or pretrained weights
are required. Data setup and licensing: [data/README.md](../data/README.md).
Run from the repository root. On Linux/macOS use `tmp/hmm-env/bin/python` and `tmp/hmm-env/bin/recap`.

```powershell
python -m venv tmp/hmm-env
tmp/hmm-env/Scripts/python.exe -m pip install -r requirements-hmm.txt
tmp/hmm-env/Scripts/python.exe -m pip install --no-deps -e .
git fetch origin feat/m5-evaluation
git worktree add --detach tmp/member2-evaluation-3044a58 3044a58e777200bcac64e6e3364922404be359bf
```

The editable install without dependencies adds the `recap` command (and makes the `scripts/` stubs
work) without the CRF packages; the HMM commands need only the standard library and PyYAML.

**Studies tuned on older code.** `plan.json` records the SHA-256 of every hashed HMM source file. On
2026-11-01 the list changed twice: once when the code moved into the package, and again when it became
the study's whole import closure (30 files, `models/hmm_experiment_identity.SOURCE_FILES`). Finalize,
verify and package all pass `verified_frozen_selection` first, so they refuse any study tuned before the
current list ("HMM implementation changed since selection"), by design. That includes the delivered
`hmm-sponspeech-v2-20261029`. A new study runs all four steps on this code. Finalize, verify and package a
study with the code it was tuned on: for the delivered study, a worktree of `main` at `72ebfd1` that holds
its own copy of the data and the study. The old scripts read the data through their own checkout and
write the package under its `artifacts/`. From the repository root:

```powershell
git worktree add --detach tmp/recap-72ebfd1 72ebfd1
git worktree add --detach tmp/recap-72ebfd1/tmp/member2-evaluation-3044a58 3044a58e777200bcac64e6e3364922404be359bf
Copy-Item -Recurse data/processed tmp/recap-72ebfd1/data/processed
Copy-Item -Recurse data/splits tmp/recap-72ebfd1/data/splits
New-Item -ItemType Directory -Force tmp/recap-72ebfd1/experiments/runs | Out-Null
Copy-Item -Recurse experiments/runs/hmm-sponspeech-v2-20261029 tmp/recap-72ebfd1/experiments/runs/
Set-Location tmp/recap-72ebfd1
../hmm-env/Scripts/python.exe scripts/run_hmm_experiments.py finalize --run-dir experiments/runs/hmm-sponspeech-v2-20261029 --evaluator-root tmp/member2-evaluation-3044a58
../hmm-env/Scripts/python.exe scripts/verify_hmm_experiments.py --run-dir experiments/runs/hmm-sponspeech-v2-20261029 --evaluator-root tmp/member2-evaluation-3044a58 --output-dir experiments/runs/hmm-sponspeech-v2-20261029/verification
../hmm-env/Scripts/python.exe scripts/package_hmm_handoff.py --run-dir experiments/runs/hmm-sponspeech-v2-20261029 --evaluator-root tmp/member2-evaluation-3044a58 --output-dir artifacts/hmm/hmm-sponspeech-v2-20261029
Set-Location ../..
```

At `72ebfd1` the files in `scripts/` are the original scripts, not stubs. Each one puts its own checkout's
`src` first on the import path, so it runs the `72ebfd1` code even though `tmp/hmm-env` has this
checkout installed. Skip a step whose output already exists; output directories must be new. Copy the
results back (`experiments/runs/...`, `artifacts/hmm/...`), then remove the evaluator worktree and the
outer one: `git worktree remove --force tmp/recap-72ebfd1/tmp/member2-evaluation-3044a58`, then
`git worktree remove --force tmp/recap-72ebfd1`. On Linux/macOS use `cp -r` and `../hmm-env/bin/python`.

Member 5's evaluator is pinned in a separate checkout. It is used directly,
without replacing it or merging its branch. A/B select non-O punctuation macro F1;
C selects masked, all-12-class joint macro F1. Both follow the pinned scorer.

## Commands

Commit reviewed code/configuration before tuning. The output directory must be new.
[hmm_sponspeech_v2.yaml](../configs/hmm_sponspeech_v2.yaml) freezes the search:
smoothing 0.01/0.1/1.0 and rare-word cutoff 1/2/5, in A/B/C. Ties in the scorer's
six-decimal validation score prefer lower cutoff, then lower smoothing.

```powershell
tmp/hmm-env/Scripts/recap.exe run-hmm-experiments tune --config configs/hmm_sponspeech_v2.yaml --output-dir experiments/runs/hmm-sponspeech-v2-20261029 --evaluator-root tmp/member2-evaluation-3044a58
tmp/hmm-env/Scripts/recap.exe run-hmm-experiments finalize --run-dir experiments/runs/hmm-sponspeech-v2-20261029 --evaluator-root tmp/member2-evaluation-3044a58
tmp/hmm-env/Scripts/recap.exe verify-hmm-experiments --run-dir experiments/runs/hmm-sponspeech-v2-20261029 --evaluator-root tmp/member2-evaluation-3044a58 --output-dir experiments/runs/hmm-sponspeech-v2-20261029/verification
```

Tuning opens train/validation records only. All mode selections freeze before test
access. Finalization uses saved winners without training on validation; it predicts
full validation/test/test_amb files and scores the primary IDs. Test and test_amb
remain separate. Shortcut baselines use identical primary IDs. Existing outputs
are never overwritten.

The run contains all candidate models, validation reports, frozen selections,
final predictions/sidecars, shared scores/confusions and actual commands/logs.
Timing uses identical mode-C validation IDs/tokens across modes, one model thread,
200 warmup sequences and three in-memory prediction repetitions. It excludes
loading, disk IO and scoring. A separate pass records per-sequence decode latency.
Hardware/environment are recorded.

Verification independently recounts scored positions and metrics, checks every
candidate's primary score, matches training counts to approved subsets, checks
normalization/gold invariance, refits each winner, and compares model bytes and all
full-split predictions. Frozen metadata is retained during the independent refit
for byte comparison. Use a new verification directory when repeating the audit.

The completed run used the equivalent isolated environment at
`tmp/member2-repro-env/Scripts/python.exe`; its raw commands and package versions
are preserved in the run. Selection froze at `2026-10-29T13:30:02.885568+00:00`,
before test access at `2026-10-29T13:30:41.053263+00:00`. Do not overwrite the
completed study or use its test outcomes to retune settings.

## Saved-model use and handoff

Training requires `--train-ids`; `--release-manifest` verifies official release
inputs. `recap predict-hmm` loads an artifact without training. Use `--all-records`
or an explicit `--ids` selection, and `--selection-record` to bind the artifact
to its frozen winner. Inspect each script's `--help`.

Package a verified run after committing the packaging code and report:

```powershell
tmp/hmm-env/Scripts/recap.exe package-hmm-handoff --run-dir experiments/runs/hmm-sponspeech-v2-20261029 --evaluator-root tmp/member2-evaluation-3044a58 --output-dir artifacts/hmm/hmm-sponspeech-v2-20261029
```

The command requires matching independent verification and checks frozen model,
candidate and report identities. It creates a ZIP, an outer `.zip.sha256`, and
an internal `SHA256SUMS`, then verifies every archived file by SHA-256. Existing
handoff paths are immutable. `MODEL_INDEX.json` identifies each selected model
by relative path and hash. Load it with `HMMModel.load_model(path)`; predict on
shared gold-free records. The raw dataset is distributed separately as the
Member 1 release. The handoff's `HANDOFF.md` explains use by Members 3, 5 and 6.

The repository is public; local packaging does not publish model files. The ZIP
is on the team Drive (`Models/hmm-sponspeech-v2-20261029.zip`). Consumer acknowledgments
are not recorded; the demo (`recap run-demo`) loads a selected HMM JSON.

Give Members 3, 5 and 6 selected models, configurations/selections, prediction
JSONL/sidecars, reports, timing/environment, verification evidence, license and
checksums. Generated artifacts stay outside Git and are handed to the project
lead as a versioned archive. Individual member acknowledgments are separate from
software verification.

A/B punctuation and C joint scores describe different tasks; do not rank those
numbers as equivalent. C punctuation/casing projections and non-final/non-initial
diagnostics are in the shared reports. Source overlap, annotation approximations
and utterance-boundary shortcuts remain documented in
[DATASET_PLAN.md](../docs/DATASET_PLAN.md).
