# HMM experiments and reproduction

Owner: Member 2. Use release `sponspeech-m1-v2` and policy `m1-v2`.
The workflow is implemented; real-data results remain pending until these
commands finish and pass verification. See [HMM notes](HMM_NOTES.md) for the math.

## Environment and inputs

Use CPython 3.13.5 with [requirements-hmm.txt](../requirements-hmm.txt).
The estimator otherwise uses the standard library. No GPU or pretrained weights
are required. Data setup and licensing: [data/README.md](../data/README.md).
Run from the repository root. On Linux/macOS use `tmp/hmm-env/bin/python`.

```powershell
python -m venv tmp/hmm-env
tmp/hmm-env/Scripts/python.exe -m pip install -r requirements-hmm.txt
git fetch origin feat/m5-evaluation
git worktree add --detach tmp/member2-evaluation-3044a58 3044a58e777200bcac64e6e3364922404be359bf
```

Member 5's evaluator is pinned in a separate checkout. It is used directly,
without replacing it or merging its branch. A/B select non-O punctuation macro F1;
C selects masked, all-12-class joint macro F1. Both follow the pinned scorer.

## Commands

Commit reviewed code/configuration before tuning. The output directory must be new.
[hmm_sponspeech_v2.yaml](../configs/hmm_sponspeech_v2.yaml) freezes the search:
smoothing 0.01/0.1/1.0 and rare-word cutoff 1/2/5, in A/B/C. Ties in the scorer's
six-decimal validation score prefer lower cutoff, then lower smoothing.

```powershell
tmp/hmm-env/Scripts/python.exe scripts/run_hmm_experiments.py tune --config configs/hmm_sponspeech_v2.yaml --output-dir experiments/runs/hmm-sponspeech-v2-20261029 --evaluator-root tmp/member2-evaluation-3044a58
tmp/hmm-env/Scripts/python.exe scripts/run_hmm_experiments.py finalize --run-dir experiments/runs/hmm-sponspeech-v2-20261029 --evaluator-root tmp/member2-evaluation-3044a58
tmp/hmm-env/Scripts/python.exe scripts/verify_hmm_experiments.py --run-dir experiments/runs/hmm-sponspeech-v2-20261029 --evaluator-root tmp/member2-evaluation-3044a58 --output-dir experiments/runs/hmm-sponspeech-v2-20261029/verification
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

## Saved-model use and handoff

Training requires `--train-ids`; `--release-manifest` verifies official release
inputs. `predict_hmm.py` loads an artifact without training. Use `--all-records`
or an explicit `--ids` selection, and `--selection-record` to bind the artifact
to its frozen winner. Inspect each script's `--help`.

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
