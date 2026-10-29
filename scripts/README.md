# Task scripts

`prepare_data.py` implements pinned SponSpeech transcript extraction and validated,
immutable release building. Run `python scripts/prepare_data.py --help` and follow
[data/README.md](../data/README.md).

`train_hmm.py` implements shared-schema HMM training, versioned JSON artifacts/run
manifests and optional aligned prediction exports. `--train-ids` is required;
`--release-manifest` authenticates release inputs, IDs and the label configuration.
The experiment runner selects settings using the pinned Member 5 scorer. Run
`python scripts/train_hmm.py --help` and see [HMM notes](../experiments/HMM_NOTES.md).

The remaining four scripts are placeholders. Run commands from the repository root.

| Script | Owner | Responsibility |
| --- | --- | --- |
| `prepare_data.py` | Member 1 | Verified sources → validated JSONL and fixed splits |
| `train_hmm.py` | Member 2 | Shared training records → HMM artifact/predictions |
| `predict_hmm.py` | Member 2 | Saved artifact → gold-free predictions |
| `run_hmm_experiments.py` | Member 2 | Frozen validation search, held-out evaluation and timing |
| `verify_hmm_experiments.py` | Member 2 | Independent score audit and full refit/prediction reproduction |
| `package_hmm_handoff.py` | Member 2 | Verified study → checksummed, portable model/prediction handoff ZIP |
| `train_crf.py` | Member 3 | Shared records + injected features → CRF artifact/predictions |
| `run_ablations.py` | Member 4 | Controlled basic/improved feature comparisons |
| `evaluate_models.py` | Member 5 | Validate alignment; score compatible targets and analyze errors |
| `run_demo.py` | Member 6 | Shared input + compatible trained model → restored text |

Other members define argument interfaces and actual commands with Member 6 as
implementations become available. Keep core logic in package modules.
