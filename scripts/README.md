# Task scripts

`prepare_data.py` implements pinned SponSpeech transcript extraction and validated,
immutable release building. Run `python scripts/prepare_data.py --help` and follow
[data/README.md](../data/README.md).

`train_hmm.py` implements shared-schema HMM training, versioned JSON artifacts/run
manifests and optional aligned prediction exports. It does not score, choose settings
or apply the release ID lists automatically. Apply explicit filtering before full-data
training; `--split-manifest` only records a checksum. Run
`python scripts/train_hmm.py --help` and see [HMM notes](../experiments/HMM_NOTES.md).

The remaining four scripts are placeholders. Run commands from the repository root.

| Script | Owner | Responsibility |
| --- | --- | --- |
| `prepare_data.py` | Member 1 | Verified sources → validated JSONL and fixed splits |
| `train_hmm.py` | Member 2 | Shared training records → HMM artifact/predictions |
| `train_crf.py` | Member 3 | Shared records + injected features → CRF artifact/predictions |
| `run_ablations.py` | Member 4 | Controlled basic/improved feature comparisons |
| `evaluate_models.py` | Member 5 | Validate alignment; score compatible targets and analyze errors |
| `run_demo.py` | Member 6 | Shared input + compatible trained model → restored text |

Other members define argument interfaces and actual commands with Member 6 as
implementations become available. Keep core logic in package modules.
