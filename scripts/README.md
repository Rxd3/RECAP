# Task scripts

`train_hmm.py` is implemented for shared-contract JSONL and supports `--help`
without data. It estimates an HMM, writes a versioned JSON artifact/run manifest,
and optionally exports aligned predictions with hashes. It does not download,
preprocess, score or choose settings. No real-data run has occurred.
See [HMM notes](../experiments/HMM_NOTES.md) for flags and future commands.

The other five task scripts remain scaffold placeholders and exit nonzero with
an unfinished message. Run commands from the repository root.

| Script | Owner | Future responsibility |
| --- | --- | --- |
| `prepare_data.py` | Member 1 | Verified sources → validated JSONL and fixed splits |
| `train_hmm.py` | Member 2 | Shared training records → HMM artifact/predictions |
| `train_crf.py` | Member 3 | Shared records + injected features → CRF artifact/predictions |
| `run_ablations.py` | Member 4 | Controlled basic/improved feature comparisons |
| `evaluate_models.py` | Member 5 | Validate alignment; score compatible targets and analyze errors |
| `run_demo.py` | Member 6 | Shared input + compatible trained model → restored text |

TODO: Members define remaining argument interfaces with Member 6 as implementations
become available. Keep core logic in package modules.
