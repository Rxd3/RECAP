# Future task scripts

Every task script is an explicit scaffold placeholder. Executing one currently
exits nonzero with an unfinished message; no flags, training, data processing,
evaluation or inference are implemented. Run future commands from the repo root.

| Script | Owner | Future responsibility |
| --- | --- | --- |
| `prepare_data.py` | Member 1 | Verified sources → validated JSONL and fixed splits |
| `train_hmm.py` | Member 2 | Shared training records → HMM artifact/predictions |
| `train_crf.py` | Member 3 | Shared records + injected features → CRF artifact/predictions |
| `run_ablations.py` | Member 4 | Controlled basic/improved feature comparisons |
| `evaluate_models.py` | Member 5 | Validate alignment; score compatible targets and analyze errors |
| `run_demo.py` | Member 6 | Shared input + compatible trained model → restored text |

TODO: Members define argument interfaces and actual commands with Member 6 as
implementations become available. Keep core logic in package modules.
