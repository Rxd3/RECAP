# Task scripts

`prepare_data.py` implements pinned SponSpeech transcript extraction and validated,
immutable release building. Run `python scripts/prepare_data.py --help` and follow
[data/README.md](../data/README.md). Other scripts remain placeholders in the data
branch; their implementations develop on the respective member branches. Run
commands from the repository root.

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
