# Reproducibility & Environment Guide

**Owner**: Member 6 (Abdulla Jamal)

## Environment Specification
- **Python**: 3.11+
- **Setup**: `pip install -e ".[dev]"`
- **Hardware Target**: CPU

## Reproduction Steps

```bash
# 1. Data Prep
python scripts/prepare_data.py --config configs/source_config.yaml

# 2. Training
python scripts/train_hmm.py --config configs/hmm_mode_a.yaml
python scripts/train_crf.py --config configs/crf_mode_a.yaml

# 3. Evaluation
python scripts/evaluate_models.py --predictions results/predictions/crf_mode_a.json

# 4. Demo Execution
python scripts/run_demo.py --model-path artifacts/models/crf_mode_a.joblib --config configs/crf_mode_a.yaml