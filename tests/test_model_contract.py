"""
Contract test verifying model integration and gold-free inference rules.
"""

import pytest
from pathlib import Path
from punct_restore.pipeline import ModelAdapter, RestorationPipeline


def test_model_adapter_raises_on_missing_files(tmp_path):
    missing_model = tmp_path / "missing.joblib"
    missing_config = tmp_path / "missing.yaml"
    
    with pytest.raises(FileNotFoundError):
        ModelAdapter(model_artifact_path=missing_model, config_path=missing_config)


def test_pipeline_gold_free_length_preservation(tmp_path):
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("mode: Mode A\nlabels: [O, COMMA, PERIOD, QUESTION]\n")
    
    dummy_model_file = tmp_path / "dummy.joblib"
    dummy_model_file.write_text("placeholder")

    adapter = ModelAdapter(model_artifact_path=dummy_model_file, config_path=config_file)
    pipeline = RestorationPipeline(adapter)

    sample_text = "hello everyone welcome to our presentation today"
    result = pipeline.process_text(sample_text)

    assert len(result["raw_tokens"]) == len(result["predicted_labels"])
    for token in result["raw_tokens"]:
        assert token.lower() in result["restored_text"].lower()