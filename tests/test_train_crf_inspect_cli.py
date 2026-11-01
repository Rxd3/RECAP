"""Tests for the train-crf `inspect` subcommand (punct_restore.cli.train_crf)."""

from __future__ import annotations

from pathlib import Path

import pytest
from crf_cli_fixtures import create_synthetic_train_jsonl, write_id_file

import punct_restore.cli.train_crf as train_crf


class TestCRFInspectCLI:
    """Tests for train_crf.py inspect subcommand."""

    def test_inspect_valid_artifact(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify inspect prints expected metadata sections for a valid model."""
        data_file = tmp_path / "data.jsonl"
        ids_file = tmp_path / "train.ids"
        model_file = tmp_path / "inspect_model.joblib"

        all_ids = create_synthetic_train_jsonl(data_file, mode="A", num_samples=3)
        write_id_file(ids_file, all_ids)

        train_crf.main([
            "train",
            "-m", "A",
            "-i", str(data_file),
            "--ids", str(ids_file),
            "-o", str(model_file),
            "--max-iterations", "10",
        ])

        ret = train_crf.main(["inspect", "-m", str(model_file)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "RECAP CRF Model Artifact" in captured.out
        assert "Label Mode:         A" in captured.out
        assert "Hyperparameters:" in captured.out
        assert "Feature Configuration:" in captured.out
        assert "Companion Manifest:" in captured.out
        assert "Code Fingerprint:" in captured.out
        assert "Fingerprinted Files:" in captured.out

    def test_inspect_missing_file_failure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify inspect fails gracefully on missing artifact."""
        ret = train_crf.main(["inspect", "-m", str(tmp_path / "deep" / "ghost.joblib")])
        assert ret == 1
        captured = capsys.readouterr()
        assert captured.err == "Error: Artifact not found at ghost.joblib\n"
