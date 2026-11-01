"""Tests for the train-crf `train` subcommand and the provenance it records (punct_restore.cli.train_crf).

Owned by Member 3 (Omar Alzyod).
Covers:
- Valid synthetic Mode A, B, and C training.
- Frozen training ID filtering and cardinality enforcement.
- Duplicate, missing, corrupt, and misaligned input rejections.
- Split protection (rejects non-train splits).
- Mode mismatch protection.
- Hyperparameter validation.
- Non-overwriting safety for models and manifests.
- The code fingerprint, the git status line and the recorded command.
- Paths with spaces and Windows compatibility.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from crf_cli_fixtures import create_synthetic_train_jsonl, write_id_file

import punct_restore.cli.train_crf as train_crf
from punct_restore.hashing import sha256_file
from punct_restore.models import crf_training
from punct_restore.models.crf import CRFModel


class TestCRFTrainCLI:
    """Tests for train_crf.py training subcommand."""

    def test_train_mode_a_success(self, tmp_path: Path) -> None:
        """Verify successful training on synthetic Mode A data with frozen IDs."""
        data_file = tmp_path / "data_a.jsonl"
        ids_file = tmp_path / "train_a.ids"
        model_file = tmp_path / "crf_a.joblib"
        manifest_file = tmp_path / "crf_a.manifest.json"

        all_ids = create_synthetic_train_jsonl(data_file, mode="A", num_samples=6)
        train_ids = all_ids[:4]
        write_id_file(ids_file, train_ids)

        ret = train_crf.main([
            "train",
            "--mode", "A",
            "--input", str(data_file),
            "--ids", str(ids_file),
            "--output", str(model_file),
            "--manifest-out", str(manifest_file),
            "--preset", "basic",
            "--max-iterations", "20",
            "--run-id", "test_mode_a",
        ])

        assert ret == 0
        assert model_file.is_file()
        assert manifest_file.is_file()

        # Verify model artifact metadata
        model = CRFModel.load_model(model_file)
        assert model.is_fitted
        assert model.label_mode == "A"
        assert model.vocabulary == ["O", "COMMA", "PERIOD", "QUESTION"]
        assert model.model_name == "crf_basic"

        # Verify manifest contents
        with open(manifest_file, encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["schema_version"] == 1
        assert manifest["label_mode"] == "A"
        assert manifest["model_name"] == "crf_basic"
        assert manifest["run_id"] == "test_mode_a"
        assert manifest["vocabulary"] == ["O", "COMMA", "PERIOD", "QUESTION"]
        assert manifest["feature_preset"] == "basic"
        assert manifest["provenance"]["selected_sequences"] == 4
        assert manifest["artifact"]["sha256"] == sha256_file(model_file)
        assert manifest["backend"]["name"] == "sklearn-crfsuite (python-crfsuite)"
        assert "training_command" in manifest
        assert "environment" in manifest
        assert "code_fingerprint" in manifest
        assert len(manifest["code_fingerprint"]) == 64
        assert "fingerprinted_files" in manifest
        assert "scripts/train_crf.py" in manifest["fingerprinted_files"]
        assert "src/punct_restore/cli/train_crf.py" in manifest["fingerprinted_files"]
        assert "configs/label_sets.yaml" in manifest["fingerprinted_files"]
        assert "pyproject.toml" in manifest["fingerprinted_files"]

    def test_train_mode_b_success(self, tmp_path: Path) -> None:
        """Verify successful training on Mode B data with 8-class vocabulary."""
        data_file = tmp_path / "data_b.jsonl"
        ids_file = tmp_path / "train_b.ids"
        model_file = tmp_path / "crf_b.joblib"

        all_ids = create_synthetic_train_jsonl(data_file, mode="B", num_samples=5)
        write_id_file(ids_file, all_ids)

        ret = train_crf.main([
            "train",
            "--mode", "B",
            "--input", str(data_file),
            "--ids", str(ids_file),
            "--output", str(model_file),
            "--preset", "improved",
            "--max-iterations", "15",
        ])

        assert ret == 0
        model = CRFModel.load_model(model_file)
        assert model.label_mode == "B"
        assert model.model_name == "crf_improved"
        assert len(model.vocabulary) == 8
        assert "EXCLAMATION" in model.vocabulary

    def test_train_mode_c_success(self, tmp_path: Path) -> None:
        """Verify successful training on Mode C data with 12 joint classes."""
        data_file = tmp_path / "data_c.jsonl"
        ids_file = tmp_path / "train_c.ids"
        model_file = tmp_path / "crf_c.joblib"

        all_ids = create_synthetic_train_jsonl(data_file, mode="C", num_samples=5)
        write_id_file(ids_file, all_ids)

        ret = train_crf.main([
            "train",
            "--mode", "C",
            "--input", str(data_file),
            "--ids", str(ids_file),
            "--output", str(model_file),
            "--preset", "basic",
            "--max-iterations", "15",
        ])

        assert ret == 0
        model = CRFModel.load_model(model_file)
        assert model.label_mode == "C"
        assert model.model_name == "crf_basic"
        assert len(model.vocabulary) == 12
        assert "O|LOWER" in model.vocabulary

    def test_train_explicit_model_name_override(self, tmp_path: Path) -> None:
        """Verify explicit --model-name overrides the default crf_<preset> name."""
        data_file = tmp_path / "data.jsonl"
        ids_file = tmp_path / "train.ids"
        model_file = tmp_path / "custom_model.joblib"

        all_ids = create_synthetic_train_jsonl(data_file, mode="A", num_samples=3)
        write_id_file(ids_file, all_ids)

        ret = train_crf.main([
            "train",
            "-m", "A",
            "-i", str(data_file),
            "--ids", str(ids_file),
            "-o", str(model_file),
            "--preset", "basic",
            "--model-name", "my_custom_crf",
            "--max-iterations", "10",
        ])
        assert ret == 0
        model = CRFModel.load_model(model_file)
        assert model.model_name == "my_custom_crf"

    def test_train_frozen_ids_excludes_unselected_records(self, tmp_path: Path) -> None:
        """Verify that records not present in the frozen ID list are excluded from training."""
        data_file = tmp_path / "data.jsonl"
        ids_file = tmp_path / "train.ids"
        model_file = tmp_path / "model.joblib"
        manifest_file = tmp_path / "model.manifest.json"

        all_ids = create_synthetic_train_jsonl(data_file, mode="A", num_samples=7)
        # Select strictly 3 records
        selected_ids = [all_ids[0], all_ids[2], all_ids[4]]
        write_id_file(ids_file, selected_ids)

        ret = train_crf.main([
            "train",
            "-m", "A",
            "-i", str(data_file),
            "--ids", str(ids_file),
            "-o", str(model_file),
            "--manifest-out", str(manifest_file),
            "--max-iterations", "10",
        ])
        assert ret == 0

        with open(manifest_file, encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["provenance"]["selected_sequences"] == 3

    def test_train_missing_id_list_failure(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify explicit failure when the specified ID file does not exist."""
        data_file = tmp_path / "data.jsonl"
        create_synthetic_train_jsonl(data_file, mode="A", num_samples=2)

        ret = train_crf.main([
            "train",
            "-m", "A",
            "-i", str(data_file),
            "--ids", str(tmp_path / "non_existent.ids"),
            "-o", str(tmp_path / "out.joblib"),
        ])
        assert ret == 1
        captured = capsys.readouterr()
        assert "non_existent.ids: No such file or directory" in captured.err and str(tmp_path) not in captured.err

    def test_train_missing_selected_id_in_input_failure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify explicit failure if an ID from the frozen list is absent in the input file."""
        data_file = tmp_path / "data.jsonl"
        ids_file = tmp_path / "train.ids"
        all_ids = create_synthetic_train_jsonl(data_file, mode="A", num_samples=3)

        write_id_file(ids_file, all_ids + ["ghost_sample_9999"])

        ret = train_crf.main([
            "train",
            "-m", "A",
            "-i", str(data_file),
            "--ids", str(ids_file),
            "-o", str(tmp_path / "out.joblib"),
        ])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Missing 1 selected IDs" in captured.err

    def test_train_duplicate_id_in_frozen_list_failure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify failure when the frozen ID file contains duplicate entries."""
        data_file = tmp_path / "data.jsonl"
        ids_file = tmp_path / "train.ids"
        create_synthetic_train_jsonl(data_file, mode="A", num_samples=2)

        with open(ids_file, "w", encoding="utf-8") as f:
            f.write("sample_0000\nsample_0001\nsample_0000\n")

        ret = train_crf.main([
            "train",
            "-m", "A",
            "-i", str(data_file),
            "--ids", str(ids_file),
            "-o", str(tmp_path / "out.joblib"),
        ])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Duplicate ID 'sample_0000'" in captured.err

    def test_train_empty_frozen_ids_failure(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify failure when the frozen ID file is empty."""
        data_file = tmp_path / "data.jsonl"
        ids_file = tmp_path / "train.ids"
        create_synthetic_train_jsonl(data_file, mode="A", num_samples=2)
        write_id_file(ids_file, [])

        ret = train_crf.main([
            "train",
            "-m", "A",
            "-i", str(data_file),
            "--ids", str(ids_file),
            "-o", str(tmp_path / "out.joblib"),
        ])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Frozen ID list is empty" in captured.err

    def test_train_rejects_non_train_split(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify that records with split != 'train' cannot enter training."""
        data_file = tmp_path / "data.jsonl"
        ids_file = tmp_path / "train.ids"

        # Create records marked with split='val'
        all_ids = create_synthetic_train_jsonl(data_file, mode="A", num_samples=3, split="val")
        write_id_file(ids_file, all_ids)

        ret = train_crf.main([
            "train",
            "-m", "A",
            "-i", str(data_file),
            "--ids", str(ids_file),
            "-o", str(tmp_path / "out.joblib"),
        ])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Non-train records cannot enter training" in captured.err

    def test_train_rejects_mode_mismatch(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify that training fails when record label_mode does not match CLI --mode."""
        data_file = tmp_path / "data.jsonl"
        ids_file = tmp_path / "train.ids"

        # Mode B records in file
        all_ids = create_synthetic_train_jsonl(data_file, mode="B", num_samples=3)
        write_id_file(ids_file, all_ids)

        # Attempt to train as Mode A
        ret = train_crf.main([
            "train",
            "-m", "A",
            "-i", str(data_file),
            "--ids", str(ids_file),
            "-o", str(tmp_path / "out.joblib"),
        ])
        assert ret == 1
        captured = capsys.readouterr()
        assert "expected 'A'" in captured.err

    def test_train_token_label_length_mismatch(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify failure when token and label lengths differ."""
        data_file = tmp_path / "corrupt_data.jsonl"
        ids_file = tmp_path / "train.ids"

        rec = {
            "sample_id": "seq_misaligned",
            "split": "train",
            "label_mode": "A",
            "tokens": ["word1", "word2", "word3"],
            "punct_labels": ["O", "PERIOD"],  # 2 labels for 3 tokens
        }
        with open(data_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        write_id_file(ids_file, ["seq_misaligned"])

        ret = train_crf.main([
            "train",
            "-m", "A",
            "-i", str(data_file),
            "--ids", str(ids_file),
            "-o", str(tmp_path / "out.joblib"),
        ])
        assert ret == 1
        captured = capsys.readouterr()
        assert "mismatched punct_labels length" in captured.err

    def test_train_corrupt_jsonl_failure(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify failure when encountering non-JSON lines."""
        data_file = tmp_path / "corrupt.jsonl"
        ids_file = tmp_path / "train.ids"

        with open(data_file, "w", encoding="utf-8") as f:
            f.write(
                '{"sample_id": "valid", "split": "train", "label_mode": "A", '
                '"tokens": ["hi"], "punct_labels": ["O"]}\n'
            )
            f.write('{this is NOT json\n')

        write_id_file(ids_file, ["valid"])

        ret = train_crf.main([
            "train",
            "-m", "A",
            "-i", str(data_file),
            "--ids", str(ids_file),
            "-o", str(tmp_path / "out.joblib"),
        ])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Corrupt JSON" in captured.err

    def test_train_invalid_hyperparameters(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify rejection of invalid hyperparameter values."""
        data_file = tmp_path / "data.jsonl"
        ids_file = tmp_path / "train.ids"
        all_ids = create_synthetic_train_jsonl(data_file, mode="A", num_samples=3)
        write_id_file(ids_file, all_ids)

        ret = train_crf.main([
            "train",
            "-m", "A",
            "-i", str(data_file),
            "--ids", str(ids_file),
            "-o", str(tmp_path / "out.joblib"),
            "--c1", "-1.0",
        ])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Configuration error" in captured.err or "non-negative" in captured.err

    def test_train_non_overwriting_protection(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify that existing model artifact and manifest are protected from silent overwrite."""
        data_file = tmp_path / "data.jsonl"
        ids_file = tmp_path / "train.ids"
        model_file = tmp_path / "model.joblib"

        all_ids = create_synthetic_train_jsonl(data_file, mode="A", num_samples=3)
        write_id_file(ids_file, all_ids)

        # First train succeeds
        ret1 = train_crf.main([
            "train",
            "-m", "A",
            "-i", str(data_file),
            "--ids", str(ids_file),
            "-o", str(model_file),
            "--max-iterations", "10",
        ])
        assert ret1 == 0

        # Second train without --overwrite fails
        ret2 = train_crf.main([
            "train",
            "-m", "A",
            "-i", str(data_file),
            "--ids", str(ids_file),
            "-o", str(model_file),
            "--max-iterations", "10",
        ])
        assert ret2 == 1
        captured = capsys.readouterr()
        assert "Output artifact already exists" in captured.err

        # With --overwrite succeeds
        ret3 = train_crf.main([
            "train",
            "-m", "A",
            "-i", str(data_file),
            "--ids", str(ids_file),
            "-o", str(model_file),
            "--max-iterations", "10",
            "--overwrite",
        ])
        assert ret3 == 0

    def test_train_paths_with_spaces_and_windows_compatibility(self, tmp_path: Path) -> None:
        """Verify CLI compatibility with paths containing spaces."""
        spaced_dir = tmp_path / "path with spaces" / "sub dir"
        spaced_dir.mkdir(parents=True)

        data_file = spaced_dir / "my data.jsonl"
        ids_file = spaced_dir / "my ids.ids"
        model_file = spaced_dir / "my model.joblib"

        all_ids = create_synthetic_train_jsonl(data_file, mode="A", num_samples=3)
        write_id_file(ids_file, all_ids)

        ret = train_crf.main([
            "train",
            "-m", "A",
            "-i", str(data_file),
            "--ids", str(ids_file),
            "-o", str(model_file),
            "--max-iterations", "10",
        ])
        assert ret == 0
        assert model_file.is_file()



class TestTrainProvenance:
    """The code fingerprint, the git status line and the training manifest."""

    def test_code_fingerprint_deterministic_and_comprehensive(self) -> None:
        """Verify code fingerprint includes entry point, configs, dependencies, and package files."""
        fp1 = crf_training.compute_code_fingerprint()
        fp2 = crf_training.compute_code_fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 64

        files = crf_training.get_fingerprinted_files()
        assert files == sorted(files)
        assert "scripts/train_crf.py" in files
        assert "src/punct_restore/cli/train_crf.py" in files
        assert "configs/label_sets.yaml" in files
        assert "pyproject.toml" in files
        assert "src/punct_restore/models/crf.py" in files
        assert "src/punct_restore/features/crf_features.py" in files

        # Verify all paths are POSIX-normalized and relative
        for f_path in files:
            assert not Path(f_path).is_absolute()
            assert "\\" not in f_path

    def test_git_status_porcelain_unknown_without_git(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def no_git(*_args: Any, **_kwargs: Any) -> None:
            raise FileNotFoundError("git")

        monkeypatch.setattr(crf_training.subprocess, "run", no_git)
        assert crf_training.git_status_porcelain() == "unknown"

        failed = subprocess.CompletedProcess(["git"], 128, stdout="", stderr="fatal")
        monkeypatch.setattr(crf_training.subprocess, "run", lambda *_a, **_k: failed)
        assert crf_training.git_status_porcelain() == "unknown"

        done = subprocess.CompletedProcess(["git"], 0, stdout=" M x.py\n", stderr="")
        monkeypatch.setattr(crf_training.subprocess, "run", lambda *_a, **_k: done)
        assert crf_training.git_status_porcelain() == " M x.py\n"

    def test_train_manifest_has_no_hardcoded_warning(self, tmp_path: Path) -> None:
        data, ids = tmp_path / "t.jsonl", tmp_path / "t.ids"
        write_id_file(ids, create_synthetic_train_jsonl(data, mode="A", num_samples=3))
        man = tmp_path / "m.man.json"
        ret = train_crf.main(["train", "-m", "A", "-i", str(data), "--ids", str(ids),
                              "-o", str(tmp_path / "m.joblib"), "--manifest-out", str(man),
                              "--max-iterations", "10"])
        assert ret == 0
        warning = json.loads(man.read_text(encoding="utf-8"))["reproducibility_warning"]
        assert "Local uncommitted modifications" not in warning


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["C:/Users/u/RECAP/.venv/Scripts/recap.exe", "train-crf", "train"], ["recap", "train-crf", "train"]),
        (["/home/u/RECAP/.venv/bin/recap", "train-crf", "train"], ["recap", "train-crf", "train"]),
        (["scripts/train_crf.py", "train", "--mode", "A"], ["scripts/train_crf.py", "train", "--mode", "A"]),
        (["-m", "train"], ["-m", "train"]),
        ([], []),
    ],
)
def test_the_recorded_command_never_holds_the_launcher_path(argv: list[str], expected: list[str]) -> None:
    assert train_crf.recorded_command(argv) == expected
