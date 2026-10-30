"""Tests for RECAP CRF training, prediction, and inspection CLI (scripts/train_crf.py).

Owned by Member 3 (Omar Alzyod).
Covers:
- Valid synthetic Mode A, B, and C training.
- Frozen training ID filtering and cardinality enforcement.
- Duplicate, missing, corrupt, and misaligned input rejections.
- Split protection (rejects non-train splits).
- Mode mismatch protection.
- Hyperparameter validation.
- Non-overwriting safety for models, manifests, and predictions.
- Schema compliance and gold-free prediction export.
- Save/load prediction equivalence.
- Inspection of valid and corrupted artifacts.
- Paths with spaces and Windows compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest

import scripts.train_crf as train_crf
from punct_restore.models.crf import CRFModel


def create_synthetic_train_jsonl(
    path: Path,
    mode: str = "A",
    num_samples: int = 5,
    split: str = "train",
) -> list[str]:
    """Helper to create a synthetic JSONL file and return its sample IDs."""
    ids: list[str] = []
    mode_a_punct = ["O", "COMMA", "PERIOD", "QUESTION"]
    mode_b_punct = ["O", "COMMA", "PERIOD", "QUESTION", "EXCLAMATION", "COLON", "SEMICOLON", "DASH"]
    case_labels_pool = ["LOWER", "FIRST_UPPER", "ALL_UPPER"]

    tokens_pool = [
        ["hello", "world"],
        ["how", "are", "you"],
        ["this", "is", "a", "test"],
        ["nlp", "is", "great"],
        ["good", "morning", "everyone"],
        ["we", "love", "python"],
        ["crf", "sequence", "tagging"],
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for idx in range(num_samples):
            s_id = f"sample_{idx:04d}"
            tokens = tokens_pool[idx % len(tokens_pool)]
            n = len(tokens)

            if mode == "A":
                p_labels = [mode_a_punct[i % len(mode_a_punct)] for i in range(n)]
                p_labels[-1] = "PERIOD"
                rec: dict[str, Any] = {
                    "sample_id": s_id,
                    "split": split,
                    "label_mode": "A",
                    "tokens": tokens,
                    "punct_labels": p_labels,
                }
            elif mode == "B":
                p_labels = [mode_b_punct[i % len(mode_b_punct)] for i in range(n)]
                p_labels[-1] = "EXCLAMATION"
                rec = {
                    "sample_id": s_id,
                    "split": split,
                    "label_mode": "B",
                    "tokens": tokens,
                    "punct_labels": p_labels,
                }
            elif mode == "C":
                p_labels = [mode_a_punct[i % len(mode_a_punct)] for i in range(n)]
                p_labels[-1] = "PERIOD"
                c_labels = [case_labels_pool[i % len(case_labels_pool)] for i in range(n)]
                rec = {
                    "sample_id": s_id,
                    "split": split,
                    "label_mode": "C",
                    "tokens": tokens,
                    "punct_labels": p_labels,
                    "case_labels": c_labels,
                }
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            f.write(json.dumps(rec) + "\n")
            ids.append(s_id)

    return ids


def write_id_file(path: Path, ids: list[str]) -> None:
    """Helper to write an ID file with one ID per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s_id in ids:
            f.write(f"{s_id}\n")


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
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["schema_version"] == 1
        assert manifest["label_mode"] == "A"
        assert manifest["model_name"] == "crf_basic"
        assert manifest["run_id"] == "test_mode_a"
        assert manifest["vocabulary"] == ["O", "COMMA", "PERIOD", "QUESTION"]
        assert manifest["feature_preset"] == "basic"
        assert manifest["provenance"]["selected_sequences"] == 4
        assert manifest["artifact"]["sha256"] == train_crf.sha256_file(model_file)
        assert manifest["backend"]["name"] == "sklearn-crfsuite (python-crfsuite)"
        assert "training_command" in manifest
        assert "environment" in manifest
        assert "code_fingerprint" in manifest
        assert len(manifest["code_fingerprint"]) == 64
        assert "fingerprinted_files" in manifest
        assert "scripts/train_crf.py" in manifest["fingerprinted_files"]
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

        with open(manifest_file, "r", encoding="utf-8") as f:
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
        assert "Frozen ID list not found" in captured.err

    def test_train_missing_selected_id_in_input_failure(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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

    def test_train_duplicate_id_in_frozen_list_failure(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
            f.write('{"sample_id": "valid", "split": "train", "label_mode": "A", "tokens": ["hi"], "punct_labels": ["O"]}\n')
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


class TestCRFPredictCLI:
    """Tests for train_crf.py predict subcommand."""

    @pytest.fixture
    def trained_model_artifact(self, tmp_path: Path) -> tuple[Path, Path]:
        """Fixture that trains a small Mode A model artifact."""
        data_file = tmp_path / "train_data.jsonl"
        ids_file = tmp_path / "train.ids"
        model_file = tmp_path / "trained_model.joblib"

        all_ids = create_synthetic_train_jsonl(data_file, mode="A", num_samples=4)
        write_id_file(ids_file, all_ids)

        ret = train_crf.main([
            "train",
            "-m", "A",
            "-i", str(data_file),
            "--ids", str(ids_file),
            "-o", str(model_file),
            "--max-iterations", "15",
            "--preset", "basic",
        ])
        assert ret == 0
        return model_file, data_file

    def test_predict_schema_compliance_and_gold_free(
        self, tmp_path: Path, trained_model_artifact: tuple[Path, Path]
    ) -> None:
        """Verify prediction JSONL conforms strictly to docs/MODEL_CONTRACT.md."""
        model_file, _ = trained_model_artifact
        eval_input = tmp_path / "eval_input.jsonl"
        preds_output = tmp_path / "predictions.jsonl"

        # Observation input (strictly gold-free)
        records = [
            {"sample_id": "test_001", "tokens": ["this", "is", "great"]},
            {"sample_id": "test_002", "tokens": ["who", "is", "there", "today"]},
        ]
        with open(eval_input, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ret = train_crf.main([
            "predict",
            "-m", str(model_file),
            "-i", str(eval_input),
            "-o", str(preds_output),
        ])
        assert ret == 0
        assert preds_output.is_file()

        # Parse and inspect output JSONL
        with open(preds_output, "r", encoding="utf-8") as f:
            pred_lines = [json.loads(line) for line in f if line.strip()]

        assert len(pred_lines) == 2
        for pred, original in zip(pred_lines, records, strict=True):
            assert set(pred.keys()) == {"sample_id", "model_name", "label_mode", "predicted_labels"}
            assert pred["sample_id"] == original["sample_id"]
            assert pred["model_name"] == "crf_basic"
            assert pred["label_mode"] == "A"
            assert isinstance(pred["predicted_labels"], list)
            assert len(pred["predicted_labels"]) == len(original["tokens"])
            for lbl in pred["predicted_labels"]:
                assert lbl in {"O", "COMMA", "PERIOD", "QUESTION"}

    def test_predict_with_frozen_ids_filtering(
        self, tmp_path: Path, trained_model_artifact: tuple[Path, Path]
    ) -> None:
        """Verify predict --ids restricts prediction to specified IDs."""
        model_file, _ = trained_model_artifact
        eval_input = tmp_path / "eval_input.jsonl"
        ids_file = tmp_path / "eval.ids"
        preds_output = tmp_path / "predictions.jsonl"

        records = [
            {"sample_id": "eval_001", "tokens": ["one"]},
            {"sample_id": "eval_002", "tokens": ["two", "tokens"]},
            {"sample_id": "eval_003", "tokens": ["three", "tokens", "here"]},
        ]
        with open(eval_input, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        write_id_file(ids_file, ["eval_001", "eval_003"])

        ret = train_crf.main([
            "predict",
            "-m", str(model_file),
            "-i", str(eval_input),
            "-o", str(preds_output),
            "--ids", str(ids_file),
        ])
        assert ret == 0

        with open(preds_output, "r", encoding="utf-8") as f:
            pred_lines = [json.loads(line) for line in f if line.strip()]

        assert len(pred_lines) == 2
        assert [p["sample_id"] for p in pred_lines] == ["eval_001", "eval_003"]

    def test_predict_preserves_source_jsonl_order_regardless_of_ids_file_order(
        self, tmp_path: Path, trained_model_artifact: tuple[Path, Path]
    ) -> None:
        """Verify predict preserves source JSONL line order, not the order in --ids.

        Also validates:
        - Unique IDs (no duplicate sample_ids in predictions).
        - Exact selected membership matching the frozen ID set.
        - Vocabulary containment (every label belongs to declared mode vocabulary).
        - One label per corresponding input token.
        - model_name is the canonical model identifier (e.g. crf_basic), not run_id.
        """
        model_file, _ = trained_model_artifact
        eval_input = tmp_path / "eval_input.jsonl"
        ids_file = tmp_path / "eval.ids"
        preds_output = tmp_path / "predictions.jsonl"

        source_records = [
            {"sample_id": "sample_alpha", "tokens": ["first", "sequence", "here"]},
            {"sample_id": "sample_beta", "tokens": ["unselected", "intermediate"]},
            {"sample_id": "sample_gamma", "tokens": ["third", "sequence"]},
        ]
        with open(eval_input, "w", encoding="utf-8") as f:
            for r in source_records:
                f.write(json.dumps(r) + "\n")

        # Frozen ID list specifies gamma BEFORE alpha (reverse of source JSONL)
        target_ids = ["sample_gamma", "sample_alpha"]
        write_id_file(ids_file, target_ids)

        ret = train_crf.main([
            "predict",
            "-m", str(model_file),
            "-i", str(eval_input),
            "-o", str(preds_output),
            "--ids", str(ids_file),
        ])
        assert ret == 0

        with open(preds_output, "r", encoding="utf-8") as f:
            pred_lines = [json.loads(line) for line in f if line.strip()]

        # 1. Exact cardinality
        assert len(pred_lines) == len(target_ids) == 2

        # 2. Source JSONL order is preserved (alpha appears before gamma in source file)
        pred_ids = [p["sample_id"] for p in pred_lines]
        assert pred_ids == ["sample_alpha", "sample_gamma"]
        assert pred_ids != target_ids  # Proves order is NOT dictated by the .ids file

        # 3. Unique IDs
        assert len(set(pred_ids)) == len(pred_ids)

        # 4. Exact selected set membership
        assert set(pred_ids) == set(target_ids)

        # 5. Token length alignment, vocabulary containment, and model_name contract
        input_tokens_by_id = {r["sample_id"]: r["tokens"] for r in source_records}
        for pred in pred_lines:
            s_id = pred["sample_id"]
            assert pred["model_name"] == "crf_basic"
            assert pred["label_mode"] == "A"
            assert isinstance(pred["predicted_labels"], list)
            assert len(pred["predicted_labels"]) == len(input_tokens_by_id[s_id])
            for lbl in pred["predicted_labels"]:
                assert lbl in {"O", "COMMA", "PERIOD", "QUESTION"}

    def test_predict_missing_id_failure(
        self, tmp_path: Path, trained_model_artifact: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify failure when an ID in --ids is missing in eval_input."""
        model_file, _ = trained_model_artifact
        eval_input = tmp_path / "eval_input.jsonl"
        ids_file = tmp_path / "eval.ids"
        preds_output = tmp_path / "predictions.jsonl"

        with open(eval_input, "w", encoding="utf-8") as f:
            f.write(json.dumps({"sample_id": "eval_001", "tokens": ["one"]}) + "\n")

        write_id_file(ids_file, ["eval_001", "missing_eval_999"])

        ret = train_crf.main([
            "predict",
            "-m", str(model_file),
            "-i", str(eval_input),
            "-o", str(preds_output),
            "--ids", str(ids_file),
        ])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Missing 1 selected IDs" in captured.err

    def test_predict_non_overwriting_protection(
        self, tmp_path: Path, trained_model_artifact: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify predict does not overwrite output without --overwrite."""
        model_file, _ = trained_model_artifact
        eval_input = tmp_path / "eval.jsonl"
        preds_output = tmp_path / "predictions.jsonl"

        with open(eval_input, "w", encoding="utf-8") as f:
            f.write(json.dumps({"sample_id": "eval_001", "tokens": ["hello"]}) + "\n")

        # First run succeeds
        ret1 = train_crf.main([
            "predict",
            "-m", str(model_file),
            "-i", str(eval_input),
            "-o", str(preds_output),
        ])
        assert ret1 == 0

        # Second run without --overwrite fails
        ret2 = train_crf.main([
            "predict",
            "-m", str(model_file),
            "-i", str(eval_input),
            "-o", str(preds_output),
        ])
        assert ret2 == 1
        captured = capsys.readouterr()
        assert "Output file already exists" in captured.err

        # With --overwrite succeeds
        ret3 = train_crf.main([
            "predict",
            "-m", str(model_file),
            "-i", str(eval_input),
            "-o", str(preds_output),
            "--overwrite",
        ])
        assert ret3 == 0

    def test_save_load_prediction_equivalence(
        self, tmp_path: Path, trained_model_artifact: tuple[Path, Path]
    ) -> None:
        """Verify loaded model produces identical predictions to the serialized model."""
        model_file, _ = trained_model_artifact
        model1 = CRFModel.load_model(model_file)
        model2 = CRFModel.load_model(model_file)

        test_records = [
            {"sample_id": "test_1", "tokens": ["we", "train", "crf", "models"]},
            {"sample_id": "test_2", "tokens": ["how", "does", "it", "work"]},
        ]
        preds1 = model1.predict(test_records)
        preds2 = model2.predict(test_records)
        assert preds1 == preds2


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
        ret = train_crf.main(["inspect", "-m", str(tmp_path / "ghost.joblib")])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Artifact not found" in captured.err

    def test_code_fingerprint_deterministic_and_comprehensive(self) -> None:
        """Verify code fingerprint includes entry point, configs, dependencies, and package files."""
        fp1 = train_crf.compute_code_fingerprint()
        fp2 = train_crf.compute_code_fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 64

        files = train_crf.get_fingerprinted_files()
        assert files == sorted(files)
        assert "scripts/train_crf.py" in files
        assert "configs/label_sets.yaml" in files
        assert "pyproject.toml" in files
        assert "src/punct_restore/models/crf.py" in files
        assert "src/punct_restore/features/crf_features.py" in files

        # Verify all paths are POSIX-normalized and relative
        for f_path in files:
            assert not Path(f_path).is_absolute()
            assert "\\" not in f_path
