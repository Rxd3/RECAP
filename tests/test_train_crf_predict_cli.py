"""Tests for the train-crf `predict` subcommand (punct_restore.cli.train_crf).

Owned by Member 3 (Omar Alzyod).
Covers:
- Schema compliance and gold-free prediction export.
- Frozen ID filtering and source-order preservation.
- Non-overwriting safety for predictions and their manifests.
- Save/load prediction equivalence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from crf_cli_fixtures import create_synthetic_train_jsonl, write_id_file

import punct_restore.cli.train_crf as train_crf
from punct_restore.hashing import sha256_file
from punct_restore.models.crf import CRFModel
from punct_restore.prediction_io import read_observations, write_predictions_jsonl


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
        with open(preds_output, encoding="utf-8") as f:
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

        with open(preds_output, encoding="utf-8") as f:
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

        with open(preds_output, encoding="utf-8") as f:
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


class TestPredictManifestAndProvenance:
    """Prediction sidecar manifest, reusable helpers and git-derived provenance."""

    @staticmethod
    def _train_mode_a(tmp_path: Path) -> Path:
        data, ids, model = tmp_path / "t.jsonl", tmp_path / "t.ids", tmp_path / "m.joblib"
        write_id_file(ids, create_synthetic_train_jsonl(data, mode="A", num_samples=4))
        ret = train_crf.main(["train", "-m", "A", "-i", str(data), "--ids", str(ids),
                              "-o", str(model), "--max-iterations", "10"])
        assert ret == 0
        return model

    @staticmethod
    def _write_gold(path: Path) -> list[str]:
        rows = [("gold_a", ["we", "love", "python"]), ("gold_b", ["how", "are", "you"])]
        with open(path, "w", encoding="utf-8") as f:
            for s_id, tokens in rows:
                n = len(tokens)
                rec = {"sample_id": s_id, "split": "test", "label_mode": "A", "tokens": tokens,
                       "punct_labels": ["O"] * (n - 1) + ["PERIOD"], "case_labels": ["LOWER"] * n}
                f.write(json.dumps(rec) + "\n")
        return [s_id for s_id, _ in rows]

    def test_predict_writes_manifest_accepted_by_evaluator(self, tmp_path: Path) -> None:
        from punct_restore.evaluation.inputs import load_bundle, load_vocab
        from punct_restore.evaluation.manifest import check_manifest

        model = self._train_mode_a(tmp_path)
        gold, ids, preds, man = (tmp_path / n for n in ("gold.jsonl", "g.ids", "p.jsonl", "p.man.json"))
        write_id_file(ids, self._write_gold(gold))

        ret = train_crf.main(["predict", "--model", str(model), "--input", str(gold),
                              "--ids", str(ids), "--output", str(preds), "--manifest-out", str(man)])
        assert ret == 0

        vocab = load_vocab()
        check_manifest(man, load_bundle(gold, ids, preds, None, vocab), vocab)
        manifest = json.loads(man.read_text(encoding="utf-8"))
        assert set(manifest) == {"model_name", "label_mode", "ordered_labels",
                                 "predictions_sha256", "input"}
        assert not Path(manifest["input"]["path"]).is_absolute()
        assert manifest["predictions_sha256"] == sha256_file(preds)

    def test_predict_rejects_a_casing_helper_model(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        records = [{"sample_id": f"c{i}", "split": "train", "label_mode": "C", "tokens": ["we", "agree"],
                    "punct_labels": ["O", "PERIOD"], "case_labels": ["FIRST_UPPER", "LOWER"]}
                   for i in range(3)]
        model = tmp_path / "case.joblib"
        CRFModel(label_mode="CASE", max_iterations=5).fit(records).save_model(model)
        inp, out, man = tmp_path / "in.jsonl", tmp_path / "p.jsonl", tmp_path / "p.man.json"
        inp.write_text(json.dumps({"sample_id": "s1", "tokens": ["hello"]}) + "\n", encoding="utf-8")

        ret = train_crf.main(["predict", "-m", str(model), "-i", str(inp), "-o", str(out),
                              "--manifest-out", str(man)])
        assert ret == 1
        assert not out.exists() and not man.exists()
        err = capsys.readouterr().err
        assert len(err.splitlines()) == 1 and "CASE" in err and "scoring mode" in err

    def test_predict_refuses_existing_manifest(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        model = self._train_mode_a(tmp_path)
        inp, man = tmp_path / "in.jsonl", tmp_path / "p.man.json"
        inp.write_text(json.dumps({"sample_id": "s1", "tokens": ["hello"]}) + "\n", encoding="utf-8")
        man.write_text("keep me", encoding="utf-8")

        ret = train_crf.main(["predict", "-m", str(model), "-i", str(inp),
                              "-o", str(tmp_path / "p.jsonl"), "--manifest-out", str(man)])
        assert ret == 1
        assert "Manifest already exists" in capsys.readouterr().err
        assert man.read_text(encoding="utf-8") == "keep me"


    def test_read_observations_is_gold_free_and_in_file_order(self, tmp_path: Path) -> None:
        gold, ids = tmp_path / "gold.jsonl", tmp_path / "g.ids"
        write_id_file(ids, list(reversed(self._write_gold(gold))))
        observations = read_observations(gold, ids)
        assert [o["sample_id"] for o in observations] == ["gold_a", "gold_b"]
        assert all(set(o) == {"sample_id", "tokens"} for o in observations)
        write_id_file(ids, ["gold_a", "ghost"])
        with pytest.raises(ValueError, match="Missing 1 selected IDs"):
            read_observations(gold, ids)

    def test_write_predictions_jsonl_uses_platform_newline(self, tmp_path: Path) -> None:
        pred = {"sample_id": "s1", "model_name": "m", "label_mode": "A", "predicted_labels": ["O"]}
        out = tmp_path / "nested" / "p.jsonl"
        write_predictions_jsonl([pred], out)
        expected = (json.dumps(pred, ensure_ascii=False) + "\n").replace("\n", os.linesep)
        assert out.read_bytes() == expected.encode("utf-8")
