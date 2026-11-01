"""Member 2 CLI integration checks with temporary synthetic JSONL files."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from punct_restore.models.hmm import HMMModel

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_hmm.py"


def jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def setup_run(tmp_path, mode="A", split="train"):
    train = tmp_path / "train.jsonl"
    sample = {
        "schema_version": 1,
        "sample_id": "unit-train",
        "source": "synthetic_unit_fixture",
        "split": split,
        "label_mode": mode,
        "tokens": ["hello", "nasa"],
        "punct_labels": ["O", "COLON" if mode == "B" else "PERIOD"],
        "case_labels": ["FIRST_UPPER", "ALL_UPPER"],
    }
    jsonl(train, [sample])
    train_ids = tmp_path / "train.ids"
    train_ids.write_text("unit-train\n", encoding="utf-8")
    manifest = tmp_path / "split_manifest.json"
    # Only its checksum is consumed; Member 1 owns the eventual manifest schema/audit.
    manifest.write_text(json.dumps({"synthetic_fixture": True}), encoding="utf-8")
    prediction = tmp_path / "predict.jsonl"
    jsonl(
        prediction, [{"schema_version": 1, "sample_id": "unit-predict", "tokens": ["hello", "new"]}]
    )
    output = tmp_path / "new-run"
    args = [
        sys.executable,
        str(SCRIPT),
        "--train",
        str(train),
        "--train-ids",
        str(train_ids),
        "--output-dir",
        str(output),
        "--label-mode",
        mode,
        "--min-token-count",
        "1",
        "--dataset-release-id",
        "synthetic-unit-fixture-not-a-dataset",
        "--preprocessing-version",
        "fixture-only",
        "--tokenizer-version",
        "fixture-only",
        "--label-policy-version",
        "fixture-only",
        "--split-manifest",
        str(manifest),
    ]
    return args, train, prediction, output


@pytest.mark.parametrize("mode", ["A", "B", "C"])
def test_cli_round_trip_manifests_and_predictions(tmp_path, mode):
    args, train, prediction_input, output = setup_run(tmp_path, mode)
    # Run outside the checkout to verify config lookup and source-checkout imports.
    result = subprocess.run(
        args + ["--predict-input", str(prediction_input)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "No evaluation metrics calculated" in result.stdout
    assert {p.name for p in output.iterdir()} == {
        "model.json",
        "run.json",
        "predictions.jsonl",
        "predictions.manifest.json",
    }
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    model = HMMModel.load_model(output / "model.json")
    assert run["model_sha256"] == hashlib.sha256((output / "model.json").read_bytes()).hexdigest()
    assert (
        run["provenance"]["training_input"]["sha256"]
        == hashlib.sha256(train.read_bytes()).hexdigest()
    )
    assert model.metadata == run["provenance"]
    assert run["provenance"]["command"][:2] == ["python", "scripts/train_hmm.py"]
    assert str(tmp_path) not in json.dumps(run)
    assert run["training_summary"]["num_tokens"] == 2 and "metrics" not in run
    predictions = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    records = [
        json.loads(line) for line in prediction_input.read_text(encoding="utf-8").splitlines()
    ]
    assert predictions == model.predict(records)
    manifest = json.loads((output / "predictions.manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_sha256"] == run["model_sha256"]
    assert (
        manifest["predictions_sha256"]
        == hashlib.sha256((output / "predictions.jsonl").read_bytes()).hexdigest()
    )
    assert manifest["ordered_labels"] == list(model.labels)


def test_cli_points_to_the_evaluator_for_scoring(tmp_path):
    args, _, _, _ = setup_run(tmp_path)
    result = subprocess.run(args, cwd=tmp_path, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "scripts/evaluate_models.py score" in result.stdout
    assert "when available" not in result.stdout


def test_cli_does_not_overwrite_an_existing_run(tmp_path):
    args, _, _, output = setup_run(tmp_path)
    output.mkdir()
    marker = output / "existing.txt"
    marker.write_text("keep this", encoding="utf-8")
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode == 1 and result.stderr.startswith("error: ") and "already exists" in result.stderr
    assert result.stderr.count("\n") == 1 and "usage:" not in result.stderr
    assert list(output.iterdir()) == [marker]
    assert marker.read_text(encoding="utf-8") == "keep this"


@pytest.mark.parametrize(
    "bad_input", ["test-split", "blank-line", "invalid-json", "missing-file", "bad-prediction"]
)
def test_cli_rejects_invalid_input_before_creating_outputs(tmp_path, bad_input):
    args, train, prediction, output = setup_run(
        tmp_path, split="test" if bad_input == "test-split" else "train"
    )
    if bad_input == "blank-line":
        train.write_text(train.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif bad_input == "invalid-json":
        train.write_text("{not-json}\n", encoding="utf-8")
    elif bad_input == "missing-file":
        train.unlink()
    elif bad_input == "bad-prediction":
        jsonl(prediction, [{"schema_version": 1, "sample_id": "broken", "tokens": []}])
        args.extend(["--predict-input", str(prediction)])
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode == 1, result.stderr
    assert result.stderr.startswith("error: ") and "usage:" not in result.stderr and "Traceback" not in result.stderr
    assert not output.exists()


def test_cli_help_needs_no_dataset(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--train" in result.stdout and "--split-manifest" in result.stdout
