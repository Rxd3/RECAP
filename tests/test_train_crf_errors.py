"""train-crf: a data or file problem is one stderr line naming files by name only; a code bug propagates."""

from __future__ import annotations

import errno
import json
from pathlib import Path
from typing import Any

import pytest
from crf_cli_fixtures import create_synthetic_train_jsonl, write_id_file

import punct_restore.cli.train_crf as train_crf
from punct_restore.models.crf import CRFModel


def _train_argv(tmp_path: Path, *extra: str) -> list[str]:
    data, ids = tmp_path / "data.jsonl", tmp_path / "train.ids"
    write_id_file(ids, create_synthetic_train_jsonl(data, mode="A", num_samples=3))
    return ["train", "-m", "A", "-i", str(data), "--ids", str(ids), "-o", str(tmp_path / "out.joblib"),
            "--max-iterations", "5", *extra]


def _predict_argv(model: Path, tmp_path: Path, *extra: str) -> list[str]:
    source = tmp_path / "eval.jsonl"
    source.write_text(json.dumps({"sample_id": "e1", "tokens": ["hello", "world"]}) + "\n", encoding="utf-8")
    return ["predict", "-m", str(model), "-i", str(source), "-o", str(tmp_path / "preds.jsonl"), *extra]


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    assert train_crf.main(_train_argv(tmp_path)) == 0
    return tmp_path / "out.joblib"


def _raising(error: BaseException) -> Any:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    return fail


def _only_line(err: str, tmp_path: Path) -> str:
    lines = err.splitlines()
    assert len(lines) == 1, err
    assert str(tmp_path) not in lines[0], "the line shows a folder"
    return lines[0]


# --- train: data problems are one line ----------------------------------------------------------------


def test_a_missing_input_file_is_one_line_naming_the_file_only(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = _train_argv(tmp_path)
    (tmp_path / "data.jsonl").unlink()
    assert train_crf.main(argv) == 1
    line = _only_line(capsys.readouterr().err, tmp_path)
    assert line == "error: Input JSONL file not found: data.jsonl"


def test_a_corrupt_input_line_is_one_line_naming_the_file_only(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = _train_argv(tmp_path)
    (tmp_path / "data.jsonl").write_text("{not json\n", encoding="utf-8")
    assert train_crf.main(argv) == 1
    line = _only_line(capsys.readouterr().err, tmp_path)
    assert line.startswith("error: ") and "Corrupt JSON at line 1 in data.jsonl" in line


def test_a_bad_hyperparameter_is_a_configuration_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert train_crf.main(_train_argv(tmp_path, "--c1", "-1.0")) == 1
    assert _only_line(capsys.readouterr().err, tmp_path).startswith("error: ")


def test_a_backend_that_refuses_the_algorithm_is_a_training_failure(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert train_crf.main(_train_argv(tmp_path, "--algorithm", "no-such-algorithm")) == 1
    assert _only_line(capsys.readouterr().err, tmp_path).startswith("error: training failed: ")


# --- train: a code bug is not reported as bad data ------------------------------------------------------


def test_a_bug_while_loading_records_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_crf, "load_and_filter_training_records", _raising(KeyError("sample_id")))
    with pytest.raises(KeyError):
        train_crf.main(_train_argv(tmp_path))


def test_a_bug_while_configuring_the_model_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_crf, "CRFModel", _raising(TypeError("bad feature_config")))
    with pytest.raises(TypeError):
        train_crf.main(_train_argv(tmp_path))


def test_a_bug_while_fitting_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CRFModel, "fit", _raising(KeyError("label")))
    with pytest.raises(KeyError):
        train_crf.main(_train_argv(tmp_path))


# --- train: the outputs ---------------------------------------------------------------------------------


def test_a_full_disk_while_saving_the_model_is_one_line_and_writes_no_manifest(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out.joblib"
    monkeypatch.setattr(CRFModel, "save_model", _raising(OSError(errno.ENOSPC, "No space left on device", str(out))))
    assert train_crf.main(_train_argv(tmp_path)) == 1
    line = _only_line(capsys.readouterr().err, tmp_path)
    assert line == "error: out.joblib: No space left on device"
    assert not (tmp_path / "out.joblib.manifest.json").exists()


def test_a_full_disk_while_writing_the_manifest_is_one_line(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = tmp_path / "out.joblib.manifest.json"
    full_disk = OSError(errno.ENOSPC, "No space left on device", str(manifest))
    monkeypatch.setattr(train_crf.json, "dump", _raising(full_disk))
    assert train_crf.main(_train_argv(tmp_path)) == 1
    line = _only_line(capsys.readouterr().err, tmp_path)
    assert line == "error: out.joblib.manifest.json: No space left on device"


def test_a_bug_while_saving_the_model_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CRFModel, "save_model", _raising(RuntimeError("Cannot save an unfitted CRFModel.")))
    with pytest.raises(RuntimeError):
        train_crf.main(_train_argv(tmp_path))


def test_an_existing_artifact_is_named_by_file_name_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = _train_argv(tmp_path)
    (tmp_path / "out.joblib").write_bytes(b"keep")
    assert train_crf.main(argv) == 1
    err = capsys.readouterr().err
    assert err == "error: out.joblib: output artifact already exists; use --overwrite to replace\n"


def test_an_existing_manifest_is_named_by_file_name_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    taken = tmp_path / "taken.manifest.json"
    taken.write_text("{}", encoding="utf-8")
    assert train_crf.main(_train_argv(tmp_path, "--manifest-out", str(taken))) == 1
    err = capsys.readouterr().err
    assert err == "error: taken.manifest.json: manifest already exists; use --overwrite to replace\n"


# --- predict ---------------------------------------------------------------------------------------------


def test_a_corrupt_model_is_one_line_naming_the_file_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"this is not a joblib file")
    assert train_crf.main(_predict_argv(bad, tmp_path)) == 1
    line = _only_line(capsys.readouterr().err, tmp_path)
    assert line.startswith("error: Corrupt or unreadable model artifact at bad.joblib: ")


def test_a_bug_while_loading_the_model_propagates(
        tmp_path: Path, model_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CRFModel, "load_model", classmethod(_raising(KeyError("backend_state"))))
    with pytest.raises(KeyError):
        train_crf.main(_predict_argv(model_file, tmp_path))


def test_a_model_that_predicts_nonsense_is_a_prediction_failure(
        tmp_path: Path, model_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(CRFModel, "predict", _raising(ValueError("Model predicted 1 labels for sequence e1")))
    assert train_crf.main(_predict_argv(model_file, tmp_path)) == 1
    line = _only_line(capsys.readouterr().err, tmp_path)
    assert line == "error: prediction failed: Model predicted 1 labels for sequence e1"


def test_a_bug_while_predicting_propagates(
        tmp_path: Path, model_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CRFModel, "predict", _raising(TypeError("tokens must be a Sequence")))
    with pytest.raises(TypeError):
        train_crf.main(_predict_argv(model_file, tmp_path))


def test_predict_names_a_missing_model_and_input_by_file_name_only(
        tmp_path: Path, model_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ghost = tmp_path / "deep" / "ghost.joblib"
    assert train_crf.main(_predict_argv(ghost, tmp_path)) == 1
    assert capsys.readouterr().err == "error: ghost.joblib: model artifact not found\n"
    argv = _predict_argv(model_file, tmp_path)
    (tmp_path / "eval.jsonl").unlink()
    assert train_crf.main(argv) == 1
    assert capsys.readouterr().err == "error: eval.jsonl: input JSONL not found\n"


def test_predict_names_existing_outputs_by_file_name_only(
        tmp_path: Path, model_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = _predict_argv(model_file, tmp_path)
    (tmp_path / "preds.jsonl").write_text("keep\n", encoding="utf-8")
    assert train_crf.main(argv) == 1
    assert capsys.readouterr().err == "error: preds.jsonl: output file already exists; use --overwrite to replace\n"
    (tmp_path / "preds.jsonl").unlink()
    taken = tmp_path / "taken.manifest.json"
    taken.write_text("{}", encoding="utf-8")
    assert train_crf.main([*argv, "--manifest-out", str(taken)]) == 1
    err = capsys.readouterr().err
    assert err == "error: taken.manifest.json: manifest already exists; use --overwrite to replace\n"


def test_predict_input_problems_name_the_files_by_file_name_only(
        tmp_path: Path, model_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = _predict_argv(model_file, tmp_path)
    ids = tmp_path / "wanted.ids"
    write_id_file(ids, ["e1", "nope"])
    assert train_crf.main([*argv, "--ids", str(ids)]) == 1
    line = _only_line(capsys.readouterr().err, tmp_path)
    assert line.startswith("error: Missing 1 selected IDs from input file eval.jsonl")


# --- inspect -----------------------------------------------------------------------------------------------


def test_inspect_with_a_readable_manifest_prints_it_and_says_nothing_on_stderr(
        model_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert train_crf.main(["inspect", "-m", str(model_file)]) == 0
    captured = capsys.readouterr()
    assert "Companion Manifest:" in captured.out and captured.err == ""


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"provenance": 5}', "\xff\xfe"], ids=[
    "corrupt", "not-an-object", "provenance-not-an-object", "not-utf8"])
def test_inspect_says_so_when_the_companion_manifest_is_unreadable(
        model_file: Path, capsys: pytest.CaptureFixture[str], content: str) -> None:
    (model_file.parent / "out.joblib.manifest.json").write_bytes(content.encode("latin-1"))
    assert train_crf.main(["inspect", "-m", str(model_file)]) == 0
    captured = capsys.readouterr()
    assert captured.err == "Note: companion manifest unreadable: out.joblib.manifest.json\n"
    assert "Companion Manifest:" not in captured.out and "RECAP CRF Model Artifact" in captured.out


def test_inspect_goes_on_to_the_next_manifest_after_an_unreadable_one(
        model_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    first = model_file.parent / "out.joblib.manifest.json"
    (model_file.parent / "out.manifest.json").write_bytes(first.read_bytes())
    first.write_text("{not json", encoding="utf-8")
    assert train_crf.main(["inspect", "-m", str(model_file)]) == 0
    captured = capsys.readouterr()
    assert captured.err == "Note: companion manifest unreadable: out.joblib.manifest.json\n"
    assert "Companion Manifest:" in captured.out
