"""train-crf train: the helpers raise one TrainingFailed that cmd_train reports once; missing memory figures are noted.

TrainingFailed lives in punct_restore.models.crf_training. The helpers of punct_restore.cli.train_crf print their
progress but never an error: the message travels in the exception and `cmd_train` writes it as the one `error:` line.
"""

from __future__ import annotations

import argparse
import errno
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from crf_cli_fixtures import create_synthetic_train_jsonl, write_id_file

import punct_restore.cli.train_crf as train_crf
from punct_restore.models import crf_training
from punct_restore.models.crf import CRFModel
from punct_restore.models.crf_training import TrainingFailed

SAVE_ERROR = "error: out.joblib: No space left on device"


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    data, ids = tmp_path / "data.jsonl", tmp_path / "train.ids"
    write_id_file(ids, create_synthetic_train_jsonl(data, mode="A", num_samples=3))
    return ["train", "-m", "A", "-i", str(data), "--ids", str(ids), "-o", str(tmp_path / "out.joblib"),
            "--max-iterations", "5", *extra]


def _args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    return train_crf.build_cli_parser().parse_args(_argv(tmp_path, *extra))


def _fails(error: BaseException) -> Callable[..., Any]:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    return fail


# --- TrainingFailed -----------------------------------------------------------------------------------------


def test_training_failed_carries_the_one_line_message_and_is_neither_an_os_nor_a_value_error() -> None:
    failure = TrainingFailed("out.joblib: boom")
    assert str(failure) == "out.joblib: boom"
    assert not isinstance(failure, (OSError, ValueError)), "a helper's own except clause must not swallow it"


# --- the helpers raise it, and print no error ---------------------------------------------------------------


def test_a_path_problem_is_raised_by_prepare_train_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _args(tmp_path)
    (tmp_path / "out.joblib").write_bytes(b"keep")
    with pytest.raises(TrainingFailed) as caught:
        train_crf._prepare_train_paths(args)
    assert str(caught.value) == "out.joblib: output artifact already exists; use --overwrite to replace"
    assert capsys.readouterr().err == ""


def test_an_existing_manifest_is_raised_by_prepare_train_paths(tmp_path: Path) -> None:
    args = _args(tmp_path)
    (tmp_path / "out.joblib.manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(TrainingFailed, match="^out.joblib.manifest.json: manifest already exists"):
        train_crf._prepare_train_paths(args)


def test_the_same_file_twice_is_raised_by_prepare_train_paths(tmp_path: Path) -> None:
    args = _args(tmp_path, "--manifest-out", str(tmp_path / "out.joblib"))
    with pytest.raises(TrainingFailed, match="--output and --manifest-out are the same file"):
        train_crf._prepare_train_paths(args)


def test_a_data_problem_is_raised_by_load_records(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _args(tmp_path)
    (tmp_path / "data.jsonl").write_text("{not json\n", encoding="utf-8")
    with pytest.raises(TrainingFailed) as caught:
        train_crf._load_records(tmp_path / "data.jsonl", tmp_path / "train.ids", args.mode)
    assert str(caught.value) == "data.jsonl line 1: not valid JSON"
    assert capsys.readouterr().err == ""


def test_a_bad_hyperparameter_is_raised_by_configure_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(TrainingFailed):
        train_crf._configure_model(_args(tmp_path, "--c1", "-1.0"), "A")
    assert capsys.readouterr().err == ""


def test_a_refusing_backend_is_raised_by_fit_model(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(CRFModel, "fit", _fails(ValueError("bad records")))
    with pytest.raises(TrainingFailed) as caught:
        train_crf._fit_model(CRFModel(label_mode="A"), [], {})
    assert str(caught.value) == "training failed: bad records"
    assert capsys.readouterr().err == ""


def test_a_failed_save_is_raised_by_save_artifact_and_removes_what_it_wrote(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out.joblib"

    def full_disk(_model: CRFModel, path: Path) -> None:
        Path(path).write_bytes(b"half")
        raise OSError(errno.ENOSPC, "No space left on device", str(path))

    monkeypatch.setattr(CRFModel, "save_model", full_disk)
    with pytest.raises(TrainingFailed) as caught:
        train_crf._save_artifact(CRFModel(label_mode="A"), out)
    assert str(caught.value) == "out.joblib: No space left on device"
    assert not out.exists() and capsys.readouterr().err == ""


def test_a_failed_manifest_write_is_raised_by_write_manifest(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = tmp_path / "m.json"
    monkeypatch.setattr(train_crf.json, "dump", _fails(OSError(errno.ENOSPC, "No space left on device", str(manifest))))
    with pytest.raises(TrainingFailed) as caught:
        train_crf._write_manifest({}, manifest)
    assert str(caught.value) == "m.json: No space left on device"
    assert capsys.readouterr().err == ""


# --- cmd_train reports it once -----------------------------------------------------------------------------


def test_cmd_train_writes_a_training_failed_message_once_and_returns_one(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(train_crf, "_fit_model", _fails(TrainingFailed("training failed: boom")))
    assert train_crf.main(_argv(tmp_path)) == 1
    assert capsys.readouterr().err == "error: training failed: boom\n"


def test_cmd_train_still_lets_a_bug_propagate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_crf, "_fit_model", _fails(KeyError("label")))
    with pytest.raises(KeyError):
        train_crf.main(_argv(tmp_path))


def test_a_failed_save_still_reads_the_same_through_main(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out.joblib"
    monkeypatch.setattr(CRFModel, "save_model", _fails(OSError(errno.ENOSPC, "No space left on device", str(out))))
    assert train_crf.main(_argv(tmp_path)) == 1
    assert capsys.readouterr().err == SAVE_ERROR + "\n"


# --- memory figures that the OS cannot give --------------------------------------------------------------------


def _manifest(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "out.joblib.manifest.json").read_text(encoding="utf-8"))


FIGURES = {"peak_working_set_mb": 10.5, "current_working_set_mb": 9.5,
           "peak_private_commit_mb": 20.25, "current_private_commit_mb": 19.25}


def test_unavailable_memory_figures_are_null_with_a_reason_and_one_note(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(train_crf, "process_memory", lambda: (None, "no kernel counters here"))
    assert train_crf.main(_argv(tmp_path)) == 0
    manifest = _manifest(tmp_path)
    assert manifest["memory_diagnostics"] is None
    assert manifest["memory_diagnostics_unavailable"] == "no kernel counters here"
    captured = capsys.readouterr()
    notes = [line for line in captured.out.splitlines() if line.startswith("[RECAP CRF] Note:")]
    assert notes == ["[RECAP CRF] Note: process memory figures are unavailable (no kernel counters here); "
                     "the manifest records none."]
    assert captured.err == ""


def test_available_memory_figures_keep_the_manifest_as_it_was(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(train_crf, "process_memory", lambda: (FIGURES, None))
    assert train_crf.main(_argv(tmp_path)) == 0
    manifest = _manifest(tmp_path)
    assert manifest["memory_diagnostics"] == FIGURES
    assert "memory_diagnostics_unavailable" not in manifest
    out = capsys.readouterr().out
    assert "Process Memory (Kernel): Peak Working Set = 10.5 MB, Peak Private Commit = 20.25 MB" in out
    assert "unavailable" not in out


# --- process_memory --------------------------------------------------------------------------------------------


def test_process_memory_returns_the_figures_and_no_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crf_training, "get_process_memory_stats", lambda: dict(FIGURES))
    assert crf_training.process_memory() == (FIGURES, None)


def test_process_memory_gives_a_reason_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(crf_training, "get_process_memory_stats", dict)
    assert crf_training.process_memory() == (None, "not available on linux")


def test_process_memory_gives_a_reason_when_windows_returns_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(crf_training, "get_process_memory_stats", dict)
    assert crf_training.process_memory() == (None, "the Windows memory API gave no figures")


# --- the library messages name files by name only ---------------------------------------------------------------


def test_the_loader_never_puts_the_folder_in_its_messages(tmp_path: Path) -> None:
    _argv(tmp_path)  # writes data.jsonl and train.ids
    data, ids = tmp_path / "data.jsonl", tmp_path / "train.ids"
    ids.write_text(ids.read_text(encoding="utf-8") + "ghost\n", encoding="utf-8")
    with pytest.raises(ValueError) as missing:
        train_crf.load_and_filter_training_records(data, ids, "A")
    assert str(missing.value).startswith("data.jsonl: missing 1 selected IDs.")
    ids.write_text("sample_0000\n", encoding="utf-8")
    lines = len(data.read_text(encoding="utf-8").splitlines())
    data.write_text(data.read_text(encoding="utf-8") * 2, encoding="utf-8")
    with pytest.raises(ValueError) as duplicate:
        train_crf.load_and_filter_training_records(data, ids, "A")
    assert str(duplicate.value) == f"data.jsonl line {lines + 1}: duplicate sample_id 'sample_0000'"
    assert str(tmp_path) not in str(missing.value) + str(duplicate.value)


@pytest.mark.parametrize(("change", "reason"), [
    ({"split": "val"}, "'sample_0001' has split 'val', expected 'train'. Non-train records cannot enter training."),
    ({"label_mode": "B"}, "'sample_0001' has label_mode 'B', expected 'A'"),
    ({"tokens": []}, "'sample_0001' has empty or non-list tokens"),
    ({"punct_labels": ["O"]}, "'sample_0001' has mismatched punct_labels length (1 vs 3)"),
], ids=["split", "label-mode", "tokens", "labels"])
def test_a_listed_record_that_breaks_the_frozen_data_rules_is_named_with_its_file_and_line(
        tmp_path: Path, change: dict[str, Any], reason: str) -> None:
    _argv(tmp_path)  # writes data.jsonl and train.ids; the record on line 2 is sample_0001, three tokens
    data, ids = tmp_path / "data.jsonl", tmp_path / "train.ids"
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    rows[1].update(change)
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        train_crf.load_and_filter_training_records(data, ids, "A")
    assert str(caught.value) == f"data.jsonl line 2: {reason}"
