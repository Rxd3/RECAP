"""A malformed input file of an HMM command is one `error: <file name>: <reason>` line and exit 1, never a traceback.

Every case runs the command's `main` in-process on a tiny valid fixture with ONE input replaced by a malformed file.
The command stops before it prints anything, so stdout stays empty.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from reporting_fixtures import RELEASE, time_argv, tiny_data_root, tiny_hmm
from test_hmm_cli import setup_run

from punct_restore.cli import package_hmm_handoff, predict_hmm, run_hmm_experiments, train_hmm, verify_hmm_experiments
from punct_restore.models import hmm_experiment_tune
from punct_restore.reporting import time_models

# What a malformed file holds: a line nested far deeper than the JSON reader allows, bytes that are no UTF-8,
# a JSON array where an object is expected, and an object that lacks every key.
BAD = {
    "nested": b"[" * 100_000 + b"\n",
    "not-utf8": b"\xff\xfe not text\n",
    "array": b"[1]\n",
    "object": b'{"a": 1}\n',
    "manifest-files-not-an-object": (b'{"schema_version": 1, "release_id": "r", "policy_version": "p", '
                                     b'"label_sets_sha256": "s", "files": [], "split_files": {}, '
                                     b'"source": {"revision": "x"}}\n'),
    "selection-without-modes": b'{"used_test_labels": false, "training_includes_validation": false}\n',
}
SHAPES = ["object", "manifest-files-not-an-object", "selection-without-modes"]
KINDS = ["nested", "not-utf8", "array", "object"]
VALID_SELECTION = {
    "plan_sha256": "0" * 64, "modes": {}, "mode_selection_sha256": {},
    "used_test_labels": False, "training_includes_validation": False,
}
VALID_LOOKING_PLAN = {"config": {}, "implementation": {}, "evaluator": {}}


def assert_one_error_line(capsys: pytest.CaptureFixture[str], tmp_path: Path, file_name: str) -> None:
    """Stdout is empty; stderr is one `error: <file name>...` line with no folder, no traceback and no usage text."""
    captured = capsys.readouterr()
    err = captured.err
    assert captured.out == ""
    assert err.startswith(f"error: {file_name}") and err.endswith("\n") and err.count("\n") == 1, err
    assert str(tmp_path) not in err and tmp_path.name not in err
    assert "Traceback" not in err and "usage:" not in err


def trained_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """A valid saved HMM, made by the real train-hmm in a folder of its own (what it printed is dropped)."""
    folder = tmp_path / "trained"
    folder.mkdir()
    argv = setup_run(folder)[0][2:]  # drop the interpreter and the script
    assert train_hmm.main(argv) == 0
    capsys.readouterr()
    return folder / "new-run" / "model.json"


@pytest.mark.parametrize("kind", KINDS)
def test_train_hmm_with_a_malformed_train_file_is_one_error_line(
        kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv, train, _, output = setup_run(tmp_path)
    train.write_bytes(BAD[kind])
    assert train_hmm.main(argv[2:]) == 1
    assert_one_error_line(capsys, tmp_path, "train.jsonl")
    assert not output.exists()


def predict_argv(tmp_path: Path, model: Path, source: Path) -> list[str]:
    return ["--model", str(model), "--input", str(source),
            "--output-dir", str(tmp_path / "predicted"), "--all-records"]


@pytest.mark.parametrize("kind", KINDS)
def test_predict_hmm_with_a_malformed_model_is_one_error_line(
        kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = setup_run(tmp_path)[2]
    model = tmp_path / "model.json"
    model.write_bytes(BAD[kind])
    assert predict_hmm.main(predict_argv(tmp_path, model, source)) == 1
    assert_one_error_line(capsys, tmp_path, "model.json")
    assert not (tmp_path / "predicted").exists()


@pytest.mark.parametrize("kind", KINDS)
def test_predict_hmm_with_a_malformed_input_file_is_one_error_line(
        kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = trained_model(tmp_path, capsys)
    source = tmp_path / "records.jsonl"
    source.write_bytes(BAD[kind])
    assert predict_hmm.main(predict_argv(tmp_path, model, source)) == 1
    assert_one_error_line(capsys, tmp_path, "records.jsonl")
    assert not (tmp_path / "predicted").exists()


@pytest.mark.parametrize("kind", KINDS)
def test_time_models_with_a_malformed_hmm_model_is_one_error_line(
        kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = tmp_path / "hmm.json"
    model.write_bytes(BAD[kind])
    argv = time_argv(model, tmp_path / "timing.json", tiny_data_root(tmp_path / "data"))
    assert time_models.main(argv) == 1
    assert_one_error_line(capsys, tmp_path, "hmm.json")
    assert not (tmp_path / "timing.json").exists()


def test_time_models_still_times_a_valid_hmm_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The fixture of the cases above is sound: with an unbroken model the same command succeeds."""
    argv = time_argv(tiny_hmm(tmp_path), tmp_path / "timing.json", tiny_data_root(tmp_path / "data"))
    assert time_models.main(argv) == 0
    assert capsys.readouterr().err == ""


def frozen_run(tmp_path: Path, plan: bytes | dict[str, object], selection: bytes | dict[str, object]) -> Path:
    """A run folder holding only a plan.json and a selection.json, each given as bytes or as a JSON value."""
    run = tmp_path / "run"
    run.mkdir()
    for name, content in (("plan.json", plan), ("selection.json", selection)):
        (run / name).write_bytes(content if isinstance(content, bytes) else json.dumps(content).encode("utf-8"))
    return run


def _finalize(run: Path, tmp_path: Path) -> int:
    return run_hmm_experiments.main(["finalize", "--run-dir", str(run), "--evaluator-root", str(tmp_path / "ev")])


def _audit(command: Callable[[list[str] | None], int]) -> Callable[[Path, Path], int]:
    def call(run: Path, tmp_path: Path) -> int:
        return command(["--run-dir", str(run), "--evaluator-root", str(tmp_path / "ev"),
                        "--output-dir", str(tmp_path / "out")])

    return call


GATE_COMMANDS = {
    "finalize": _finalize,
    "verify": _audit(verify_hmm_experiments.main),
    "package": _audit(package_hmm_handoff.main),
}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", GATE_COMMANDS)
def test_a_malformed_plan_is_one_error_line_in_every_command_that_passes_the_frozen_gate(
        name: str, kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = frozen_run(tmp_path, BAD[kind], VALID_SELECTION)
    assert GATE_COMMANDS[name](run, tmp_path) == 1
    assert_one_error_line(capsys, tmp_path, "plan.json")
    assert not (tmp_path / "out").exists() and not (run / "final").exists()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", GATE_COMMANDS)
def test_a_malformed_selection_is_one_error_line_in_every_command_that_passes_the_frozen_gate(
        name: str, kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = frozen_run(tmp_path, VALID_LOOKING_PLAN, BAD[kind])
    assert GATE_COMMANDS[name](run, tmp_path) == 1
    assert_one_error_line(capsys, tmp_path, "selection.json")
    assert not (tmp_path / "out").exists() and not (run / "final").exists()


@pytest.fixture
def clean_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real `tune` runs, on a checkout `git status` calls clean (the one under test has uncommitted changes)."""
    monkeypatch.setattr(hmm_experiment_tune, "code_identity", lambda: {"commit": "c", "dirty": False})


@pytest.mark.parametrize("kind", ["not-utf8", "nested"])
def test_tune_with_a_malformed_config_is_one_error_line(
        kind: str, tmp_path: Path, clean_checkout: None, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_bytes(BAD[kind])
    argv = ["tune", "--config", str(config), "--output-dir", str(tmp_path / "out"),
            "--evaluator-root", str(tmp_path / "ev")]
    assert run_hmm_experiments.main(argv) == 1
    assert_one_error_line(capsys, tmp_path, "experiment.yaml")
    assert not (tmp_path / "out").exists()


def test_time_models_names_a_corrupt_hmm_model_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The HMM loader starts its message with the file name; the command must not put it there a second time."""
    bad = tmp_path / "broken.json"
    bad.write_bytes(b"{oops")
    assert time_models.main(time_argv(bad, tmp_path / "t.json", tiny_data_root(tmp_path / "data"))) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: broken.json: invalid HMM artifact: ") and err.count("broken.json") == 1


@pytest.mark.parametrize("kind", ["nested", "not-utf8"])
def test_time_models_with_a_malformed_validation_file_is_one_error_line(
        kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tiny_data_root(tmp_path / "data")
    records = data / "processed" / RELEASE / "mode_A" / "validation.jsonl"
    records.write_bytes(BAD[kind])
    assert time_models.main(time_argv(tiny_hmm(tmp_path), tmp_path / "t.json", data)) == 1
    assert_one_error_line(capsys, tmp_path, "validation.jsonl")
    assert not (tmp_path / "t.json").exists()


@pytest.mark.parametrize("kind", SHAPES)
def test_train_hmm_with_a_wrong_shape_release_manifest_is_one_error_line(
        kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv, _, _, output = setup_run(tmp_path)
    broken = tmp_path / "broken.input"
    broken.write_bytes(BAD[kind])
    assert train_hmm.main([*argv[2:], "--release-manifest", str(broken)]) == 1
    assert_one_error_line(capsys, tmp_path, "broken.input")
    assert not output.exists()


@pytest.mark.parametrize("flag", ["--release-manifest", "--selection-record"])
@pytest.mark.parametrize("kind", SHAPES)
def test_predict_hmm_with_a_wrong_shape_manifest_or_selection_record_is_one_error_line(
        flag: str, kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = trained_model(tmp_path, capsys)
    broken = tmp_path / "broken.input"
    broken.write_bytes(BAD[kind])
    argv = predict_argv(tmp_path, model, model.parents[1] / "predict.jsonl")
    assert predict_hmm.main([*argv, flag, str(broken)]) == 1
    assert_one_error_line(capsys, tmp_path, "broken.input")
    assert not (tmp_path / "predicted").exists()


@pytest.mark.parametrize("flag", ["--label-config", "--release-manifest"])
@pytest.mark.parametrize("kind", ["nested", "not-utf8", "array"])
def test_train_hmm_with_a_malformed_config_or_manifest_is_one_error_line(
        flag: str, kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv, _, _, output = setup_run(tmp_path)
    broken = tmp_path / "broken.input"
    broken.write_bytes(BAD[kind])
    assert train_hmm.main([*argv[2:], flag, str(broken)]) == 1
    assert_one_error_line(capsys, tmp_path, "broken.input")
    assert not output.exists()


@pytest.mark.parametrize("flag", ["--release-manifest", "--selection-record"])
@pytest.mark.parametrize("kind", ["nested", "not-utf8", "array"])
def test_predict_hmm_with_a_malformed_manifest_or_selection_record_is_one_error_line(
        flag: str, kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = trained_model(tmp_path, capsys)
    broken = tmp_path / "broken.input"
    broken.write_bytes(BAD[kind])
    argv = predict_argv(tmp_path, model, model.parents[1] / "predict.jsonl")
    assert predict_hmm.main([*argv, flag, str(broken)]) == 1
    assert_one_error_line(capsys, tmp_path, "broken.input")
    assert not (tmp_path / "predicted").exists()
