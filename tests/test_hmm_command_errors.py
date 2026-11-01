"""The five HMM commands: a file or value problem is one `error:` line and exit 1; a bug propagates; usage stays 2."""

from __future__ import annotations

import errno
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from test_hmm_cli import setup_run

from punct_restore.cli import (
    package_hmm_handoff,
    predict_hmm,
    run_hmm_experiments,
    train_hmm,
    verify_hmm_experiments,
)

RUN_ARGV = ["--run-dir", "run", "--evaluator-root", "ev", "--output-dir", "out"]
# (command module, the function main() calls and a test can replace, a command line that reaches it)
COMMANDS = {
    "tune": (run_hmm_experiments, "tune",
             ["tune", "--config", "c.json", "--output-dir", "out", "--evaluator-root", "ev"]),
    "finalize": (run_hmm_experiments, "finalize", ["finalize", "--run-dir", "run", "--evaluator-root", "ev"]),
    "train": (train_hmm, "_run", None),
    "verify": (verify_hmm_experiments, "verify", RUN_ARGV),
    "package": (package_hmm_handoff, "package", RUN_ARGV),
    "predict": (predict_hmm, "predict_run",
                ["--model", "m.json", "--input", "in.jsonl", "--output-dir", "out", "--all-records"]),
}


def _raising(error: BaseException) -> Any:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    return fail


def _setup(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
           error: BaseException) -> tuple[ModuleType, list[str]]:
    module, target, argv = COMMANDS[name]
    if argv is None:
        argv = setup_run(tmp_path)[0][2:]  # drop the interpreter and the script
    monkeypatch.setattr(module, target, _raising(error))
    if module is run_hmm_experiments:
        monkeypatch.setattr(module, "code_identity", lambda: {"dirty": False})
    return module, argv


@pytest.mark.parametrize("name", COMMANDS)
def test_a_missing_file_is_one_error_line_naming_the_file_only_and_exit_1(
        name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    gone = OSError(errno.ENOENT, "No such file or directory", str(tmp_path / "deep" / "plan.json"))
    module, argv = _setup(name, tmp_path, monkeypatch, gone)
    assert module.main(argv) == 1
    captured = capsys.readouterr()
    assert captured.err == "error: plan.json: No such file or directory\n"
    assert captured.out == ""


@pytest.mark.parametrize("name", COMMANDS)
def test_a_bad_value_is_one_error_line_and_exit_1(
        name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    module, argv = _setup(name, tmp_path, monkeypatch, ValueError("first\nsecond"))
    assert module.main(argv) == 1
    assert capsys.readouterr().err == "error: first second\n"


@pytest.mark.parametrize("name", COMMANDS)
@pytest.mark.parametrize("bug", [KeyError("k"), TypeError("t"), RuntimeError("r")], ids=["key", "type", "runtime"])
def test_a_bug_is_not_reported_as_a_user_error(
        name: str, bug: BaseException, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module, argv = _setup(name, tmp_path, monkeypatch, bug)
    with pytest.raises(type(bug)):
        module.main(argv)


@pytest.mark.parametrize("module", [run_hmm_experiments, train_hmm, verify_hmm_experiments, package_hmm_handoff,
                                    predict_hmm], ids=lambda m: m.__name__.rsplit(".", 1)[1])
def test_a_usage_error_is_still_exit_2_with_the_usage_text(
        module: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        module.main([])
    assert stop.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_train_hmm_with_a_missing_ids_file_is_one_error_line(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The case the grade reproduced: it used to print the whole usage text and `[Errno 2]`, exit 2."""
    argv = setup_run(tmp_path)[0][2:]
    argv[argv.index("--train-ids") + 1] = str(tmp_path / "deep" / "no.ids")
    assert train_hmm.main(argv) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: no.ids: ") and err.count("\n") == 1
    assert "usage:" not in err and "Errno" not in err and str(tmp_path) not in err
    assert not (tmp_path / "new-run").exists()


def test_train_hmm_flags_predict_ids_without_predict_input_as_a_usage_error(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = [*setup_run(tmp_path)[0][2:], "--predict-ids", str(tmp_path / "train.ids")]
    with pytest.raises(SystemExit) as stop:
        train_hmm.main(argv)
    assert stop.value.code == 2 and "--predict-ids requires --predict-input" in capsys.readouterr().err


def test_predict_hmm_with_a_missing_model_is_one_error_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "in.jsonl"
    source.write_text('{"sample_id": "a", "tokens": ["x"]}\n', encoding="utf-8")
    argv = ["--model", str(tmp_path / "deep" / "nomodel.json"), "--input", str(source),
            "--output-dir", str(tmp_path / "out"), "--all-records"]
    assert predict_hmm.main(argv) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "nomodel.json" in err and err.count("\n") == 1
    assert "usage:" not in err and str(tmp_path) not in err

