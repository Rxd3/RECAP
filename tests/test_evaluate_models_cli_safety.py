"""Safety tests for scripts/evaluate_models.py: non-UTF-8 input, hostile model names, failed output, bad --out.

main(argv) runs in-process through the `cli` fixture; every test works inside its own tmp_path.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn

import pytest
from eval_fixtures import FILE_NAMES, SCORED, Case, corrupt_line_two, gold_rows, perfect_preds, valid_case, write_case

from punct_restore.evaluation import EvaluationInputError


# ---------------------------------------------------------------- helpers
def run(cli: ModuleType, capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str, str]:
    """Call cli.main(argv) in-process and return (exit code, stdout, stderr)."""
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def assert_one_error_line(err: str) -> None:
    assert err.startswith("error: "), repr(err)
    assert err.endswith("\n"), repr(err)
    assert err.count("\n") == 1, repr(err)
    assert "\r" not in err, repr(err)
    assert "Traceback" not in err, err


def score_argv(case: Case, out: Path) -> list[str]:
    argv = ["score", "--gold", str(case.gold), "--ids", str(case.ids), "--predictions", str(case.predictions)]
    if case.masks is not None:
        argv += ["--masks", str(case.masks)]
    return argv + ["--out", str(out)]


def raising(error: BaseException) -> Callable[..., NoReturn]:
    def _raise(*args: Any, **kwargs: Any) -> NoReturn:
        raise error
    return _raise


def listing(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


# ---------------------------------------------------------------- non-UTF-8 input
@pytest.mark.parametrize("target", FILE_NAMES)
def test_cli_non_utf8_input_exit_1(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                   target: str) -> None:
    case = valid_case(tmp_path, "C")
    corrupt_line_two(getattr(case, target))
    out = tmp_path / "RUN.json"
    code, stdout, err = run(cli, capsys, score_argv(case, out))
    assert code == 1
    assert err == f"error: {FILE_NAMES[target]} line 2: not UTF-8 text\n"
    assert stdout == ""
    assert not out.exists()
    assert not out.with_suffix(".md").exists()


def test_cli_baseline_non_utf8_gold_exit_1(cli: ModuleType, capsys: pytest.CaptureFixture[str],
                                           tmp_path: Path) -> None:
    case = valid_case(tmp_path, "A")
    corrupt_line_two(case.gold)
    out = tmp_path / "base.jsonl"
    code, stdout, err = run(cli, capsys, ["baseline", "--gold", str(case.gold), "--ids", str(case.ids),
                                          "--out", str(out)])
    assert code == 1
    assert err == "error: gold.jsonl line 2: not UTF-8 text\n"
    assert stdout == ""
    assert not out.exists()


# ---------------------------------------------------------------- hostile model names
@pytest.mark.parametrize("model", ["m\nx|y", "m\rx", "m\x85x"], ids=["newline_and_pipe", "carriage_return", "c1_nel"])
def test_cli_success_line_has_no_raw_model_newline(cli: ModuleType, capsys: pytest.CaptureFixture[str],
                                                   tmp_path: Path, model: str) -> None:
    case = write_case(tmp_path, gold_rows("A"), SCORED, perfect_preds("A", model=model))
    out = tmp_path / "RUN.json"
    code, stdout, err = run(cli, capsys, score_argv(case, out))
    assert (code, err) == (0, "")
    assert stdout.count("\n") == 1, repr(stdout)
    assert "\r" not in stdout, repr(stdout)
    assert "m?x" in stdout, repr(stdout)
    assert json.loads(out.read_text(encoding="utf-8"))["model_name"] == model


# ---------------------------------------------------------------- failed Markdown writes nothing
def test_cli_markdown_input_error_writes_nothing(cli: ModuleType, capsys: pytest.CaptureFixture[str],
                                                 tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = valid_case(tmp_path, "A")
    out = tmp_path / "RUN.json"
    monkeypatch.setattr(cli, "to_markdown", raising(EvaluationInputError("boom")))
    code, stdout, err = run(cli, capsys, score_argv(case, out))
    assert code == 1
    assert err == "error: boom\n"
    assert stdout == ""
    assert not out.exists()
    assert not out.with_suffix(".md").exists()


def test_cli_markdown_unexpected_error_propagates_and_writes_nothing(cli: ModuleType, tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    case = valid_case(tmp_path, "A")
    out = tmp_path / "RUN.json"
    monkeypatch.setattr(cli, "to_markdown", raising(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        cli.main(score_argv(case, out))
    assert not out.exists()
    assert not out.with_suffix(".md").exists()


# ---------------------------------------------------------------- --out must be a .json path
@pytest.mark.parametrize("out_name", ["run.txt", "RUN"])
def test_cli_out_must_be_json(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                              out_name: str) -> None:
    case = valid_case(tmp_path, "A")
    before = listing(tmp_path)
    code, stdout, err = run(cli, capsys, score_argv(case, tmp_path / out_name))
    assert code == 1
    assert_one_error_line(err)
    assert "--out" in err
    assert stdout == ""
    assert listing(tmp_path) == before


def test_cli_out_suffix_is_case_insensitive(cli: ModuleType, capsys: pytest.CaptureFixture[str],
                                            tmp_path: Path) -> None:
    case = valid_case(tmp_path, "A")
    out = tmp_path / "RUN.JSON"
    code, _, err = run(cli, capsys, score_argv(case, out))
    assert (code, err) == (0, "")
    assert out.exists()
    assert (tmp_path / "RUN.md").exists()
