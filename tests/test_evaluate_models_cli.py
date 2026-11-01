"""Command line tests for punct_restore.cli.evaluate_models (spec section 8; RUN.json and RUN.md from section 7).

main(argv) runs in-process through the `cli` fixture; only the exit-code and encoding tests start a subprocess.
Tests marked EXPECTED TO FAIL document known defects of the current code and must not be weakened.
"""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn

import pytest
from eval_fixtures import (
    SCORED,
    SCRIPT_PATH,
    Case,
    all_masks,
    gold_rows,
    perfect_preds,
    pred_row,
    valid_case,
    write_case,
    write_jsonl,
)

from punct_restore.evaluation import (
    evaluate,
    load_bundle,
    load_scored_gold,
    load_vocab,
    shortcut_predictions,
    to_markdown,
)

OUT_NAMES = {"score": "RUN.json", "baseline": "base.jsonl"}
NON_ASCII_MODEL = "模型"
NON_ASCII_FILE = "предсказания.jsonl"


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


def score_argv(case: Case, out: Path, masks: bool = True) -> list[str]:
    argv = ["score", "--gold", str(case.gold), "--ids", str(case.ids), "--predictions", str(case.predictions)]
    if masks and case.masks is not None:
        argv += ["--masks", str(case.masks)]
    return argv + ["--out", str(out)]


def baseline_argv(case: Case, out: Path) -> list[str]:
    return ["baseline", "--gold", str(case.gold), "--ids", str(case.ids), "--out", str(out)]


def command_argv(command: str, case: Case, out: Path) -> list[str]:
    return (score_argv if command == "score" else baseline_argv)(case, out)


def output_paths(command: str, out: Path) -> tuple[Path, ...]:
    """Files the command writes for --out: RUN.json and RUN.md for score, the predictions file for baseline."""
    return (out, out.with_suffix(".md")) if command == "score" else (out,)


def assert_not_created(*paths: Path) -> None:
    for path in paths:
        assert not path.exists(), f"{path.name} must not exist"


def new_folder(parent: Path, name: str) -> Path:
    folder = parent / name
    folder.mkdir()
    return folder


def read_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bad_length_case(folder: Path) -> Case:
    """Mode A case whose s1 prediction has 2 labels for 3 tokens (preds.jsonl line 1)."""
    predictions = perfect_preds("A")
    predictions[0] = pred_row("s1", "A", ["O", "O"])
    return write_case(folder, gold_rows("A"), SCORED, predictions)


def run_script(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
               ) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT_PATH), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=60, cwd=cwd, env=env, check=False)


def cp1252_env() -> dict[str, str]:
    """Child environment with a cp1252 console on any OS: no UTF-8 mode, PYTHONIOENCODING=cp1252."""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONUTF8"}
    env["PYTHONIOENCODING"] = "cp1252"
    return env


def raising(error: BaseException) -> Callable[..., NoReturn]:
    def _raise(*args: Any, **kwargs: Any) -> NoReturn:
        raise error
    return _raise


def sorted_keys_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    assert keys == sorted(keys), f"object keys are not sorted: {keys}"
    return dict(pairs)


# ---------------------------------------------------------------- score: success and report files
def test_cli_score_success(cli, capsys, vocab, tmp_path):
    case = valid_case(tmp_path, "A")
    out = tmp_path / "RUN.json"
    code, _, err = run(cli, capsys, score_argv(case, out))
    expected = evaluate(load_bundle(case.gold, case.ids, case.predictions, None, vocab), vocab)
    assert code == 0
    assert err == ""
    assert out.exists()
    assert (tmp_path / "RUN.md").exists()
    assert read_report(out) == expected
    assert (tmp_path / "RUN.md").read_text(encoding="utf-8") == to_markdown(expected)


def test_cli_score_json_keys_sorted_and_ends_with_newline(cli, capsys, tmp_path):
    case = valid_case(tmp_path, "A")
    out = tmp_path / "RUN.json"
    code, _, _ = run(cli, capsys, score_argv(case, out))
    text = out.read_text(encoding="utf-8")
    assert code == 0
    json.loads(text, object_pairs_hook=sorted_keys_hook)
    assert text.endswith("\n")


def test_cli_score_mode_c_success(cli, capsys, tmp_path):
    case = valid_case(tmp_path, "C")
    out = tmp_path / "RUN.json"
    code, _, err = run(cli, capsys, score_argv(case, out, masks=True))
    report = read_report(out)
    assert code == 0
    assert err == ""
    assert report["primary"]["block"] == "joint/unmasked"
    assert report["inputs"]["masks"] is not None


# ---------------------------------------------------------------- baseline
@pytest.mark.parametrize("mode", ["A", "C"])
def test_cli_baseline_then_score(cli, capsys, vocab, tmp_path, mode):
    case = valid_case(tmp_path, mode)
    base = tmp_path / "base.jsonl"
    code, _, err = run(cli, capsys, baseline_argv(case, base))
    lines = [json.loads(line) for line in base.read_text(encoding="utf-8").splitlines()]
    assert code == 0
    assert err == ""
    assert lines == shortcut_predictions(load_scored_gold(case.gold, case.ids, vocab)[1], mode)

    out = tmp_path / "RUN.json"
    code, _, err = run(cli, capsys, score_argv(replace(case, predictions=base), out))
    report = read_report(out)
    assert code == 0
    assert err == ""
    assert report["model_name"] == "shortcut_baseline"
    if mode == "A":  # s2's COMMA is missed: COMMA f1 0, PERIOD 1, QUESTION 1
        assert report["primary"]["value"] == 0.666667


# ---------------------------------------------------------------- input errors
def test_cli_input_error_exit_1_one_line(cli, capsys, tmp_path):
    case = bad_length_case(tmp_path)
    out = tmp_path / "RUN.json"
    code, stdout, err = run(cli, capsys, score_argv(case, out))
    assert code == 1
    assert_one_error_line(err)
    assert "preds.jsonl" in err
    assert "line 1" in err
    assert stdout == ""
    assert_not_created(*output_paths("score", out))


def test_cli_masks_for_mode_a_exit_1(cli, capsys, tmp_path):
    case = valid_case(tmp_path, "A")
    masks = write_jsonl(tmp_path / "masks.jsonl", all_masks())
    out = tmp_path / "RUN.json"
    code, _, err = run(cli, capsys, score_argv(replace(case, masks=masks), out))
    assert code == 1
    assert_one_error_line(err)
    assert_not_created(*output_paths("score", out))


@pytest.mark.parametrize("flag", ["gold", "ids", "predictions", "masks"])
def test_cli_missing_file_exit_1(cli, capsys, tmp_path, flag):
    case = valid_case(tmp_path, "C")
    missing = tmp_path / "missing.jsonl"
    out = tmp_path / "RUN.json"
    code, _, err = run(cli, capsys, score_argv(replace(case, **{flag: missing}), out))
    assert code == 1
    assert err == f"error: missing.jsonl: {os.strerror(errno.ENOENT)}\n"
    assert_not_created(*output_paths("score", out))


@pytest.mark.parametrize("flag", ["gold", "ids", "predictions"])
def test_cli_input_is_folder_exit_1(cli, capsys, tmp_path, flag):
    case = valid_case(tmp_path, "C")
    folder = new_folder(tmp_path, "adir")
    out = tmp_path / "RUN.json"
    code, _, err = run(cli, capsys, score_argv(replace(case, **{flag: folder}), out))
    assert code == 1
    assert err == "error: adir: is a folder, not a file\n"
    assert_not_created(*output_paths("score", out))


# ---------------------------------------------------------------- OS errors
def os_error_cases(folder: Path) -> dict[str, tuple[OSError, str]]:
    """Name -> (error raised by the loader, exact stderr expected)."""
    return {
        "eacces": (PermissionError(errno.EACCES, "Permission denied", "secret.jsonl"),
                   "error: secret.jsonl: Permission denied\n"),
        "eperm": (PermissionError(errno.EPERM, "Permission denied", "secret.jsonl"),
                  "error: secret.jsonl: Permission denied\n"),
        "eacces_on_folder": (PermissionError(errno.EACCES, "Permission denied", str(folder)),
                             "error: adir: is a folder, not a file\n"),
        "eio_with_path": (OSError(errno.EIO, "I/O error", "x.jsonl"), "error: x.jsonl: I/O error\n"),
    }


@pytest.mark.parametrize("name", ["eacces", "eperm", "eacces_on_folder", "eio_with_path"])
def test_cli_os_errors_exit_1(cli, capsys, monkeypatch, tmp_path, name):
    case = valid_case(tmp_path, "A")
    error, expected = os_error_cases(new_folder(tmp_path, "adir"))[name]
    monkeypatch.setattr(cli, "load_bundle", raising(error))
    out = tmp_path / "RUN.json"
    code, _, err = run(cli, capsys, score_argv(case, out))
    assert code == 1
    assert err == expected
    assert_not_created(*output_paths("score", out))


def test_cli_os_error_without_filename_is_just_the_reason(cli, capsys, monkeypatch, tmp_path):
    case = valid_case(tmp_path, "A")
    monkeypatch.setattr(cli, "load_bundle", raising(OSError(errno.EIO, "I/O error")))
    code, _, err = run(cli, capsys, score_argv(case, tmp_path / "RUN.json"))
    assert code == 1
    assert err == "error: I/O error\n"


def test_cli_baseline_os_error_exit_1(cli, capsys, monkeypatch, tmp_path):
    case = valid_case(tmp_path, "A")
    error = PermissionError(errno.EACCES, "Permission denied", "secret.jsonl")
    monkeypatch.setattr(cli, "load_scored_gold", raising(error))
    out = tmp_path / "base.jsonl"
    code, _, err = run(cli, capsys, baseline_argv(case, out))
    assert code == 1
    assert err == "error: secret.jsonl: Permission denied\n"
    assert_not_created(out)


@pytest.mark.parametrize("command", ["score", "baseline"])
def test_cli_invalid_label_config_exit_1(cli, capsys, monkeypatch, tmp_path, command):
    bad = tmp_path / "bad.yaml"
    bad.write_text("punctuation: [O, COMMA", encoding="utf-8")
    monkeypatch.setattr(cli, "load_vocab", lambda: load_vocab(bad))
    case = valid_case(tmp_path, "A")
    out = tmp_path / OUT_NAMES[command]
    code, _, err = run(cli, capsys, command_argv(command, case, out))
    assert code == 1
    assert_one_error_line(err)
    assert "bad.yaml" in err
    assert_not_created(*output_paths(command, out))


# ---------------------------------------------------------------- output paths
@pytest.mark.parametrize("command", ["score", "baseline"])
def test_cli_missing_out_folder(cli, capsys, tmp_path, command):
    case = valid_case(tmp_path, "A")
    out = tmp_path / "missing" / OUT_NAMES[command]
    code, _, err = run(cli, capsys, command_argv(command, case, out))
    assert code == 1
    assert err == f"error: {out.name}: {os.strerror(errno.ENOENT)}\n"
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize(("command", "existing_name"),
                         [("score", "RUN.json"), ("score", "RUN.md"), ("baseline", "base.jsonl")],
                         ids=["score_json_exists", "score_md_exists", "baseline_out_exists"])
def test_cli_refuses_overwrite(cli, capsys, tmp_path, command, existing_name):
    case = valid_case(tmp_path, "A")
    out = tmp_path / OUT_NAMES[command]
    existing = tmp_path / existing_name
    existing.write_text("keep", encoding="utf-8")
    code, _, err = run(cli, capsys, command_argv(command, case, out))
    assert code == 1
    assert err == f"error: {existing.name}: already exists; refusing to overwrite\n"
    assert existing.read_text(encoding="utf-8") == "keep"
    assert_not_created(*(path for path in output_paths(command, out) if path != existing))


def test_cli_out_is_existing_folder(cli, capsys, tmp_path):
    case = valid_case(new_folder(tmp_path, "case"), "A")
    before = sorted(path.name for path in tmp_path.iterdir())
    code, stdout, err = run(cli, capsys, score_argv(case, tmp_path))
    assert code == 1
    assert_one_error_line(err)
    assert stdout == ""
    assert sorted(path.name for path in tmp_path.iterdir()) == before
    assert_not_created(tmp_path.with_suffix(".md"))


def test_cli_md_out_path_writes_nothing(cli, capsys, tmp_path):
    """EXPECTED TO FAIL: --out run.md makes both outputs the same file, so JSON lands in run.md (decision logged)."""
    case = valid_case(tmp_path, "A")
    out = tmp_path / "run.md"
    code, _, err = run(cli, capsys, score_argv(case, out))
    assert code == 1
    assert_one_error_line(err)
    assert not out.exists(), "run.md must not be written"


# ---------------------------------------------------------------- usage errors and subprocess
@pytest.mark.parametrize("argv", [
    [],
    ["score"],
    ["score", "--gold", "g.jsonl"],
    ["frobnicate"],
    ["baseline", "--gold", "g", "--ids", "i", "--out", "o", "--bogus"],
], ids=["no_args", "score_no_options", "score_missing_options", "unknown_command", "unknown_option"])
def test_cli_usage_error_exit_2(cli, capsys, argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_cli_subprocess_exit_codes(tmp_path):
    usage = run_script([])
    assert usage.returncode == 2
    assert "usage:" in usage.stderr
    assert "Traceback" not in usage.stderr

    bad = bad_length_case(new_folder(tmp_path, "bad"))
    failed = run_script(score_argv(bad, tmp_path / "bad" / "RUN.json"))
    assert failed.returncode == 1
    assert_one_error_line(failed.stderr)

    good = valid_case(new_folder(tmp_path, "good"), "A")
    out = tmp_path / "good" / "RUN.json"
    passed = run_script(score_argv(good, out), cwd=tmp_path)
    assert passed.returncode == 0, passed.stderr
    assert out.exists()
    assert passed.stderr == ""


# ---------------------------------------------------------------- sample IDs with line breaks
def line_break_case(kind: str, folder: Path) -> Case:
    gold, predictions = gold_rows("A"), perfect_preds("A")
    if kind == "duplicate_gold_id":
        for row in gold[:2]:
            row["sample_id"] = "a\nb"
    else:
        separator = "\n" if kind == "lf_prediction_id" else "\r"
        predictions.append(pred_row(f"zz{separator}INJECTED", "A", ["O"]))
    return write_case(folder, gold, SCORED, predictions)


@pytest.mark.parametrize("kind", ["lf_prediction_id", "cr_prediction_id", "duplicate_gold_id"])
def test_cli_sample_id_with_line_break_stays_one_line(cli, capsys, tmp_path, kind):
    """EXPECTED TO FAIL (at least LF and duplicate): inputs.py puts sample_id into the message without repr()."""
    case = line_break_case(kind, tmp_path)
    code, _, err = run(cli, capsys, score_argv(case, tmp_path / "RUN.json"))
    assert code == 1
    assert_one_error_line(err)


# ---------------------------------------------------------------- encoding and privacy
@pytest.mark.parametrize("command", ["score", "baseline"])
def test_cli_non_ascii_names_do_not_crash_stdout(tmp_path, command):
    """EXPECTED TO FAIL: the success print raises UnicodeEncodeError on a cp1252 console after the files are written."""
    case = write_case(new_folder(tmp_path, "case"), gold_rows("A"), SCORED, perfect_preds("A", model=NON_ASCII_MODEL))
    out = tmp_path / "case" / ("RUN.json" if command == "score" else NON_ASCII_FILE)
    result = run_script(command_argv(command, case, out), env=cp1252_env())
    assert "Traceback" not in result.stderr, result.stderr
    assert result.returncode == 0, result.stderr
    if command == "score":
        assert read_report(out)["model_name"] == NON_ASCII_MODEL


def test_cli_outputs_have_no_absolute_paths(cli, capsys, tmp_path):
    case = valid_case(tmp_path, "C")
    out = tmp_path / "RUN.json"
    code, _, _ = run(cli, capsys, score_argv(case, out))
    texts = [out.read_text(encoding="utf-8"), (tmp_path / "RUN.md").read_text(encoding="utf-8")]
    forbidden = [str(tmp_path), tmp_path.as_posix(), json.dumps(str(tmp_path))[1:-1]]  # last: JSON-escaped form
    assert code == 0
    for text in texts:
        for path_text in forbidden:
            assert path_text not in text
