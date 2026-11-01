"""Command line tests for the compare and errors commands of punct_restore.cli.evaluate_models.

main(argv) runs in-process through the `cli` fixture; only the working-directory test starts a subprocess.
Cases come from cli_analysis_cases (8 gold rows over 4 groups; model_a and model_b make different mistakes).
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from cli_analysis_cases import (
    assert_nothing_written,
    assert_one_line,
    assert_usage_error,
    bundle_of,
    compare_argv,
    predictions_b,
    read_json,
)
from eval_fixtures import SCRIPT_PATH, Case, assert_one_error_line, cli_argv, run_cli

from punct_restore.evaluation import BootstrapSpec, compare, comparison_markdown, error_examples, errors_markdown
from punct_restore.evaluation.inputs import Vocab


# ---------------------------------------------------------------- compare
def test_compare_writes_json_and_markdown(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                          vocab: Vocab, grouped_case: Case) -> None:
    out = tmp_path / "cmp.json"

    code, stdout, err = run_cli(cli, capsys, compare_argv(grouped_case, out))

    expected = compare(bundle_of(grouped_case, vocab), bundle_of(grouped_case, vocab, predictions_b(grouped_case)),
                       vocab)
    markdown = out.with_suffix(".md").read_text(encoding="utf-8")
    assert (code, err) == (0, "")
    assert read_json(out) == expected
    assert markdown == comparison_markdown(expected)
    assert "95% CI" not in markdown
    assert_one_line(stdout)
    assert "model_b vs model_a" in stdout


def test_compare_with_bootstrap(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path, vocab: Vocab,
                                grouped_case: Case) -> None:
    out = tmp_path / "cmp.json"

    code, _, err = run_cli(cli, capsys, compare_argv(grouped_case, out, "--bootstrap", "100", "--seed", "3"))

    expected = compare(bundle_of(grouped_case, vocab), bundle_of(grouped_case, vocab, predictions_b(grouped_case)),
                       vocab, BootstrapSpec(100, 3))
    report = read_json(out)
    assert (code, err) == (0, "")
    assert report == expected
    assert all("difference_ci" in block for block in report["blocks"].values())
    assert "95% CI of B" in out.with_suffix(".md").read_text(encoding="utf-8")


def test_compare_mode_c_needs_masks(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                    mode_c_case: Case) -> None:
    refused = tmp_path / "refused.json"

    code, _, err = run_cli(cli, capsys, compare_argv(replace(mode_c_case, masks=None), refused))

    assert code == 1
    assert_one_error_line(err)
    assert "masks" in err
    assert_nothing_written(refused)
    out = tmp_path / "cmp.json"
    code, _, err = run_cli(cli, capsys, compare_argv(mode_c_case, out))
    assert (code, err) == (0, "")
    assert read_json(out)["primary_block"] == "joint/unmasked"


@pytest.mark.parametrize("existing_name", ["cmp.json", "cmp.md"])
def test_compare_refuses_overwrite(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                   grouped_case: Case, existing_name: str) -> None:
    existing = tmp_path / existing_name
    existing.write_text("keep", encoding="utf-8")
    other = tmp_path / ({"cmp.json": "cmp.md", "cmp.md": "cmp.json"}[existing_name])

    code, _, err = run_cli(cli, capsys, compare_argv(grouped_case, tmp_path / "cmp.json"))

    assert code == 1
    assert err == f"error: {existing.name}: already exists; refusing to overwrite\n"
    assert existing.read_text(encoding="utf-8") == "keep"
    assert not other.exists()


USAGE_TAILS = {"missing_b": ["--predictions-a", "{a}"],
               "predictions_instead_of_a": ["--predictions", "{a}", "--predictions-b", "{b}"],
               "bootstrap_below_minimum": ["--predictions-a", "{a}", "--predictions-b", "{b}",
                                           "--bootstrap", "50"]}


@pytest.mark.parametrize("tail", list(USAGE_TAILS.values()), ids=list(USAGE_TAILS))
def test_compare_usage_errors(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                              grouped_case: Case, tail: list[str]) -> None:
    out = tmp_path / "cmp.json"
    flags = [flag.format(a=grouped_case.predictions, b=predictions_b(grouped_case)) for flag in tail]

    assert_usage_error(cli, capsys, cli_argv("compare", grouped_case, out, *flags, predictions=False))

    assert_nothing_written(out)


def test_compare_out_must_be_json(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                  grouped_case: Case) -> None:
    out = tmp_path / "cmp.txt"

    code, stdout, err = run_cli(cli, capsys, compare_argv(grouped_case, out))

    assert code == 1
    assert_one_error_line(err)
    assert stdout == ""
    assert sorted(path.name for path in tmp_path.iterdir()) == ["case"]


# ---------------------------------------------------------------- errors
@pytest.mark.parametrize("extra, options",
                         [([], {}),
                          (["--block", "punctuation/non_final", "--per-pair", "1", "--seed", "9"],
                           {"block": "punctuation/non_final", "per_pair": 1, "seed": 9})],
                         ids=["defaults", "all_options"])
def test_errors_writes_json_and_markdown(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                         vocab: Vocab, grouped_case: Case, extra: list[str],
                                         options: dict[str, Any]) -> None:
    out = tmp_path / "err.json"

    code, stdout, err = run_cli(cli, capsys, cli_argv("errors", grouped_case, out, *extra))

    expected = error_examples(bundle_of(grouped_case, vocab), vocab, **options)
    assert (code, err) == (0, "")
    assert expected["errors_total"] > 0
    assert read_json(out) == expected
    assert out.with_suffix(".md").read_text(encoding="utf-8") == errors_markdown(expected)
    assert_one_line(stdout)


def test_errors_mode_c_joint_block(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                   vocab: Vocab, mode_c_case: Case) -> None:
    out = tmp_path / "err.json"

    code, _, err = run_cli(cli, capsys, cli_argv("errors", mode_c_case, out, "--block", "joint/unmasked"))

    assert (code, err) == (0, "")
    assert read_json(out) == error_examples(bundle_of(mode_c_case, vocab), vocab, "joint/unmasked")


@pytest.mark.parametrize("block", ["nope", "joint/unmasked"])
def test_errors_bad_block_exit_1(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                 grouped_case: Case, block: str) -> None:
    out = tmp_path / "err.json"

    code, stdout, err = run_cli(cli, capsys, cli_argv("errors", grouped_case, out, "--block", block))

    assert code == 1
    assert_one_error_line(err)
    assert block in err
    assert stdout == ""
    assert_nothing_written(out)


@pytest.mark.parametrize("extra", [["--per-pair", "0"], ["--per-pair", "x"], ["--seed", "y"]],
                         ids=["per_pair_zero", "per_pair_not_a_number", "seed_not_a_number"])
def test_errors_usage_errors(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                             grouped_case: Case, extra: list[str]) -> None:
    out = tmp_path / "err.json"

    assert_usage_error(cli, capsys, cli_argv("errors", grouped_case, out, *extra))

    assert_nothing_written(out)


def test_errors_per_pair_minimum_is_accepted(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                             grouped_case: Case) -> None:
    out = tmp_path / "err.json"

    code, _, err = run_cli(cli, capsys, cli_argv("errors", grouped_case, out, "--per-pair", "1"))

    assert (code, err) == (0, "")
    assert out.exists()


def test_errors_refuses_overwrite(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                  grouped_case: Case) -> None:
    out = tmp_path / "err.json"
    existing = out.with_suffix(".md")
    existing.write_text("keep", encoding="utf-8")

    code, _, err = run_cli(cli, capsys, cli_argv("errors", grouped_case, out))

    assert code == 1
    assert err == f"error: {existing.name}: already exists; refusing to overwrite\n"
    assert existing.read_text(encoding="utf-8") == "keep"
    assert not out.exists()


# ---------------------------------------------------------------- real process
@pytest.mark.parametrize("command", ["compare", "errors"])
def test_new_commands_from_another_cwd(tmp_path: Path, grouped_case: Case, command: str) -> None:
    out = tmp_path / f"{command}.json"
    argv = compare_argv(grouped_case, out) if command == "compare" else cli_argv("errors", grouped_case, out)

    result = subprocess.run([sys.executable, str(SCRIPT_PATH), *argv], cwd=tmp_path, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=60, check=False)

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert out.exists()
