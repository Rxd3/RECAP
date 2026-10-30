"""Command line tests for the score extensions: --bootstrap (grouped CIs) and --manifest.

main(argv) runs in-process through the `cli` fixture; cases come from cli_analysis_cases.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from cli_analysis_cases import (MANIFEST_NAME, assert_nothing_written, assert_one_line, build_case, bundle_of,
                                read_json, valid_manifest, write_manifest)
from eval_fixtures import Case, assert_one_error_line, cli_argv, run_cli

from punct_restore.evaluation import BootstrapSpec, evaluate, to_markdown
from punct_restore.evaluation.inputs import Bundle, Vocab, sha256_file


# ---------------------------------------------------------------- score --bootstrap
def test_score_with_bootstrap_writes_ci(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                        vocab: Vocab, grouped_case: Case) -> None:
    out = tmp_path / "RUN.json"
    argv = cli_argv("score", grouped_case, out, "--bootstrap", "120", "--seed", "5")

    code, stdout, err = run_cli(cli, capsys, argv)

    expected = evaluate(bundle_of(grouped_case, vocab), vocab, BootstrapSpec(120, 5))
    markdown = out.with_suffix(".md").read_text(encoding="utf-8")
    assert (code, err) == (0, "")
    assert read_json(out) == expected
    assert markdown == to_markdown(expected)
    assert "| Macro F1 95% CI |" in markdown
    assert_one_line(stdout)


@pytest.mark.parametrize(("extra", "fragment"),
                         [(["--bootstrap", "99"], "must be 100 to 100000"),
                          (["--bootstrap", "0"], "must be 100 to 100000"),
                          (["--bootstrap", "100001"], "must be 100 to 100000"),
                          (["--bootstrap", "abc"], "invalid integer value"),
                          (["--bootstrap", "200", "--seed", "x"], "invalid int value")],
                         ids=["below_minimum", "zero", "above_cap", "not_an_integer", "seed_not_an_integer"])
def test_score_bootstrap_usage_errors(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                      grouped_case: Case, extra: list[str], fragment: str) -> None:
    out = tmp_path / "RUN.json"

    with pytest.raises(SystemExit) as raised:
        cli.main(cli_argv("score", grouped_case, out, *extra))

    err = capsys.readouterr().err
    assert raised.value.code == 2
    assert "usage:" in err and fragment in err
    assert_nothing_written(out)


def test_score_bootstrap_minimum_is_accepted(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                             grouped_case: Case) -> None:
    out = tmp_path / "RUN.json"

    code, _, err = run_cli(cli, capsys, cli_argv("score", grouped_case, out, "--bootstrap", "100"))

    assert (code, err) == (0, "")
    assert out.exists()


def test_score_bootstrap_needs_group_ids(cli: ModuleType, capsys: pytest.CaptureFixture[str],
                                         tmp_path: Path) -> None:
    case = build_case(tmp_path / "case", "A", no_group=("s3",))
    out = tmp_path / "RUN.json"

    code, stdout, err = run_cli(cli, capsys, cli_argv("score", case, out, "--bootstrap", "100"))

    assert code == 1
    assert_one_error_line(err)
    assert "'s3'" in err
    assert stdout == ""
    assert_nothing_written(out)
    plain = tmp_path / "plain.json"
    code, _, err = run_cli(cli, capsys, cli_argv("score", case, plain))
    assert (code, err) == (0, "")
    assert plain.exists()


# ---------------------------------------------------------------- score --manifest
def test_score_with_manifest(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path, vocab: Vocab,
                             grouped_case: Case) -> None:
    manifest = write_manifest(grouped_case, json.dumps(valid_manifest(grouped_case, vocab)))
    out = tmp_path / "RUN.json"

    code, _, err = run_cli(cli, capsys, cli_argv("score", grouped_case, out, "--manifest", str(manifest)))

    identity = {"path": f"{manifest.parent.name}/{MANIFEST_NAME}", "sha256": sha256_file(manifest)}
    assert (code, err) == (0, "")
    assert read_json(out)["inputs"]["manifest"] == identity
    assert identity["sha256"] in out.with_suffix(".md").read_text(encoding="utf-8")


def _other_model(valid: dict[str, Any]) -> str:
    return json.dumps({**valid, "model_name": "other"})


def _multiline_model(valid: dict[str, Any]) -> str:
    return json.dumps({**valid, "model_name": "m\nx"})


def _gold_hash_of_predictions(valid: dict[str, Any]) -> str:
    return json.dumps({**valid, "input": {"sha256": valid["predictions_sha256"]}})


def _not_an_object(valid: dict[str, Any]) -> str:
    return "[]"


MISMATCHES = {"other_model": _other_model, "multiline_model": _multiline_model,
              "wrong_input_hash": _gold_hash_of_predictions, "not_an_object": _not_an_object}


@pytest.mark.parametrize("break_manifest", list(MISMATCHES.values()), ids=list(MISMATCHES))
def test_score_manifest_mismatch_writes_nothing(cli: ModuleType, capsys: pytest.CaptureFixture[str],
                                                tmp_path: Path, vocab: Vocab, grouped_case: Case,
                                                break_manifest: Callable[[dict[str, Any]], str]) -> None:
    manifest = write_manifest(grouped_case, break_manifest(valid_manifest(grouped_case, vocab)))
    out = tmp_path / "RUN.json"

    code, stdout, err = run_cli(cli, capsys, cli_argv("score", grouped_case, out, "--manifest", str(manifest)))

    assert code == 1
    assert_one_error_line(err)
    assert MANIFEST_NAME in err
    assert stdout == ""
    assert_nothing_written(out)


def test_score_manifest_missing_file(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                     grouped_case: Case) -> None:
    missing = grouped_case.predictions.parent / "nope.json"
    out = tmp_path / "RUN.json"

    code, _, err = run_cli(cli, capsys, cli_argv("score", grouped_case, out, "--manifest", str(missing)))

    assert code == 1
    assert err == f"error: file or folder not found: {missing}\n"
    assert_nothing_written(out)


def test_bootstrap_cap_is_accepted(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                   grouped_case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[BootstrapSpec | None] = []

    def fake_evaluate(bundle: Bundle, vocabulary: Vocab, bootstrap: BootstrapSpec | None = None) -> dict[str, Any]:
        recorded.append(bootstrap)
        return evaluate(bundle, vocabulary)  # no resampling: the cap itself is only parsed

    monkeypatch.setattr(cli, "evaluate", fake_evaluate)
    out = tmp_path / "RUN.json"

    code, _, err = run_cli(cli, capsys, cli_argv("score", grouped_case, out, "--bootstrap", "100000"))

    assert (code, err) == (0, "")
    assert recorded == [BootstrapSpec(100_000, 2026)]
    assert out.exists()


def _evaluate_must_not_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise AssertionError("evaluate must not run")


def test_manifest_checked_before_bootstrap(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                           vocab: Vocab, grouped_case: Case,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    ungrouped = build_case(tmp_path / "case_no_group", "A", no_group=("s3",))  # s3 is scored, has no group_id
    manifest = write_manifest(ungrouped, _other_model(valid_manifest(ungrouped, vocab)))
    out = tmp_path / "RUN.json"

    code, stdout, err = run_cli(cli, capsys,
                                cli_argv("score", ungrouped, out, "--bootstrap", "100", "--manifest", str(manifest)))

    assert code == 1
    assert_one_error_line(err)
    assert MANIFEST_NAME in err and "model_name" in err
    assert "group_id" not in err
    assert stdout == ""
    assert_nothing_written(out)
    monkeypatch.setattr(cli, "evaluate", _evaluate_must_not_run)
    wrong = write_manifest(grouped_case, _other_model(valid_manifest(grouped_case, vocab)))
    second = tmp_path / "RUN2.json"

    code, _, err = run_cli(cli, capsys, cli_argv("score", grouped_case, second, "--manifest", str(wrong)))

    assert code == 1
    assert_one_error_line(err)
    assert MANIFEST_NAME in err and "model_name" in err
    assert_nothing_written(second)
