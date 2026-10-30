"""scripts/factor_predictions.py: mode A punctuation + CASE casing predictions -> mode C predictions."""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from eval_fixtures import (
    CASE,
    PUNCT,
    SCORED,
    assert_one_error_line,
    perfect_preds,
    pred_row,
    valid_case,
    write_jsonl,
)

from punct_restore.data.labeling import load_label_sets
from punct_restore.evaluation import load_vocab
from punct_restore.evaluation.inputs import load_bundle, sha256_file
from punct_restore.evaluation.manifest import check_manifest
from scripts import factor_predictions

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "factor_predictions.py"


def _punct_rows(ids: tuple[str, ...] = SCORED) -> list[dict[str, Any]]:
    return [pred_row(sid, "A", PUNCT["A"][sid], "crf_a") for sid in ids]


def _case_rows(ids: tuple[str, ...] = SCORED) -> list[dict[str, Any]]:
    return [pred_row(sid, "CASE", CASE[sid], "crf_case") for sid in ids]


@dataclass(frozen=True)
class Setup:
    punct: Path
    case: Path
    gold: Path
    ids: Path
    masks: Path
    out: Path
    man: Path

    def argv(self, *extra: str, **override: Path) -> list[str]:
        paths = {"punct": self.punct, "case": self.case, "gold": self.gold, "out": self.out,
                 "man": self.man, **override}
        return ["--punct", str(paths["punct"]), "--case", str(paths["case"]), "--gold", str(paths["gold"]),
                "--out", str(paths["out"]), "--manifest-out", str(paths["man"]), *extra]


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    (tmp_path / "eval").mkdir()
    gold = valid_case(tmp_path / "eval", "C")
    assert gold.masks is not None
    return Setup(punct=write_jsonl(tmp_path / "punct.jsonl", _punct_rows()),
                 case=write_jsonl(tmp_path / "case.jsonl", _case_rows()),
                 gold=gold.gold, ids=gold.ids, masks=gold.masks,
                 out=tmp_path / "c_pred.jsonl", man=tmp_path / "c_pred.manifest.json")


def _run(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str, str]:
    code = factor_predictions.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _refused(capsys: pytest.CaptureFixture[str], tmp_path: Path, argv: list[str]) -> str:
    code, out, err = _run(capsys, argv)
    assert code == 1 and out == ""
    assert_one_error_line(err)
    assert str(tmp_path) not in err, "an error must name files by name only"
    return err


def _written(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_writes_joint_predictions_in_punct_order_and_prints_a_summary(setup: Setup, capsys) -> None:
    write_jsonl(setup.punct, list(reversed(_punct_rows())))
    code, out, err = _run(capsys, setup.argv())
    assert (code, err) == (0, "")
    assert out == "factored 2 predictions to c_pred.jsonl\n"
    assert _written(setup.out) == list(reversed(perfect_preds("C", SCORED, "factored")))
    assert all(list(row) == ["sample_id", "model_name", "label_mode", "predicted_labels"]
               for row in _written(setup.out))


def test_model_name_option_names_the_output_and_the_manifest(setup: Setup, capsys) -> None:
    assert _run(capsys, setup.argv("--model-name", "crf_factored"))[0] == 0
    assert {row["model_name"] for row in _written(setup.out)} == {"crf_factored"}
    manifest = json.loads(setup.man.read_text(encoding="utf-8"))
    assert manifest["model_name"] == "crf_factored" and manifest["label_mode"] == "C"
    assert manifest["ordered_labels"] == load_label_sets()["C"]


def test_manifest_passes_the_evaluator_check_against_the_scored_gold(setup: Setup, capsys) -> None:
    assert _run(capsys, setup.argv())[0] == 0
    vocab = load_vocab()
    bundle = load_bundle(setup.gold, setup.ids, setup.out, setup.masks, vocab)
    check_manifest(setup.man, bundle, vocab)
    manifest = json.loads(setup.man.read_text(encoding="utf-8"))
    assert manifest["input"]["sha256"] == sha256_file(setup.gold)
    assert manifest["model_name"] == "factored" and manifest["label_mode"] == "C"


def test_blank_lines_are_skipped_but_counted_in_line_numbers(setup: Setup, capsys, tmp_path: Path) -> None:
    write_jsonl(setup.case, ["", json.dumps(_case_rows()[0]), "  ", "{not json"])
    err = _refused(capsys, tmp_path, setup.argv())
    assert "case.jsonl line 4: not valid JSON" in err
    assert not setup.out.exists() and not setup.man.exists()


READ_ERRORS = [
    ("not JSON", "{not json", "line 1: not valid JSON"),
    ("not an object", "[1, 2]", "line 1: not a JSON object"),
    ("nested too deeply", "[" * 5000 + "]" * 5000, "line 1: not valid JSON"),
]


@pytest.mark.parametrize("which", ["punct", "case"])
@pytest.mark.parametrize(("line", "expected"), [pytest.param(r, e, id=i) for i, r, e in READ_ERRORS])
def test_unreadable_lines_name_the_file_and_the_line(
        setup: Setup, capsys, tmp_path: Path, which: str, line: str, expected: str) -> None:
    write_jsonl(getattr(setup, which), [line])
    err = _refused(capsys, tmp_path, setup.argv())
    assert f"{which}.jsonl {expected}" in err
    assert not setup.out.exists() and not setup.man.exists()


@pytest.mark.parametrize("which", ["punct", "case"])
def test_invalid_utf8_and_empty_files_name_the_file(setup: Setup, capsys, tmp_path: Path, which: str) -> None:
    path = getattr(setup, which)
    path.write_bytes(b'{"sample_id": "s1"}\n\xe9\xff\n')
    assert f"{which}.jsonl: not valid UTF-8" in _refused(capsys, tmp_path, setup.argv())
    write_jsonl(path, ["", "  "])
    assert f"{which}.jsonl: no predictions found" in _refused(capsys, tmp_path, setup.argv())
    assert not setup.out.exists() and not setup.man.exists()


def _swap_modes(setup: Setup) -> None:
    write_jsonl(setup.punct, _case_rows())
    write_jsonl(setup.case, _punct_rows())


MISMATCHES = [
    pytest.param(lambda s: write_jsonl(s.case, _case_rows(("s1",))), "'s2'", id="id-missing-from-case"),
    pytest.param(lambda s: write_jsonl(s.punct, _punct_rows(("s1",))), "'s2'", id="id-missing-from-punct"),
    pytest.param(lambda s: write_jsonl(s.case, [*_case_rows(), _case_rows()[0]]), "duplicate sample_id 's1'",
                 id="duplicate-id"),
    pytest.param(lambda s: write_jsonl(s.case, [pred_row("s1", "CASE", ["LOWER"]), *_case_rows(("s2",))]),
                 "sample_id 's1': 3 punct labels but 1 case labels", id="length-mismatch"),
    pytest.param(_swap_modes, "label_mode is 'CASE', expected 'A'", id="modes-swapped"),
    pytest.param(lambda s: write_jsonl(s.punct, [pred_row("s1", "B", ["O"]), *_punct_rows(("s2",))]),
                 "label_mode is 'B', expected 'A'", id="punct-is-mode-b"),
    pytest.param(lambda s: write_jsonl(s.case, [pred_row("s1", "CASE", ["LOWER", "MIXED", "LOWER"]),
                                                *_case_rows(("s2",))]),
                 "unknown mode CASE label 'MIXED'", id="unknown-case-label"),
]


@pytest.mark.parametrize(("change", "expected"), MISMATCHES)
def test_inconsistent_predictions_give_one_error_line_and_write_nothing(
        setup: Setup, capsys, tmp_path: Path, change: Any, expected: str) -> None:
    change(setup)
    assert expected in _refused(capsys, tmp_path, setup.argv())
    assert not setup.out.exists() and not setup.man.exists()


@pytest.mark.parametrize("target", ["out", "man"])
def test_existing_output_is_refused_before_anything_is_read(
        setup: Setup, capsys, tmp_path: Path, target: str) -> None:
    existing = {"out": setup.out, "man": setup.man}[target]
    existing.write_text("keep me", encoding="utf-8")
    setup.punct.write_bytes(b"\xe9\xff")  # would be a different error if it were read first
    err = _refused(capsys, tmp_path, setup.argv())
    assert "already exists" in err
    assert existing.read_text(encoding="utf-8") == "keep me"
    assert [setup.out.exists(), setup.man.exists()].count(True) == 1


@pytest.mark.parametrize(
    "clash",
    [("out", "man"), ("out", "punct"), ("out", "case"), ("out", "gold"), ("man", "punct"), ("man", "case"),
     ("man", "gold")],
)
def test_output_paths_may_not_collide_with_each_other_or_with_inputs(
        setup: Setup, capsys, tmp_path: Path, clash: tuple[str, str]) -> None:
    first, second = clash
    paths = {"punct": setup.punct, "case": setup.case, "gold": setup.gold, "out": setup.out, "man": setup.man}
    before = {name: path.read_bytes() for name, path in paths.items() if path.exists()}
    err = _refused(capsys, tmp_path, setup.argv(**{first: paths[second]}))
    assert "are the same file" in err or "already exists" in err
    assert {name: path.read_bytes() for name, path in paths.items() if path.exists()} == before


def test_output_folder_is_refused(setup: Setup, capsys, tmp_path: Path) -> None:
    setup.out.mkdir()
    assert "is a folder" in _refused(capsys, tmp_path, setup.argv())


@pytest.mark.parametrize("missing", ["punct", "case", "gold"])
def test_missing_input_file_is_a_one_line_error_naming_the_file_only(
        setup: Setup, capsys, tmp_path: Path, missing: str) -> None:
    err = _refused(capsys, tmp_path, setup.argv(**{missing: tmp_path / f"absent_{missing}.jsonl"}))
    assert f"absent_{missing}.jsonl" in err
    assert not setup.out.exists() and not setup.man.exists()


def test_os_error_is_reported_as_one_line_without_the_full_path(
        setup: Setup, capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def full_disk(*args: Any, **kwargs: Any) -> None:
        raise OSError(28, "No space left on device", str(setup.out))

    monkeypatch.setattr(factor_predictions, "write_prediction_outputs", full_disk)
    assert _refused(capsys, tmp_path, setup.argv()) == "error: c_pred.jsonl: No space left on device\n"


def test_missing_required_option_is_a_usage_error(setup: Setup, capsys) -> None:
    with pytest.raises(SystemExit) as stop:
        factor_predictions.main(["--punct", str(setup.punct)])
    assert stop.value.code == 2
    assert "--case" in capsys.readouterr().err


def test_stdout_without_reconfigure_still_prints_the_summary(
        setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    assert factor_predictions.main(setup.argv()) == 0
    assert buffer.getvalue() == "factored 2 predictions to c_pred.jsonl\n"


def test_script_runs_standalone_without_PYTHONPATH_or_an_installed_package(setup: Setup) -> None:
    done = subprocess.run([sys.executable, "-I", str(SCRIPT), *setup.argv()], capture_output=True, text=True,
                          check=False)
    assert (done.returncode, done.stderr) == (0, "")
    assert done.stdout.strip() == "factored 2 predictions to c_pred.jsonl"
    assert setup.out.is_file() and setup.man.is_file()


def test_script_failure_is_exit_1_with_one_error_line(setup: Setup) -> None:
    setup.out.write_text("keep me", encoding="utf-8")
    done = subprocess.run([sys.executable, "-I", str(SCRIPT), *setup.argv()], capture_output=True, text=True,
                          check=False)
    assert done.returncode == 1 and done.stdout == ""
    assert_one_error_line(done.stderr)


def test_script_has_no_personal_paths() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert not re.search(r"[A-Za-z]:[/\\]", source)
