"""scripts/project_joint.py: mode C predictions projected to mode A (the joint-vs-punctuation-only study, H1)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from eval_fixtures import (
    assert_one_error_line,
    gold_row,
    pred_row,
    write_ids,
    write_jsonl,
)

from punct_restore.evaluation import load_vocab
from punct_restore.evaluation.inputs import load_bundle, sha256_file
from punct_restore.evaluation.manifest import check_manifest
from scripts import project_joint

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "project_joint.py"
MODEL = "crf_joint"
GOLD_ROWS = [
    ("s1", ["well", "i", "think", "so"], ["COMMA", "O", "O", "PERIOD"]),
    ("s2", ["how", "are", "you"], ["O", "O", "QUESTION"]),
]
C_LABELS = {
    "s1": ["COMMA|FIRST_UPPER", "O|LOWER", "O|LOWER", "PERIOD|ALL_UPPER"],
    "s2": ["O|LOWER", "O|LOWER", "QUESTION|LOWER"],
}


@dataclass(frozen=True)
class Case:
    preds: Path
    gold: Path
    ids: Path
    out: Path
    man: Path

    def argv(self, **override: Path) -> list[str]:
        paths = {"preds": self.preds, "gold": self.gold, "out": self.out, "man": self.man, **override}
        return ["--predictions", str(paths["preds"]), "--gold", str(paths["gold"]),
                "--out", str(paths["out"]), "--manifest-out", str(paths["man"])]


def _gold(path: Path) -> Path:
    rows = [{**gold_row(sid, "A", tokens, punct, ["LOWER"] * len(tokens)), "split": "validation"}
            for sid, tokens, punct in GOLD_ROWS]
    return write_jsonl(path, rows)


def _preds(path: Path, rows: Iterable[dict[str, Any] | str] | None = None) -> Path:
    default = [pred_row(sid, "C", labels, MODEL) for sid, labels in C_LABELS.items()]
    return write_jsonl(path, default if rows is None else rows)


@pytest.fixture
def case(tmp_path: Path) -> Case:
    return Case(preds=_preds(tmp_path / "c_pred.jsonl"), gold=_gold(tmp_path / "gold_a.jsonl"),
                ids=write_ids(tmp_path / "val.ids", C_LABELS), out=tmp_path / "a_pred.jsonl",
                man=tmp_path / "a_pred.manifest.json")


def _run(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str, str]:
    code = project_joint.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _assert_refused(capsys: pytest.CaptureFixture[str], tmp_path: Path, argv: list[str]) -> str:
    code, out, err = _run(capsys, argv)
    assert code == 1 and out == ""
    assert_one_error_line(err)
    assert str(tmp_path) not in err, "an error must name files by name only"
    return err


def test_projection_keeps_punctuation_renames_model_and_sets_mode_a(case: Case, capsys) -> None:
    code, out, err = _run(capsys, case.argv())
    assert (code, err) == (0, "")
    assert out == "projected 2 predictions to a_pred.jsonl\n"
    rows = [json.loads(line) for line in case.out.read_text(encoding="utf-8").splitlines()]
    assert [list(row) for row in rows] == [["sample_id", "model_name", "label_mode", "predicted_labels"]] * 2
    assert rows[0] == {"sample_id": "s1", "model_name": "crf_joint_joint_C", "label_mode": "A",
                       "predicted_labels": ["COMMA", "O", "O", "PERIOD"]}
    assert rows[1]["predicted_labels"] == ["O", "O", "QUESTION"]


def test_all_punctuation_and_casing_combinations_project_to_the_punctuation_part(case: Case, capsys) -> None:
    labels = ["O|LOWER", "COMMA|FIRST_UPPER", "PERIOD|ALL_UPPER"]
    _preds(case.preds, [pred_row("s1", "C", labels, MODEL)])
    assert _run(capsys, case.argv())[0] == 0
    row = json.loads(case.out.read_text(encoding="utf-8"))
    assert row["predicted_labels"] == ["O", "COMMA", "PERIOD"]


def test_manifest_passes_the_evaluator_check_against_the_scored_gold(case: Case, capsys) -> None:
    assert _run(capsys, case.argv())[0] == 0
    vocab = load_vocab()
    bundle = load_bundle(case.gold, case.ids, case.out, None, vocab)
    check_manifest(case.man, bundle, vocab)
    manifest = json.loads(case.man.read_text(encoding="utf-8"))
    assert manifest["input"]["sha256"] == sha256_file(case.gold)
    assert manifest["model_name"] == "crf_joint_joint_C" and manifest["label_mode"] == "A"


def test_blank_lines_are_skipped_but_still_counted_in_line_numbers(case: Case, capsys, tmp_path: Path) -> None:
    rows = ["", json.dumps(pred_row("s1", "C", C_LABELS["s1"], MODEL)), "  ",
            json.dumps(pred_row("s2", "A", ["O"], MODEL))]
    _preds(case.preds, rows)
    err = _assert_refused(capsys, tmp_path, case.argv())
    assert "c_pred.jsonl line 4: label_mode is 'A', expected 'C'" in err
    assert not case.out.exists()


BAD_RECORDS = [
    ("mode A", [pred_row("s1", "A", ["O"], MODEL)], "line 1: label_mode is 'A', expected 'C'"),
    ("mode B", [pred_row("s1", "B", ["O"], MODEL)], "line 1: label_mode is 'B', expected 'C'"),
    ("label without a bar", [pred_row("s1", "C", ["COMMA"], MODEL)], "line 1: a mode C label must be PUNCT|CASE"),
    ("unknown joint label", [pred_row("s1", "C", ["O|MIXED"], MODEL)], "line 1: a mode C label must be PUNCT|CASE"),
    ("non-string label", [pred_row("s1", "C", [["O|LOWER"]], MODEL)], "line 1: a mode C label must be PUNCT|CASE"),
    ("mixed model names", [pred_row("s1", "C", ["O|LOWER"], "m1"), pred_row("s2", "C", ["O|LOWER"], "m2")],
     "line 2: model_name 'm2' differs from 'm1'"),
    ("duplicate sample_id", [pred_row("s1", "C", ["O|LOWER"], MODEL), pred_row("s1", "C", ["O|LOWER"], MODEL)],
     "line 2: duplicate sample_id 's1'"),
    ("empty sample_id", [pred_row("", "C", ["O|LOWER"], MODEL)], "line 1: sample_id must be a non-empty string"),
    ("empty model_name", [pred_row("s1", "C", ["O|LOWER"], "")], "line 1: model_name must be a non-empty string"),
    ("empty label list", [pred_row("s1", "C", [], MODEL)], "line 1: predicted_labels must be a non-empty list"),
    ("labels not a list", [{**pred_row("s1", "C", [], MODEL), "predicted_labels": "O|LOWER"}],
     "line 1: predicted_labels must be a non-empty list"),
    ("missing label_mode", [{"sample_id": "s1", "model_name": MODEL, "predicted_labels": ["O|LOWER"]}],
     "line 1: label_mode is None, expected 'C'"),
    ("not JSON", ["{not json"], "line 1: not valid JSON"),
    ("not an object", ["[1, 2]"], "line 1: not a JSON object"),
    ("empty file", [], "c_pred.jsonl: no predictions found"),
    ("only blank lines", ["", "   "], "c_pred.jsonl: no predictions found"),
]


@pytest.mark.parametrize(("records", "expected"), [pytest.param(r, e, id=i) for i, r, e in BAD_RECORDS])
def test_invalid_predictions_give_one_error_line_and_write_nothing(
        case: Case, capsys, tmp_path: Path, records: list[Any], expected: str) -> None:
    _preds(case.preds, records)
    err = _assert_refused(capsys, tmp_path, case.argv())
    assert expected in err
    assert not case.out.exists() and not case.man.exists()


def test_invalid_utf8_predictions_give_one_error_line(case: Case, capsys, tmp_path: Path) -> None:
    case.preds.write_bytes(b'{"sample_id": "s1"}\n\xe9\xff\n')
    err = _assert_refused(capsys, tmp_path, case.argv())
    assert "c_pred.jsonl: not valid UTF-8" in err
    assert not case.out.exists() and not case.man.exists()


@pytest.mark.parametrize("target", ["out", "man"])
def test_existing_output_is_never_overwritten(case: Case, capsys, tmp_path: Path, target: str) -> None:
    existing = {"out": case.out, "man": case.man}[target]
    existing.write_text("keep me", encoding="utf-8")
    err = _assert_refused(capsys, tmp_path, case.argv())
    assert "already exists" in err
    assert existing.read_text(encoding="utf-8") == "keep me"
    assert [case.out.exists(), case.man.exists()].count(True) == 1


@pytest.mark.parametrize("clash", [("out", "man"), ("out", "preds"), ("man", "preds"), ("out", "gold"), ("man", "gold")])
def test_output_paths_may_not_collide_with_each_other_or_with_inputs(
        case: Case, capsys, tmp_path: Path, clash: tuple[str, str]) -> None:
    first, second = clash
    paths = {"preds": case.preds, "gold": case.gold, "out": case.out, "man": case.man}
    before = {name: path.read_bytes() for name, path in paths.items() if path.exists()}
    err = _assert_refused(capsys, tmp_path, case.argv(**{first: paths[second]}))
    assert "are the same file" in err
    assert {name: path.read_bytes() for name, path in paths.items() if path.exists()} == before


def test_output_folder_is_refused(case: Case, capsys, tmp_path: Path) -> None:
    case.out.mkdir()
    err = _assert_refused(capsys, tmp_path, case.argv())
    assert "is a folder" in err


@pytest.mark.parametrize("missing", ["preds", "gold"])
def test_missing_input_file_is_a_one_line_error_naming_the_file_only(
        case: Case, capsys, tmp_path: Path, missing: str) -> None:
    absent = tmp_path / f"absent_{missing}.jsonl"
    err = _assert_refused(capsys, tmp_path, case.argv(**{missing: absent}))
    assert f"absent_{missing}.jsonl" in err
    assert not case.out.exists() and not case.man.exists()


def test_os_error_is_reported_as_one_line_without_the_full_path(
        case: Case, capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def full_disk(*args: Any, **kwargs: Any) -> None:
        raise OSError(28, "No space left on device", str(case.out))

    monkeypatch.setattr(project_joint, "write_prediction_outputs", full_disk)
    err = _assert_refused(capsys, tmp_path, case.argv())
    assert err == "error: a_pred.jsonl: No space left on device\n"


def test_missing_required_option_is_a_usage_error(case: Case, capsys) -> None:
    with pytest.raises(SystemExit) as stop:
        project_joint.main(["--predictions", str(case.preds)])
    assert stop.value.code == 2
    assert "--gold" in capsys.readouterr().err


def test_stdout_without_reconfigure_still_prints_the_summary(
        case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    assert project_joint.main(case.argv()) == 0
    assert buffer.getvalue() == "projected 2 predictions to a_pred.jsonl\n"


def test_script_runs_standalone_without_an_installed_package(case: Case) -> None:
    done = subprocess.run([sys.executable, "-I", str(SCRIPT), *case.argv()], capture_output=True, text=True,
                          check=False)
    assert (done.returncode, done.stderr) == (0, "")
    assert done.stdout.strip() == "projected 2 predictions to a_pred.jsonl"
    assert case.out.is_file() and case.man.is_file()
