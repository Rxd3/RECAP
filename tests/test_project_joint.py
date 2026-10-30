"""scripts/project_joint.py: mode C predictions projected to mode A (the joint-vs-punctuation-only study, H1)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
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


# --- --from B: mode B predictions projected with the approved D5 mapping read from configs/label_sets.yaml ---

B_PREDS = {"s1": ["EXCLAMATION", "COLON", "SEMICOLON", "DASH"], "s2": ["O", "COMMA", "QUESTION"]}
B_ALL = ["O", "COMMA", "PERIOD", "QUESTION", "EXCLAMATION", "COLON", "SEMICOLON", "DASH"]
B_ALL_AS_A = ["O", "COMMA", "PERIOD", "QUESTION", "PERIOD", "COMMA", "PERIOD", "COMMA"]
Change = Callable[[dict[str, Any]], dict[str, Any]]


def _from_b(case: Case, rows: list[Any] | None = None, flags: Sequence[str] = ("--from", "B")) -> list[str]:
    """Write mode B predictions (B_PREDS by default) as the case's predictions; the argv gets ``flags``."""
    default = [pred_row(sid, "B", labels, MODEL) for sid, labels in B_PREDS.items()]
    _preds(case.preds, default if rows is None else rows)
    return [*case.argv(), *flags]


def _written(case: Case) -> list[dict[str, Any]]:
    return [json.loads(line) for line in case.out.read_text(encoding="utf-8").splitlines()]


def _remap(**edits: str) -> Change:
    return lambda projection: {**projection, "mapping": {**projection["mapping"], **edits}}


def _patch_projection(monkeypatch: pytest.MonkeyPatch, change: Change) -> None:
    real = project_joint.load_label_sets

    def patched() -> dict[str, Any]:
        label_sets = real()
        return {**label_sets, "projection": change(label_sets["projection"])}

    monkeypatch.setattr(project_joint, "load_label_sets", patched)


def test_from_b_projects_all_eight_labels_with_the_approved_mapping(case: Case, capsys) -> None:
    argv = _from_b(case, [pred_row("s1", "B", B_ALL, MODEL)])
    code, out, err = _run(capsys, argv)
    assert (code, err) == (0, "") and out == "projected 1 predictions to a_pred.jsonl\n"
    assert _written(case) == [{"sample_id": "s1", "model_name": "crf_joint_from_B", "label_mode": "A",
                               "predicted_labels": B_ALL_AS_A}]
    assert list(_written(case)[0]) == ["sample_id", "model_name", "label_mode", "predicted_labels"]


def test_from_c_default_keeps_the_joint_c_suffix_and_explicit_from_c_is_the_same(case: Case, capsys) -> None:
    assert _run(capsys, [*case.argv(), "--from", "C"])[0] == 0
    assert {row["model_name"] for row in _written(case)} == {"crf_joint_joint_C"}


def test_from_b_manifest_passes_the_evaluator_check_against_the_mode_a_gold(case: Case, capsys) -> None:
    assert _run(capsys, _from_b(case))[0] == 0
    assert [row["predicted_labels"] for row in _written(case)] == [["PERIOD", "COMMA", "PERIOD", "COMMA"],
                                                                   ["O", "COMMA", "QUESTION"]]
    vocab = load_vocab()
    check_manifest(case.man, load_bundle(case.gold, case.ids, case.out, None, vocab), vocab)
    manifest = json.loads(case.man.read_text(encoding="utf-8"))
    assert manifest["input"]["sha256"] == sha256_file(case.gold)
    assert manifest["model_name"] == "crf_joint_from_B" and manifest["label_mode"] == "A"


BAD_B_RECORDS = [
    ("mode C", [pred_row("s1", "C", ["O|LOWER"], MODEL)], "line 1: label_mode is 'C', expected 'B'"),
    ("mode A", [pred_row("s1", "A", ["O"], MODEL)], "line 1: label_mode is 'A', expected 'B'"),
    ("joint label", [pred_row("s1", "B", ["O|LOWER"], MODEL)], "line 1: a mode B label must be one of the mode B"),
    ("unknown label", [pred_row("s1", "B", ["DASHH"], MODEL)], "line 1: a mode B label must be one of the mode B"),
    ("non-string label", [pred_row("s1", "B", [["O"]], MODEL)], "line 1: a mode B label must be one of the mode B"),
]


@pytest.mark.parametrize(("records", "expected"), [pytest.param(r, e, id=i) for i, r, e in BAD_B_RECORDS])
def test_from_b_refuses_other_modes_and_unknown_labels_with_one_error_line(
        case: Case, capsys, tmp_path: Path, records: list[Any], expected: str) -> None:
    err = _assert_refused(capsys, tmp_path, _from_b(case, records))
    assert expected in err
    assert not case.out.exists() and not case.man.exists()


@pytest.mark.parametrize("flags", [(), ("--from", "C")], ids=["default", "explicit"])
def test_from_c_refuses_mode_b_input(case: Case, capsys, tmp_path: Path, flags: tuple[str, ...]) -> None:
    err = _assert_refused(capsys, tmp_path, _from_b(case, None, flags))
    assert "line 1: label_mode is 'B', expected 'C'" in err
    assert not case.out.exists() and not case.man.exists()


UNAPPROVED = "the B -> A projection is not approved"
BAD_PROJECTIONS = [
    pytest.param(lambda p: {**p, "approved": False}, UNAPPROVED, id="unapproved"),
    pytest.param(lambda p: None, UNAPPROVED, id="projection-is-null"),
    pytest.param(lambda p: [p], UNAPPROVED, id="projection-is-a-list"),
    pytest.param(lambda p: {**p, "approved": "true"}, UNAPPROVED, id="approved-not-a-bool"),
    pytest.param(lambda p: {k: v for k, v in p.items() if k != "approved"}, UNAPPROVED, id="no-approved-key"),
    pytest.param(lambda p: {**p, "mapping": {k: v for k, v in p["mapping"].items() if k != "SEMICOLON"}},
                 "must cover exactly the mode B labels", id="missing-b-label"),
    pytest.param(_remap(BOGUS="O"), "must cover exactly the mode B labels", id="extra-key"),
    pytest.param(_remap(DASH="BOGUS"), "must map to mode A labels", id="value-outside-mode-a"),
]


@pytest.mark.parametrize(("change", "expected"), BAD_PROJECTIONS)
def test_from_b_refuses_an_unapproved_or_incomplete_projection_in_the_config(
        case: Case, capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: Change, expected: str) -> None:
    _patch_projection(monkeypatch, change)
    err = _assert_refused(capsys, tmp_path, _from_b(case))
    assert expected in err and "label_sets.yaml" in err
    assert not case.out.exists() and not case.man.exists()


def test_from_b_follows_the_mapping_in_the_config_not_a_built_in_one(
        case: Case, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_projection(monkeypatch, _remap(DASH="PERIOD", COLON="QUESTION"))
    assert _run(capsys, _from_b(case))[0] == 0
    assert [row["predicted_labels"] for row in _written(case)] == [["PERIOD", "QUESTION", "PERIOD", "PERIOD"],
                                                                   ["O", "COMMA", "QUESTION"]]


def test_from_option_is_documented_and_limited_to_b_and_c(case: Case, capsys) -> None:
    with pytest.raises(SystemExit) as stop:
        project_joint.main(["--help"])
    assert stop.value.code == 0
    assert "--from {C,B}" in capsys.readouterr().out
    with pytest.raises(SystemExit) as stop:
        project_joint.main([*case.argv(), "--from", "A"])
    assert stop.value.code == 2 and "--from" in capsys.readouterr().err
