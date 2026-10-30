"""Tests for scripts/reporting/length_position.py on a small hand-computed mode A case (no real data)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from eval_fixtures import gold_row, mask_row, pred_row, write_ids, write_jsonl
from reporting_fixtures import assert_one_error_line, run_main

from punct_restore.evaluation.inputs import sha256_file
from scripts.reporting import length_position

# (sample_id, tokens, gold marks, predicted marks); every other position is O
CASE: list[tuple[str, int, dict[int, str], dict[int, str]]] = [
    ("r1", 10, {4: "COMMA", 9: "QUESTION"}, {4: "COMMA", 9: "PERIOD"}),
    ("r2", 10, {9: "PERIOD"}, {9: "PERIOD"}),
    ("r3", 11, {2: "COMMA", 5: "COMMA", 10: "PERIOD"}, {2: "COMMA", 10: "PERIOD"}),
    ("r4", 25, {10: "QUESTION", 24: "QUESTION"}, {3: "COMMA", 10: "QUESTION", 24: "QUESTION"}),
    ("r5", 26, {}, {25: "PERIOD"}),
    ("r6", 50, {49: "PERIOD"}, {10: "COMMA", 49: "PERIOD"}),
    ("r7", 51, {20: "QUESTION", 50: "QUESTION"}, {50: "QUESTION"}),
]
# Hand computation, micro F1 over COMMA/PERIOD/QUESTION = 2*TP / (gold non-O + predicted non-O):
#   1-10  : r1+r2   gold 3, predicted 3, TP 2 (COMMA@4, PERIOD@9 of r2)          -> 2/3  = 0.666667
#   11-25 : r3+r4   gold 5, predicted 5, TP 4 (COMMA@2, PERIOD@10, QUESTION x2)  -> 4/5  = 0.8
#   26-50 : r5+r6   gold 1, predicted 3, TP 1 (PERIOD@49 of r6)                  -> 2/4  = 0.5
#   > 50  : r7      gold 2, predicted 1, TP 1 (QUESTION@50)                      -> 2/3  = 0.666667
# QUESTION recall: final tokens r1@9 (missed), r4@24, r7@50 -> 2 of 3; inside r4@10 (hit), r7@20 (missed) -> 1 of 2
LABELS = ["1-10", "11-25", "26-50", ">50"]
BOUNDS = [(1, 10), (11, 25), (26, 50), (51, None)]
COUNTS = [2, 2, 2, 1]
MICRO_F1 = [0.666667, 0.8, 0.5, 0.666667]
MICRO_PCT = ["66.67", "80.00", "50.00", "66.67"]


@dataclass(frozen=True)
class Case:
    gold: Path
    ids: Path
    predictions: dict[str, Path]


def labels(length: int, marks: dict[int, str]) -> list[str]:
    return [marks.get(i, "O") for i in range(length)]


def write_case(folder: Path, rows: Sequence[Any] = CASE, *, unscored: Sequence[Any] = (), mode: str = "A") -> Case:
    """gold.jsonl (+ `unscored` rows missing from the ID list), the ID list, and two prediction files."""
    folder.mkdir(parents=True, exist_ok=True)
    everything = [*rows, *unscored]

    def gold(sid: str, length: int, marks: dict[int, str]) -> dict[str, Any]:
        return gold_row(sid, mode, [f"w{i}" for i in range(length)], labels(length, marks), ["LOWER"] * length)

    write_jsonl(folder / "gold.jsonl", [gold(sid, n, g) for sid, n, g, _ in everything])
    write_ids(folder / "test.ids", [sid for sid, *_ in rows])
    write_jsonl(folder / "first.jsonl", [pred_row(sid, mode, labels(n, p)) for sid, n, _, p in everything])
    write_jsonl(folder / "perfect.jsonl",
                [pred_row(sid, mode, labels(n, g), model="perfect_model") for sid, n, g, _ in everything])
    return Case(folder / "gold.jsonl", folder / "test.ids",
                {"first": folder / "first.jsonl", "perfect": folder / "perfect.jsonl"})


def argv(case: Case, out: Path, *extra: str) -> list[str]:
    specs = [f"{name}={path}" for name, path in case.predictions.items()]
    return ["--gold", str(case.gold), "--ids", str(case.ids), "--predictions", *specs, *extra, "--out", str(out)]


def report(case: Case, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    out = tmp_path / "out" / "lp.json"
    out.parent.mkdir(exist_ok=True)
    code, _, err = run_main(length_position, capsys, *argv(case, out))
    assert (code, err) == (0, "")
    assert out.read_text(encoding="utf-8").endswith("}\n")
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture
def case(tmp_path: Path) -> Case:
    return write_case(tmp_path / "case")


# ---------------------------------------------------------------- the numbers
def test_micro_f1_per_length_bucket_matches_the_hand_computation(
    case: Case, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    buckets = report(case, tmp_path, capsys)["models"]["first"]["length_buckets"]
    assert [b["label"] for b in buckets] == LABELS
    assert [(b["min_tokens"], b["max_tokens"]) for b in buckets] == BOUNDS
    assert [b["sequences"] for b in buckets] == COUNTS
    assert [b["micro_f1"] for b in buckets] == MICRO_F1
    assert [b["micro_f1_pct"] for b in buckets] == MICRO_PCT


def test_question_recall_at_the_final_position_and_inside(
    case: Case, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recall = report(case, tmp_path, capsys)["models"]["first"]["question_recall"]
    assert recall == {"final": {"support": 3, "recall": 0.666667, "recall_pct": "66.67"},
                      "inside": {"support": 2, "recall": 0.5, "recall_pct": "50.00"}}


def test_every_prediction_file_gets_its_own_entry_in_the_given_order(
    case: Case, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models = report(case, tmp_path, capsys)["models"]
    assert list(models) == ["first", "perfect"]
    perfect = models["perfect"]
    assert perfect["model_name"] == "perfect_model" and models["first"]["model_name"] == "m"
    assert [b["micro_f1"] for b in perfect["length_buckets"]] == [1.0] * 4
    assert perfect["question_recall"]["final"]["recall_pct"] == "100.00"
    assert perfect["question_recall"]["inside"]["support"] == 2  # the gold counts do not depend on the model


def test_inputs_are_recorded_with_their_sha256(
    case: Case, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    doc = report(case, tmp_path, capsys)
    assert doc["label_mode"] == "A"
    assert doc["inputs"]["gold"]["sha256"] == sha256_file(case.gold)
    assert doc["inputs"]["ids"]["sha256"] == sha256_file(case.ids)
    assert doc["inputs"]["masks"] is None
    for name, path in case.predictions.items():
        assert doc["models"][name]["predictions"]["sha256"] == sha256_file(path)
    assert "Users" not in json.dumps(doc) and str(tmp_path) not in json.dumps(doc)  # repo-relative, no absolute paths


def test_gold_records_outside_the_id_list_are_not_counted(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    extra = [("x1", 3, {2: "PERIOD"}, {})]
    case = write_case(tmp_path / "case", unscored=extra)
    buckets = report(case, tmp_path, capsys)["models"]["first"]["length_buckets"]
    assert [b["sequences"] for b in buckets] == COUNTS
    assert [b["micro_f1"] for b in buckets] == MICRO_F1


def test_an_empty_bucket_has_a_zero_count_and_no_score(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = [("a", 5, {4: "PERIOD"}, {4: "PERIOD"}), ("b", 60, {59: "PERIOD"}, {})]
    buckets = report(write_case(tmp_path / "case", rows), tmp_path, capsys)["models"]["first"]["length_buckets"]
    assert [b["sequences"] for b in buckets] == [1, 0, 0, 1]
    assert [(b["micro_f1"], b["micro_f1_pct"]) for b in buckets] == [(1.0, "100.00"), (None, None), (None, None),
                                                                      (0.0, "0.00")]


def test_without_a_gold_question_the_recall_is_null(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = [("a", 3, {2: "PERIOD"}, {2: "QUESTION"})]
    recall = report(write_case(tmp_path / "case", rows), tmp_path, capsys)["models"]["first"]["question_recall"]
    assert recall == {"final": {"support": 0, "recall": None, "recall_pct": None},
                      "inside": {"support": 0, "recall": None, "recall_pct": None}}


@pytest.mark.parametrize(("length", "index"), [(1, 0), (10, 0), (11, 1), (25, 1), (26, 2), (50, 2), (51, 3),
                                               (10_000, 3)])
def test_bucket_edges(length: int, index: int) -> None:
    assert length_position.bucket_index(length) == index


def test_bucket_edges_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = [(f"s{n}", n, {}, {}) for n in (10, 11, 25, 26, 50, 51)]
    buckets = report(write_case(tmp_path / "case", rows), tmp_path, capsys)["models"]["first"]["length_buckets"]
    assert [b["sequences"] for b in buckets] == [1, 2, 2, 1]


# ---------------------------------------------------------------- refusals (one line, exit 1, nothing written)
def refuse(capsys: pytest.CaptureFixture[str], args: list[str], out: Path, needle: str) -> None:
    code, stdout, err = run_main(length_position, capsys, *args)
    assert (code, stdout) == (1, "")
    assert_one_error_line(err, needle)
    assert "/" not in err and "\\" not in err  # file names only, never a folder
    assert not out.exists() or out.read_text(encoding="utf-8") == "keep me"


def test_a_mode_b_gold_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    case, out = write_case(tmp_path / "case", mode="B"), tmp_path / "o.json"
    refuse(capsys, argv(case, out), out, "mode B: this report handles mode A only")


def test_masks_are_refused_with_the_evaluators_message_for_mode_a(
    case: Case, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    masks, out = write_jsonl(tmp_path / "masks.jsonl", [mask_row("r1", [])]), tmp_path / "o.json"
    refuse(capsys, argv(case, out, "--masks", str(masks)), out, "casing masks apply to mode C only")


def test_a_prediction_file_missing_a_scored_id_is_refused(
    case: Case, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_jsonl(case.predictions["first"], [pred_row("r1", "A", labels(10, {}))])
    out = tmp_path / "o.json"
    refuse(capsys, argv(case, out), out, "no prediction for 6 scored ID(s)")


def test_a_prediction_label_outside_mode_a_is_refused(
    case: Case, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [pred_row(sid, "A", labels(n, {0: "DASH"})) for sid, n, _, _ in CASE]
    write_jsonl(case.predictions["first"], rows)
    out = tmp_path / "o.json"
    refuse(capsys, argv(case, out), out, "predicted_labels label 'DASH' is not one of")


def test_a_missing_gold_file_is_refused_by_name_only(
    case: Case, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case.gold.unlink()
    out = tmp_path / "o.json"
    refuse(capsys, argv(case, out), out, "gold.jsonl")


def test_an_existing_output_is_never_replaced(case: Case, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "o.json"
    out.write_text("keep me", encoding="utf-8")
    refuse(capsys, argv(case, out), out, "o.json already exists")


def test_an_output_in_a_missing_folder_is_refused(case: Case, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "nope" / "o.json"
    refuse(capsys, argv(case, out), out, "folder nope not found")


@pytest.mark.parametrize("which", ["gold", "ids", "predictions"])
def test_an_output_that_is_an_input_is_refused_and_the_input_survives(
    which: str, case: Case, capsys: pytest.CaptureFixture[str]
) -> None:
    target = {"gold": case.gold, "ids": case.ids, "predictions": case.predictions["first"]}[which]
    before = sha256_file(target)
    code, _, err = run_main(length_position, capsys, *argv(case, target))
    assert code == 1
    assert_one_error_line(err, "same file")
    assert sha256_file(target) == before


@pytest.mark.parametrize("spec", ["no_equals_sign", "=missing_name.jsonl", "name="])
def test_a_malformed_name_path_pair_is_a_usage_error(spec: str, case: Case, tmp_path: Path) -> None:
    args = ["--gold", str(case.gold), "--ids", str(case.ids), "--predictions", spec, "--out", str(tmp_path / "o.json")]
    with pytest.raises(SystemExit) as raised:
        length_position.main(args)
    assert raised.value.code == 2


def test_a_repeated_model_name_is_a_usage_error(case: Case, tmp_path: Path) -> None:
    path = case.predictions["first"]
    args = ["--gold", str(case.gold), "--ids", str(case.ids), "--predictions", f"same={path}", f"same={path}",
            "--out", str(tmp_path / "o.json")]
    with pytest.raises(SystemExit) as raised:
        length_position.main(args)
    assert raised.value.code == 2


def test_no_arguments_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as raised:
        length_position.main([])
    assert raised.value.code == 2
