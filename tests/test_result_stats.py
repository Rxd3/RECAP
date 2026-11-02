"""Tests for `recap result-stats`: the Holm, K-choice and label-audit files regenerate from committed inputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from reporting_fixtures import assert_one_error_line, run_main

from punct_restore.paths import REPO_ROOT
from punct_restore.reporting import label_audit, result_stats
from punct_restore.reporting.holm import holm_adjust

SCORES = REPO_ROOT / "results" / "scores"
# subcommand -> the committed file it regenerates (holm.json is new: it backs the "Multiple comparisons" section)
OUTPUTS = {
    "holm": "holm.json",
    "clusters": "clusters/clusters_summary.json",
    "label-audit": "gold_noise/label_audit.json",
}
CLUSTER_FILES = [f"clusters/compare_{mode}_k{k}.json" for mode in "ABC" for k in (100, 300)]
SHEET = "gold_noise/label_audit_sheet.csv"
Capsys = pytest.CaptureFixture[str]


def committed(rel: str) -> bytes:
    """The committed file as git stores it: LF line ends (a Windows checkout may hold CRLF)."""
    return (SCORES / rel).read_bytes().replace(b"\r\n", b"\n")


def run(capsys: Capsys, *argv: object) -> tuple[int, str]:
    code, out, err = run_main(result_stats, capsys, *argv)
    assert out in ("", out.strip() + "\n")  # at most one confirmation line on stdout
    return code, err


def holm_entries(text: str | bytes) -> list[dict[str, Any]]:
    families = json.loads(text)["families"]
    return [entry for family in families.values() for entry in family["comparisons"]]


def copy_inputs(root: Path, rels: list[str]) -> Path:
    """Copy committed files into a scores root under ``root``, keeping their relative paths."""
    for rel in rels:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SCORES / rel, target)
    return root


def holm_root(tmp_path: Path) -> Path:
    """A scores root holding exactly the 39 compare files that the committed holm.json lists."""
    rels = [entry["file"] for entry in holm_entries(committed("holm.json"))]
    return copy_inputs(tmp_path / "scores", rels)


def set_share(root: Path, rel: str, share: float) -> None:
    path = root / rel
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["blocks"][doc["primary_block"]]["difference_ci"]["share_b_better"] = share
    path.write_text(json.dumps(doc), encoding="utf-8")


# ---- byte for byte against the committed files ----------------------------------------------------------------


@pytest.mark.parametrize("command", OUTPUTS)
def test_regenerates_the_committed_file_byte_for_byte(command: str, tmp_path: Path, capsys: Capsys) -> None:
    out = tmp_path / "again.json"
    assert run(capsys, command, "--out", out) == (0, "")
    assert out.read_bytes() == committed(OUTPUTS[command])


@pytest.mark.parametrize("command", OUTPUTS)
def test_an_existing_output_is_never_replaced(command: str, tmp_path: Path, capsys: Capsys) -> None:
    out = tmp_path / "kept.json"
    out.write_text("keep me", encoding="utf-8")
    code, err = run(capsys, command, "--out", out)
    assert code == 1
    assert_one_error_line(err, "kept.json already exists")
    assert out.read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize("command", OUTPUTS)
def test_a_scores_folder_that_does_not_exist_gives_one_line_and_no_file(
    command: str, tmp_path: Path, capsys: Capsys
) -> None:
    out = tmp_path / "none.json"
    code, err = run(capsys, command, "--scores-root", tmp_path / "absent", "--out", out)
    assert code == 1 and not out.exists()
    assert_one_error_line(err, "No such file")
    assert str(tmp_path) not in err  # a file name only, never a folder


# ---- Holm ----------------------------------------------------------------------------------------------------


def test_holm_adjust_is_the_step_down_running_maximum() -> None:
    assert holm_adjust([0.01, 0.04, 0.03, 0.005]) == [0.03, 0.06, 0.06, 0.02]
    assert holm_adjust([0.5, 0.4]) == [0.8, 0.8]
    assert holm_adjust([0.9]) == [0.9]
    assert holm_adjust([0.6, 0.7]) == [1.0, 1.0]  # capped at 1


def test_the_holm_file_has_the_two_families_of_the_results_doc() -> None:
    families = json.loads(committed("holm.json"))["families"]
    assert {name: len(family["comparisons"]) for name, family in families.items()} == {
        "test_comparisons": 21, "ablations": 18}


def test_the_holm_survivors_and_p_values_are_the_ones_quoted_in_the_results_doc() -> None:
    families = json.loads(committed("holm.json"))["families"]
    tests = families["test_comparisons"]["comparisons"]
    lost = {(e["comparison"], e["mode"]): e["p"] for e in tests if not e["survives"]}
    assert lost == {("threshold - improved", "B"): 0.1, ("hidden-event LM - hmm", "C"): 0.028,
                    ("hidden-event LM - improved", "B"): 0.486}
    ablations = families["ablations"]["comparisons"]
    kept = {(e["group"], e["mode"]) for e in ablations if e["survives"]}
    assert kept == {(group, mode) for group in ("ngrams", "window2", "question") for mode in "AB"}
    mode_c = {e["group"]: e["p"] for e in ablations if e["mode"] == "C"}
    assert (mode_c["ngrams"], mode_c["window2"], mode_c["question"]) == (0.096, 0.042, 0.032)


def test_a_share_of_resamples_is_turned_into_a_two_sided_p(tmp_path: Path, capsys: Capsys) -> None:
    root = holm_root(tmp_path)
    rel = "threshold/tau_vs_improved_A_test.json"
    set_share(root, rel, 0.3)
    out = tmp_path / "holm.json"
    assert run(capsys, "holm", "--scores-root", root, "--out", out) == (0, "")
    entry = next(e for e in holm_entries(out.read_text(encoding="utf-8")) if e["file"] == rel)
    assert (entry["share_b_better"], entry["p"], entry["survives"]) == (0.3, 0.6, False)


# ---- clusters ------------------------------------------------------------------------------------------------


def set_differences(root: Path, k: int, value: float) -> None:
    """Make the primary difference of K in every mode ``value``."""
    for mode in "ABC":
        path = root / f"clusters/compare_{mode}_k{k}.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["blocks"][doc["primary_block"]]["difference"] = value
        path.write_text(json.dumps(doc), encoding="utf-8")


@pytest.mark.parametrize(("value_100", "value_300", "chosen"), [(0.5, 0.25, 100), (0.25, 0.5, 300), (0.25, 0.25, 100)])
def test_the_chosen_k_has_the_higher_mean_primary_difference_and_a_tie_goes_to_the_smaller(
    value_100: float, value_300: float, chosen: int, tmp_path: Path, capsys: Capsys
) -> None:
    root = copy_inputs(tmp_path / "scores", CLUSTER_FILES)
    set_differences(root, 100, value_100)
    set_differences(root, 300, value_300)
    out = tmp_path / "summary.json"
    assert run(capsys, "clusters", "--scores-root", root, "--out", out) == (0, "")
    selection = json.loads(out.read_text(encoding="utf-8"))["k_selection"]
    assert selection["chosen_k"] == chosen
    assert selection["mean_primary_difference"] == {"100": value_100, "300": value_300}


def test_the_compare_file_hash_does_not_depend_on_the_line_ends(tmp_path: Path, capsys: Capsys) -> None:
    root = copy_inputs(tmp_path / "scores", CLUSTER_FILES)
    for rel in CLUSTER_FILES:  # a CRLF checkout of the same files
        path = root / rel
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    out = tmp_path / "summary.json"
    assert run(capsys, "clusters", "--scores-root", root, "--out", out) == (0, "")
    assert out.read_bytes() == committed("clusters/clusters_summary.json")


# ---- label audit ---------------------------------------------------------------------------------------------


def test_wilson_interval_matches_the_committed_audit_values() -> None:
    assert label_audit.wilson(221, 223) == {"k": 221, "n": 223, "p": 0.991031, "low": 0.967895, "high": 0.997537}
    assert label_audit.wilson(0, 0) == {"k": 0, "n": 0, "p": None, "low": None, "high": None}


def test_the_audit_sections_split_the_sheet_by_row_number(tmp_path: Path, capsys: Capsys) -> None:
    out = tmp_path / "audit.json"
    assert run(capsys, "label-audit", "--out", out) == (0, "")
    doc = json.loads(out.read_text(encoding="utf-8"))
    hand, agent, both = (doc[k]["pooled"] for k in ("hand_checked_rows_51_100", "agent_only_rows_1_50", "all_100"))
    assert (hand["rows"], agent["rows"], both["rows"]) == (50, 50, 100)
    assert both["our_marks"] == hand["our_marks"] + agent["our_marks"]
    assert doc["agent_only_rows_1_50_by_design_removals_not_missed"]["pooled"]["missed_marks"] == 0


# ---- errors: a missing and a damaged input, one line each ---------------------------------------------------


def test_a_missing_compare_file_is_one_line_naming_it(tmp_path: Path, capsys: Capsys) -> None:
    root = holm_root(tmp_path)
    (root / "hidden_event_lm" / "hel_vs_hmm_C_test.json").unlink()
    code, err = run(capsys, "holm", "--scores-root", root, "--out", tmp_path / "h.json")
    assert code == 1 and not (tmp_path / "h.json").exists()
    assert_one_error_line(err, "error: hel_vs_hmm_C_test.json: ")
    assert str(tmp_path) not in err


@pytest.mark.parametrize("damage, reason", [
    ("{not json", "not valid JSON"),
    (json.dumps({"primary_block": "x", "blocks": {"x": {"difference": 0.1}}}), "difference_ci"),
    (json.dumps({"primary_block": "x", "blocks": {"x": {"difference": 0.1, "difference_ci": {"share_b_better": 1.5}}}}),
     "share_b_better is not between 0 and 1"),
])
def test_a_damaged_compare_file_is_one_line_naming_it(
    damage: str, reason: str, tmp_path: Path, capsys: Capsys
) -> None:
    root = holm_root(tmp_path)
    (root / "ablations" / "mode_B" / "full_vs_improved_minus_clause.json").write_text(damage, encoding="utf-8")
    code, err = run(capsys, "holm", "--scores-root", root, "--out", tmp_path / "h.json")
    assert code == 1
    assert_one_error_line(err, "error: full_vs_improved_minus_clause.json: ", reason)


def test_a_missing_cluster_compare_file_is_one_line_naming_it(tmp_path: Path, capsys: Capsys) -> None:
    root = copy_inputs(tmp_path / "scores", [rel for rel in CLUSTER_FILES if not rel.endswith("C_k300.json")])
    code, err = run(capsys, "clusters", "--scores-root", root, "--out", tmp_path / "c.json")
    assert code == 1 and not (tmp_path / "c.json").exists()
    assert_one_error_line(err, "error: compare_C_k300.json: ")


def test_a_cluster_compare_file_without_the_class_blocks_is_one_line(tmp_path: Path, capsys: Capsys) -> None:
    root = copy_inputs(tmp_path / "scores", CLUSTER_FILES)
    path = root / "clusters" / "compare_B_k100.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    del doc["classes"]["QUESTION"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    code, err = run(capsys, "clusters", "--scores-root", root, "--out", tmp_path / "c.json")
    assert code == 1
    assert_one_error_line(err, "error: compare_B_k100.json: ", "classes.QUESTION")


def test_a_missing_audit_sheet_is_one_line_naming_it(tmp_path: Path, capsys: Capsys) -> None:
    code, err = run(capsys, "label-audit", "--scores-root", tmp_path, "--out", tmp_path / "a.json")
    assert code == 1 and not (tmp_path / "a.json").exists()
    assert_one_error_line(err, "error: label_audit_sheet.csv: ")


@pytest.mark.parametrize("edit, reason", [
    (lambda text: text.replace(",5,0,0,0,,", ",5,x,0,0,,", 1), "row 1: wrong_marks is not a whole number"),
    (lambda text: text.replace(",5,0,0,0,,", ",5,,0,0,,", 1), "row 1: wrong_marks is empty"),
    (lambda text: text.replace(",rare,", ",odd,", 1), "row 1: stratum is not rare or random"),
    (lambda text: text.split("\n", 2)[0] + "\n" + text.split("\n", 2)[2], "has 99 rows, not 100"),
    (lambda text: text.replace("our_marks,", "marks,", 1), "missing column our_marks"),
])
def test_a_damaged_audit_sheet_is_one_line_naming_it(
    edit: Any, reason: str, tmp_path: Path, capsys: Capsys
) -> None:
    root = tmp_path / "scores"
    sheet = root / SHEET
    sheet.parent.mkdir(parents=True)
    text = (SCORES / SHEET).read_text(encoding="utf-8").replace("\r\n", "\n")
    sheet.write_text(edit(text), encoding="utf-8", newline="\n")
    code, err = run(capsys, "label-audit", "--scores-root", root, "--out", tmp_path / "a.json")
    assert code == 1 and not (tmp_path / "a.json").exists()
    assert_one_error_line(err, "error: label_audit_sheet.csv: ", reason)


def test_an_audit_sheet_that_is_not_utf8_is_one_line(tmp_path: Path, capsys: Capsys) -> None:
    root = tmp_path / "scores"
    (root / "gold_noise").mkdir(parents=True)
    (root / SHEET).write_bytes(b"row,sample_id\n1,\xff\xfe\n")
    code, err = run(capsys, "label-audit", "--scores-root", root, "--out", tmp_path / "a.json")
    assert code == 1
    assert_one_error_line(err, "error: label_audit_sheet.csv: not valid UTF-8")
