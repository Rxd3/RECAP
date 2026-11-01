"""Tests for punct_restore.reporting.asr_noise_table using synthetic score fixtures only (no real data).

The numbers asserted here are worked out by hand from the formulas in asr_noise_table_fixtures.py (see the
comments), so a wrong mean, drop or v2 - v1 difference fails a test. Findings that must follow the data are in
test_asr_noise_findings.py.
"""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from asr_noise_table_fixtures import (
    patch_json,
    report,
    row,
    run_main,
    section,
    set_f1,
    v1_run,
    v2_run,
    write_asr_scores,
)
from reporting_fixtures import assert_one_error_line, dump

EN, MINUS = "–", "−"  # U+2013 en dash, U+2212 minus sign


@pytest.fixture
def asr_scores(tmp_path: Path) -> Path:
    return write_asr_scores(tmp_path / "scores")


def test_asr_noise_table_end_to_end(
    asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from punct_restore.reporting import asr_noise_table

    json_out = tmp_path / "summary.json"
    md_out = tmp_path / "ASR_NOISE_V2.md"
    code = run_main(asr_scores, json_out, md_out)
    captured = capsys.readouterr()
    assert (code, captured.err) == (0, "")
    assert json_out.is_file() and md_out.is_file()

    summary = json.loads(json_out.read_text(encoding="utf-8"))
    md_text = md_out.read_text(encoding="utf-8")

    # Markdown is generated purely from the JSON summary dict
    assert asr_noise_table.render_markdown(summary) == md_text

    # Mode A CRF improved at WER 0.05: seeds 0.484, 0.485, 0.4865 -> mean 0.4851666... (48.52%), min 48.40, max 48.65
    imp_a_05 = summary["v2"]["modes"]["A"]["models"]["crf_improved"]["by_wer"]["0.05"]
    assert imp_a_05["mean"] == pytest.approx(0.4851666666666667, abs=1e-15)
    assert imp_a_05["mean_pct"] == "48.52"
    assert imp_a_05["min_pct"] == "48.40"
    assert imp_a_05["max_pct"] == "48.65"
    assert imp_a_05["cell"] == f"48.52 (48.40{EN}48.65)"

    assert "### Mode A (`punctuation/all`)" in md_text
    assert "### Mode B (`punctuation/all`)" in md_text
    assert "### Mode C (`joint/unmasked`)" in md_text
    assert f"| CRF improved | 50.00 | 48.52 (48.40{EN}48.65) |" in md_text

    assert "## 1. Method" in md_text
    assert "## 2. Achieved Edit Mix (`v2` and `v1`)" in md_text
    assert "## 3. Primary Macro F1 by Mode (`v2`)" in md_text
    assert "## 4. Comparison: `v1` vs `v2` at 15% WER" in md_text
    assert "## 5. Findings" in md_text
    assert "Does the improved CRF still degrade faster than the HMM?" in md_text
    assert "Is the ranking stable across seeds?" in md_text
    assert "How do `v1` and `v2` differ?" in md_text
    assert "## 6. Limits" in md_text


def test_mode_table_cells_and_drops_match_hand_computed_values(asr_scores: Path, tmp_path: Path) -> None:
    # v2 F1 = clean - step * drop + seed offset (-0.001, 0, +0.0015): the seed mean adds +0.000167.
    # Drop (0% -> 15%) = 3 * drop - 0.000167 -> 0.58, 2.38, 4.48 points in every mode.
    _, md = report(asr_scores, tmp_path)
    mode_a = section(md, "### Mode A (`punctuation/all`)")
    assert row(mode_a, "HMM") == [
        "HMM", "35.00", f"34.82 (34.70{EN}34.95)", f"34.62 (34.50{EN}34.75)", f"34.42 (34.30{EN}34.55)",
        f"{MINUS}0.58"]
    assert row(mode_a, "CRF basic") == [
        "CRF basic", "43.00", f"42.22 (42.10{EN}42.35)", f"41.42 (41.30{EN}41.55)", f"40.62 (40.50{EN}40.75)",
        f"{MINUS}2.38"]
    assert row(mode_a, "CRF improved") == [
        "CRF improved", "50.00", f"48.52 (48.40{EN}48.65)", f"47.02 (46.90{EN}47.15)", f"45.52 (45.40{EN}45.65)",
        f"{MINUS}4.48"]
    assert row(section(md, "### Mode C (`joint/unmasked`)"), "CRF improved")[-1] == f"{MINUS}4.48"


def test_v2_minus_v1_cells_match_hand_computed_values(asr_scores: Path, tmp_path: Path) -> None:
    # v1 at 15%: seeds clean - 3 * (drop - 0.001) + (0, 0, +0.0015), so its range is 0.15 and its mean is +0.0005.
    # v2 mean - v1 mean = -0.003333 in every cell -> -0.33 points.
    _, md = report(asr_scores, tmp_path)
    comparison = section(md, "## 4. Comparison: `v1` vs `v2` at 15% WER")
    assert row(comparison, "A", "HMM") == [
        "A", "HMM", "35.00", f"34.75 (34.70{EN}34.85)", f"34.42 (34.30{EN}34.55)", f"{MINUS}0.33", "0.15", "0.25"]
    assert row(comparison, "A", "CRF improved") == [
        "A", "CRF improved", "50.00", f"45.85 (45.80{EN}45.95)", f"45.52 (45.40{EN}45.65)", f"{MINUS}0.33",
        "0.15", "0.25"]
    assert row(comparison, "C", "CRF basic")[5] == f"{MINUS}0.33"


def test_v2_minus_v1_cell_has_the_right_sign_and_a_gain_prints_a_plus(asr_scores: Path, tmp_path: Path) -> None:
    for seed in (2026, 2027, 2028):
        set_f1(v2_run(asr_scores, "0.15", seed), "C", "hmm", 0.40)  # above its clean 0.37
        set_f1(v1_run(asr_scores, seed), "C", "hmm", 0.30)
    _, md = report(asr_scores, tmp_path)
    assert row(section(md, "## 4. Comparison: `v1` vs `v2` at 15% WER"), "C", "HMM")[5] == "+10.00"
    # the mean score rose 3 points above clean: the drop column shows a gain, not "minus minus 3"
    assert row(section(md, "### Mode C (`joint/unmasked`)"), "HMM")[-1] == "+3.00"


def test_means_are_exact_and_rounded_once_at_display(asr_scores: Path, tmp_path: Path) -> None:
    # mode B improved at 15%: v2 seeds sum 0.592851 (mean 0.197617), v1 seeds sum 0.623 (mean 0.2076666...).
    # Exact v2 - v1 = -0.0100497 -> -1.00. Rounding both means to 6 digits first gives -0.010050 -> -1.01.
    for seed, v2_f1, v1_f1 in ((2026, 0.197, 0.207), (2027, 0.198, 0.208), (2028, 0.197851, 0.208)):
        set_f1(v2_run(asr_scores, "0.15", seed), "B", "crf_improved", v2_f1)
        set_f1(v1_run(asr_scores, seed), "B", "crf_improved", v1_f1)
    summary, md = report(asr_scores, tmp_path)
    assert row(section(md, "## 4. Comparison: `v1` vs `v2` at 15% WER"), "B", "CRF improved")[5] == f"{MINUS}1.00"
    v1_cell = summary["v1"]["wer0.15"]["modes"]["B"]["models"]["crf_improved"]
    assert v1_cell["mean"] == pytest.approx(0.623 / 3, abs=1e-15)
    assert v1_cell["mean"] != 0.207667  # the exact mean, not the 6-digit one


def test_half_up_ties_round_away_from_zero_at_display() -> None:
    from punct_restore.reporting import asr_noise_stats, asr_noise_table

    assert asr_noise_stats.signed_pts(Decimal("0.00005")) == "+0.01"
    assert asr_noise_stats.signed_pts(Decimal("-0.00005")) == f"{MINUS}0.01"
    assert asr_noise_stats.signed_pts(Decimal("0.0000499999")) == "+0.00"
    assert asr_noise_stats.signed_pts(Decimal("-0.0000499999")) == "+0.00"
    # mean exactly 0.67985 (sum 2.03955 / 3): a binary float of it times 100 is 67.98499..., half-up prints 67.99
    stats = asr_noise_table.compute_seed_stats(0.7, {2026: 0.6798, 2027: 0.6799, 2028: 0.67985})
    assert stats.mean_pct == "67.99"
    assert stats.drop_from_clean_pct == "2.02"  # 0.7 - 0.67985 = 0.02015, a tie at the second digit


def test_compute_seed_stats_reports_range_min_max_and_drop() -> None:
    from punct_restore.reporting import asr_noise_table

    stats = asr_noise_table.compute_seed_stats(0.5, {2026: 0.40, 2027: 0.42, 2028: 0.47})
    assert (stats.min, stats.max, stats.min_pct, stats.max_pct) == (0.40, 0.47, "40.00", "47.00")
    assert stats.range == pytest.approx(0.07, abs=1e-15) and stats.range_pct == "7.00"
    assert stats.mean == pytest.approx(0.43, abs=1e-15) and stats.mean_pct == "43.00"
    assert stats.drop_from_clean == pytest.approx(0.07, abs=1e-15) and stats.drop_from_clean_pct == "7.00"
    assert stats.cell == f"43.00 (40.00{EN}47.00)"
    assert stats.per_seed == {"2026": 0.40, "2027": 0.42, "2028": 0.47}


def test_v1_row_of_the_mix_table_is_read_from_the_data(asr_scores: Path, tmp_path: Path) -> None:
    summary, md = report(asr_scores, tmp_path)
    mix = section(md, "## 2. Achieved Edit Mix (`v2` and `v1`)")
    zero = f"0.00 (0.00{EN}0.00)"
    assert row(mix, "`v1`")[1] == "15.00%"
    assert row(mix, "`v1`")[6:] == [f"25.00 (25.00{EN}25.00)", f"25.00 (25.00{EN}25.00)", zero,
                                     f"50.00 (50.00{EN}50.00)"]
    assert summary["v1"]["wer0.15"]["achieved"]["cost_share"]["insertion"]["mean"] == 0.0
    for seed in (2026, 2027, 2028):  # a v1 summary that does record insertions
        path = v1_run(asr_scores, seed) / "summary.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        patch_json(path, edits={**document["edits"], "insertion": 10},
                   cost_share={**document["cost_share"], "insertion": 0.1})
    (tmp_path / "again").mkdir()
    _, md = report(asr_scores, tmp_path / "again")
    assert row(section(md, "## 2. Achieved Edit Mix (`v2` and `v1`)"), "`v1`")[8] == f"10.00 (10.00{EN}10.00)"


def test_output_files_are_written_with_lf_line_endings(asr_scores: Path, tmp_path: Path) -> None:
    json_out, md_out = tmp_path / "summary.json", tmp_path / "ASR_NOISE_V2.md"
    assert run_main(asr_scores, json_out, md_out) == 0
    assert b"\r" not in json_out.read_bytes()
    assert b"\r" not in md_out.read_bytes()


def test_success_message_names_the_files_not_their_folder(
    asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "private_folder"
    out_dir.mkdir()
    assert run_main(asr_scores, out_dir / "s.json", out_dir / "r.md") == 0
    captured = capsys.readouterr()
    assert captured.out == "wrote s.json and r.md\n"
    assert captured.err == ""


@pytest.mark.parametrize("race", ["json", "md"])
def test_a_failed_second_write_leaves_neither_output(
    race: str, asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another process creates one output after the up-front check: no half-finished pair stays behind."""
    from punct_restore.reporting import asr_noise_table

    json_out, md_out = tmp_path / "s.json", tmp_path / "r.md"
    racer = json_out if race == "json" else md_out
    real_render = asr_noise_table.render_markdown

    def render_then_race(summary: dict[str, Any]) -> str:
        text = real_render(summary)
        racer.write_text("keep", encoding="utf-8")
        return text

    monkeypatch.setattr(asr_noise_table, "render_markdown", render_then_race)
    assert run_main(asr_scores, json_out, md_out) == 1
    assert_one_error_line(capsys.readouterr().err, racer.name)
    assert racer.read_text(encoding="utf-8") == "keep"  # the other process's file is not ours to delete
    assert [p for p in (json_out, md_out) if p != racer and p.exists()] == []


def test_a_failed_cleanup_does_not_replace_the_error_that_caused_it(
    asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second write fails and the first output cannot be removed: the user still hears about the second file."""
    from punct_restore.reporting import asr_noise_table

    json_out, md_out = tmp_path / "s.json", tmp_path / "r.md"
    real_render, real_unlink = asr_noise_table.render_markdown, Path.unlink

    def render_then_race(summary: dict[str, Any]) -> str:
        text = real_render(summary)
        md_out.write_text("keep", encoding="utf-8")
        return text

    def unlink_refused(self: Path, missing_ok: bool = False) -> None:
        if self == json_out:
            raise PermissionError(errno.EACCES, "Permission denied", str(self))
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(asr_noise_table, "render_markdown", render_then_race)
    monkeypatch.setattr(Path, "unlink", unlink_refused)
    assert run_main(asr_scores, json_out, md_out) == 1
    err = capsys.readouterr().err
    assert_one_error_line(err, md_out.name)
    assert json_out.name not in err and "denied" not in err


def test_a_write_that_fails_halfway_leaves_neither_output(
    asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from punct_restore.reporting import asr_noise_table

    monkeypatch.setattr(asr_noise_table, "render_markdown", lambda summary: "bad \ud800 text")  # cannot be UTF-8
    json_out, md_out = tmp_path / "s.json", tmp_path / "r.md"
    assert run_main(asr_scores, json_out, md_out) == 1
    assert_one_error_line(capsys.readouterr().err)
    assert not json_out.exists() and not md_out.exists()


@pytest.mark.parametrize("bad", ["file", "folder", "no_parent"])
def test_an_invalid_out_leaves_no_json_out(
    bad: str, asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    md_out = {"file": tmp_path / "there.md", "folder": tmp_path / "dir", "no_parent": tmp_path / "x" / "r.md"}[bad]
    if bad == "file":
        md_out.write_text("keep", encoding="utf-8")
    elif bad == "folder":
        md_out.mkdir()
    json_out = tmp_path / "s.json"
    assert run_main(asr_scores, json_out, md_out) == 1
    assert_one_error_line(capsys.readouterr().err, md_out.name)
    assert not json_out.exists()


def test_asr_noise_table_rejects_malformed_json_and_bad_values(
    asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_file = asr_scores / "hmm" / "hmm_A_test.json"
    bad_file.write_text("{broken", encoding="utf-8")
    assert run_main(asr_scores, tmp_path / "s.json", tmp_path / "o.md") == 1
    assert_one_error_line(capsys.readouterr().err, "hmm_A_test.json")

    dump(bad_file, {"label_mode": "A", "blocks": {"punctuation/all": {"macro_f1": "nan_str"}}})
    assert run_main(asr_scores, tmp_path / "s.json", tmp_path / "o.md") == 1
    assert_one_error_line(capsys.readouterr().err, "hmm_A_test.json", "macro_f1")


@pytest.mark.parametrize("value", [1.5, -0.1])
def test_macro_f1_outside_zero_one_is_rejected_naming_the_file(
    value: float, asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    set_f1(v2_run(asr_scores, "0.10", 2027), "B", "crf_basic", value)
    assert run_main(asr_scores, tmp_path / "s.json", tmp_path / "o.md") == 1
    err = capsys.readouterr().err
    assert_one_error_line(err, "score_crf_basic_B.json", "macro_f1")
    assert str(tmp_path) not in err
    assert not (tmp_path / "s.json").exists()


def test_a_score_file_for_the_wrong_mode_is_rejected_naming_the_file(
    asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_json(asr_scores / "crf_test" / "m3_basic_C_test.json", label_mode="A")
    assert run_main(asr_scores, tmp_path / "s.json", tmp_path / "o.md") == 1
    assert_one_error_line(capsys.readouterr().err, "m3_basic_C_test.json", "label_mode")


@pytest.mark.parametrize(
    ("relative", "key"),
    [
        (("hmm", "hmm_B_test.json"), "blocks"),
        (("asr_noise_v2", "v2", "wer0.05", "seed2028", "summary.json"), "cost_share"),
        (("asr_noise", "wer0.10", "summary.json"), "tokens_before"),
    ],
)
def test_a_missing_key_is_reported_with_the_file_name(
    relative: tuple[str, ...], key: str, asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = asr_scores.joinpath(*relative)
    document = json.loads(path.read_text(encoding="utf-8"))
    del document[key]
    dump(path, document)
    assert run_main(asr_scores, tmp_path / "s.json", tmp_path / "o.md") == 1
    err = capsys.readouterr().err
    assert_one_error_line(err, relative[-1], key)
    assert str(tmp_path) not in err


@pytest.mark.parametrize(
    ("updates", "needle"),
    [
        ({"seed": 1}, "expected seed 2027"),
        ({"edit_model": "v1"}, "edit_model must be 'v2'"),
        ({"wer": 0.2}, "expected wer 0.10"),
        ({"budget": 0}, "budget must be a positive integer"),
        ({"tokens_after": "many"}, "tokens_after must be a positive integer"),
        ({"achieved_wer": float("inf")}, "achieved_wer must be a finite number"),
        ({"edits": {"deletion": -1, "insertion": 1, "substitution": 1, "swap": 1}}, "edits.deletion must be a non-neg"),
        ({"edits": [1, 2]}, "a field has the wrong type"),
    ],
)
def test_a_bad_simulator_summary_is_rejected_naming_the_file(
    updates: dict[str, Any], needle: str, asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_json(v2_run(asr_scores, "0.10", 2027) / "summary.json", **updates)
    assert run_main(asr_scores, tmp_path / "s.json", tmp_path / "o.md") == 1
    err = capsys.readouterr().err
    assert_one_error_line(err, "summary.json", needle)
    assert str(tmp_path) not in err and not (tmp_path / "s.json").exists()


def test_summaries_that_disagree_across_seeds_are_rejected(
    asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_json(v2_run(asr_scores, "0.10", 2028) / "summary.json", budget=201)
    assert run_main(asr_scores, tmp_path / "s.json", tmp_path / "o.md") == 1
    assert_one_error_line(capsys.readouterr().err, "disagree on tokens_before or budget")


def test_asr_noise_table_usage_error_exits_with_2() -> None:
    from punct_restore.reporting import asr_noise_table

    with pytest.raises(SystemExit) as raised:
        asr_noise_table.main([])
    assert raised.value.code == 2


def test_asr_noise_table_refuses_existing_outputs_before_reading(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    existing = tmp_path / "exists.json"
    existing.write_text("keep", encoding="utf-8")
    assert run_main(tmp_path / "missing", existing, tmp_path / "out.md") == 1
    assert_one_error_line(capsys.readouterr().err, "exists.json", "already exists")
    assert existing.read_text(encoding="utf-8") == "keep"


def test_asr_noise_table_refuses_same_path_for_json_and_md(
    asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    same = tmp_path / "same.out"
    assert run_main(asr_scores, same, same) == 1
    assert_one_error_line(capsys.readouterr().err, "--json-out", "--out", "same file")
    assert not same.exists()


def test_asr_noise_table_refuses_folder_or_missing_parent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "a_folder"
    folder.mkdir()
    assert run_main(tmp_path / "missing", tmp_path / "s.json", folder) == 1
    err = capsys.readouterr().err
    assert_one_error_line(err, "a_folder", "is a folder")
    assert str(tmp_path) not in err

    assert run_main(tmp_path / "missing", tmp_path / "s.json", tmp_path / "no_such_dir" / "out.md") == 1
    err = capsys.readouterr().err
    assert_one_error_line(err, "out.md: folder no_such_dir not found")
    assert str(tmp_path) not in err


def test_asr_noise_table_reports_missing_input_without_folder_path(
    asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (v2_run(asr_scores, "0.10", 2027) / "score_hmm_B.json").unlink()
    assert run_main(asr_scores, tmp_path / "s.json", tmp_path / "out.md") == 1
    err = capsys.readouterr().err
    assert_one_error_line(err, "score_hmm_B.json")
    assert str(asr_scores) not in err
    assert not (tmp_path / "s.json").exists() and not (tmp_path / "out.md").exists()


def test_asr_noise_table_rejects_wrong_vocabulary_source(
    asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_json(v2_run(asr_scores, "0.05", 2026) / "summary.json", vocabulary_source="validation")
    assert run_main(asr_scores, tmp_path / "s.json", tmp_path / "out.md") == 1
    assert_one_error_line(capsys.readouterr().err, "summary.json", "default_train")


def test_asr_noise_table_runs_as_a_module_on_cp1252_console(asr_scores: Path, tmp_path: Path) -> None:
    json_out = tmp_path / "Ωsummary.json"
    md_out = tmp_path / "Ωreport.md"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"PYTHONIOENCODING": "cp1252"}
    done = subprocess.run(
        [sys.executable, "-m", "punct_restore.reporting.asr_noise_table", "--scores-root", str(asr_scores),
         "--json-out", str(json_out), "--out", str(md_out)],
        env=env, capture_output=True, check=False, timeout=60,
    )
    assert (done.returncode, done.stderr) == (0, b"")
    assert json_out.is_file() and md_out.is_file()
