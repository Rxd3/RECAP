"""Tests for punct_restore.reporting: the shared Markdown helpers, the table builders and time_models.

Everything runs on a tiny generated scores tree (tests/reporting_fixtures.py); no real data.
"""

from __future__ import annotations

import json
import statistics
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest
import reporting_fixtures
from reporting_fixtures import (
    RELEASE,
    TRAIN,
    assert_one_error_line,
    dump,
    output_lines,
    run_main,
    set_json,
    time_argv,
    tiny_data_root,
    tiny_hmm,
    train_rows,
    write_jsonl,
    write_manifests,
)

from punct_restore.models.crf import CRFModel
from punct_restore.models.hmm import HMMModel
from punct_restore.reporting import (
    build_ablation_table,
    build_m3_final_sections,
    build_m3_tables,
    build_tables,
    time_models,
)

scores = reporting_fixtures.scores  # the fixture is defined once, there; pytest finds it by this name
ROOT = Path(__file__).resolve().parents[1]
MINUS, EN, ARROW = "\u2212", "\u2013", "\u2192"
STAR_UP = "+10.00 (5.00 to 15.00) *"


def cell_text(macro: str, low: str, high: str) -> str:
    return f"{macro} ({low}{EN}{high})"


# ---------------------------------------------------------------- shared helpers (build_tables)
@pytest.mark.parametrize(("value", "text"), [(0.5118449, "51.18"), (0.51185, "51.19"), (0.67985, "67.99"),
                                              (0.0, "0.00"), (1, "100.00"), (-0.02, "-2.00")])
def test_pct_rounds_half_up_on_the_stored_decimal(value: float, text: str) -> None:
    assert build_tables.pct(value) == text  # 0.67985 * 100 is 67.98499... in float, so float rounding gives 67.98


@pytest.mark.parametrize(("value", "places", "grouping", "text"), [
    (0.125, 2, False, "0.13"), (0.305, 2, False, "0.31"), (2.5, 0, False, "3"), (1234.5, 0, True, "1,235"),
    (1234.5, 1, False, "1234.5"), (50, 0, False, "50"), (Decimal("62.5"), 0, False, "63"),
    ("0.0005", 3, False, "0.001")])
def test_half_up_rounds_the_shortest_decimal_half_up(
        value: float | str | Decimal, places: int, grouping: bool, text: str) -> None:
    assert build_tables.half_up(value, places, grouping=grouping) == text  # f"{0.125:.2f}" gives 0.12, f"{2.5:.0f}" 2


def test_ci_and_cell_show_the_interval_or_na() -> None:
    block = {"macro_f1": 0.5, "macro_f1_ci": {"low": 0.4, "high": 0.6}}
    assert build_tables.ci(block) == f"40.00{EN}60.00"
    assert build_tables.cell(block) == cell_text("50.00", "40.00", "60.00")
    assert build_tables.ci({"macro_f1": 0.5}) == "n/a"
    assert build_tables.cell({"macro_f1": 0.5}) == "50.00 (n/a)"


@pytest.mark.parametrize(("difference", "low", "high", "text"), [
    (0.1, 0.05, 0.15, STAR_UP),
    (-0.1, -0.15, -0.05, "-10.00 (-15.00 to -5.00) *"),
    (-0.02, -0.05, 0.01, "-2.00 (-5.00 to 1.00)"),
    (0.0, -0.01, 0.01, "+0.00 (-1.00 to 1.00)"),
    (0.01, 0.0, 0.02, "+1.00 (0.00 to 2.00)"),
])
def test_diff_cell_has_a_sign_and_a_star_only_when_the_interval_excludes_zero(
        difference: float, low: float, high: float, text: str) -> None:
    block = {"difference": difference, "difference_ci": {"low": low, "high": high}}
    assert build_tables.diff_cell(block) == text


def test_load_reads_json_and_names_a_malformed_file(tmp_path: Path) -> None:
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text('{"a": 1}', encoding="utf-8")
    bad.write_text("{not json", encoding="utf-8")
    assert build_tables.load(good) == {"a": 1}
    assert build_tables.load(str(good)) == {"a": 1}
    with pytest.raises(ValueError, match=r"^bad\.json: "):
        build_tables.load(bad)


# ---------------------------------------------------------------- build_m3_tables
def test_validation_tables_end_to_end(scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "m3.md"
    code, stdout, stderr = run_main(build_m3_tables, capsys, "--scores-root", scores, "--out", out)
    assert (code, stdout, stderr) == (0, f"wrote {out}\n", "")
    lines = output_lines(out)
    improved = cell_text("40.00", "30.00", "50.00")
    assert lines[0] == "## 1. Main result (validation, primary ID lists)"
    assert f"| CRF improved (M3) | {improved} | 41.00 / 42.00 / 43.00 | 90.00 |" in lines
    assert f"| Shortcut | {cell_text('10.00', '0.00', '20.00')} | 11.00 / 12.00 / 13.00 | 90.00 |" in lines
    assert "**Mode C** (`joint/unmasked`)" in lines
    assert f"| A | M3 improved {MINUS} M3 basic | {STAR_UP} | +11.00 (6.00 to 16.00) * |" in lines
    assert f"| B | M3 basic {MINUS} HMM | -2.00 (-5.00 to 1.00) | -1.00 (-4.00 to 2.00) |" in lines
    assert f"| C | M3 improved {MINUS} HMM | +20.00 (10.00 to 30.00) * | +21.00 (11.00 to 31.00) * |" in lines
    non_final = cell_text("35.00", "25.00", "45.00")
    casing = [cell_text(f"{n}.00", f"{n - 10}.00", f"{n + 10}.00") for n in (40, 41, 42, 43)]
    assert f"| CRF improved (M3) | {non_final} | {non_final} | {non_final} | " + " | ".join(casing) + " |" in lines
    assert f"| A | 200 | O{ARROW}COMMA 50 (25%); COMMA{ARROW}O 30 (15%) |" in lines
    assert "## Confusion matrices, M3 CRFs (validation)" in lines
    assert not any("threshold" in line for line in lines)


def test_error_shares_are_rounded_half_up(scores: Path, tmp_path: Path) -> None:
    dump(scores / "crf_validation" / "m3_improved_A.errors.json",
         {"errors_total": 8, "pair_counts": [{"gold": "O", "pred": "COMMA", "count": 1}]})  # 12.5%: f"{12.5:.0f}" is 12
    out = tmp_path / "m3.md"
    assert build_m3_tables.main(["--scores-root", str(scores), "--out", str(out)]) == 0
    assert f"| A | 8 | O{ARROW}COMMA 1 (13%) |" in output_lines(out)


@pytest.mark.parametrize("total", [0, -3])
def test_a_non_positive_error_total_is_refused_by_file_name(
        total: int, scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    set_json(scores / "crf_validation" / "m3_improved_A.errors.json", ("errors_total",), total)
    out = tmp_path / "m3.md"
    code, stdout, err = run_main(build_m3_tables, capsys, "--scores-root", scores, "--out", out)
    assert (code, stdout) == (1, "")  # not the vague "wrong type" line, and no division by zero
    assert_one_error_line(err, "m3_improved_A.errors.json: errors_total must be positive")
    assert not out.exists()


def test_class_tables_list_every_label_and_escape_the_pipe_of_joint_labels(scores: Path, tmp_path: Path) -> None:
    out = tmp_path / "m3.md"
    assert build_m3_tables.main(["--scores-root", str(scores), "--out", str(out)]) == 0
    lines = output_lines(out)
    header = ("| Class | Support | Shortcut P / R / F1 | HMM (M2) P / R / F1 | CRF basic (M3) P / R / F1 | "
              "CRF improved (M3) P / R / F1 |")
    assert header in lines
    assert "| O | 1,000 | 10.00 / 10.00 / 10.00 | 20.00 / 20.00 / 20.00 | 30.00 / 30.00 / 30.00 | "\
           "40.00 / 40.00 / 40.00 |" in lines
    assert "| DASH | 3,000 | 10.00 / 10.00 / 10.00 | 20.00 / 20.00 / 20.00 | 30.00 / 30.00 / 30.00 | "\
           "40.00 / 40.00 / 40.00 |" in lines
    assert r"| O\|LOWER | 1,000 | 10.00 / 10.00 / 10.00 | 20.00 / 20.00 / 20.00 | 30.00 / 30.00 / 30.00 | "\
           "40.00 / 40.00 / 40.00 |" in lines
    assert r"| COMMA\|FIRST_UPPER | 2,000 | 10.00 / 10.00 / 10.00 | 20.00 / 20.00 / 20.00 | 30.00 / 30.00 / "\
           "30.00 | 40.00 / 40.00 / 40.00 |" in lines
    assert "| LOWER | 1,000 | 11.00 / 11.00 / 11.00 | 21.00 / 21.00 / 21.00 | 31.00 / 31.00 / 31.00 | "\
           "41.00 / 41.00 / 41.00 |" in lines
    assert "| O|LOWER |" not in "\n".join(lines)


def test_confusion_matrices_cover_the_two_crfs_only(scores: Path, tmp_path: Path) -> None:
    out = tmp_path / "m3.md"
    build_m3_tables.main(["--scores-root", str(scores), "--out", str(out)])
    lines = output_lines(out)
    assert "### Mode C" in lines
    assert "*CRF basic (M3), punctuation* (rows = gold, columns = predicted)" in lines
    assert "*CRF improved (M3), casing (masked)* (rows = gold, columns = predicted)" in lines
    assert r"| gold \ pred | O | COMMA | DASH |" in lines
    assert "| O | 10 | 1 | 1 |" in lines
    assert not any(line.startswith(("*Shortcut,", "*HMM (M2),")) for line in lines)


def test_test_split_reads_crf_test_and_adds_the_threshold_model_with_tau(scores: Path, tmp_path: Path) -> None:
    out = tmp_path / "m3_test.md"
    assert build_m3_tables.main(["--scores-root", str(scores), "--split", "test", "--tau", "--out", str(out)]) == 0
    lines = output_lines(out)
    assert lines[0] == "## 1. Main result (test, primary test ID lists, each model predicted once)"
    assert f"| Shortcut | {cell_text('11.00', '1.00', '21.00')} | 12.00 / 13.00 / 14.00 | 90.00 |" in lines
    tau = cell_text("51.00", "41.00", "61.00")
    assert f"| CRF improved + threshold (M5) | {tau} | 52.00 / 53.00 / 54.00 | 90.00 |" in lines
    assert f"| A | threshold {MINUS} M3 improved | +0.00 (-1.00 to 1.00) | +1.00 (0.00 to 2.00) |" in lines
    assert "## Confusion matrices, M3 CRFs (test)" in lines
    assert not any("threshold), punctuation" in line for line in lines)


def test_without_tau_there_is_no_threshold_row(scores: Path, tmp_path: Path) -> None:
    out = tmp_path / "m3_test.md"
    assert build_m3_tables.main(["--scores-root", str(scores), "--split", "test", "--out", str(out)]) == 0
    assert "threshold" not in out.read_text(encoding="utf-8")


def test_a_confusion_matrix_with_a_missing_row_is_refused() -> None:
    block = {"labels": ["O", "COMMA"], "confusion": [[1, 0]]}
    with pytest.raises(ValueError, match="shorter"):
        build_m3_tables.matrix(block, "title")


# ---------------------------------------------------------------- build_m3_final_sections
def test_final_sections_end_to_end(scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifests, out = write_manifests(tmp_path / "manifests"), tmp_path / "final.md"
    code, stdout, stderr = run_main(build_m3_final_sections, capsys, "--scores-root", scores,
                                    "--manifests", manifests, "--out", out)
    assert (code, stdout, stderr) == (0, f"wrote {out}\n", "")
    text, lines = out.read_text(encoding="utf-8"), output_lines(out)
    assert lines[0] == "## 7. Threshold decoding (Member 5 extension)"
    assert "A 0.5, B 0.05, C 0.95. Test effects are in sections 1\u20132." in text
    assert "| 0.05 | 32.00 | **33.00** | 34.00 |" in lines
    assert "| 0.50 | **50.00** | 51.00 | 52.00 |" in lines
    assert "| 0.95 | 68.00 | 69.00 | **70.00** |" in lines


def test_final_sections_leakage_and_test_amb_rows(scores: Path, tmp_path: Path) -> None:
    out = tmp_path / "final.md"
    build_m3_final_sections.main(["--scores-root", str(scores), "--manifests",
                                  str(write_manifests(tmp_path / "m")), "--out", str(out)])
    lines = output_lines(out)
    hmm, crf = cell_text("20.00", "10.00", "30.00"), cell_text("40.00", "30.00", "50.00")
    assert f"| Official test, all utterances | canonical | 1,234 | {hmm} | {crf} |" in lines
    hmm, crf = cell_text("25.00", "15.00", "35.00"), cell_text("45.00", "35.00", "55.00")
    assert f"| Official test, all utterances | SponSpeech native | 6,234 | {hmm} | {crf} |" in lines
    short, tau = cell_text("12.00", "2.00", "22.00"), cell_text("52.00", "42.00", "62.00")
    assert f"| Shortcut | {short} | {short} | {short} |" in lines
    assert f"| CRF improved + threshold (M5) | {tau} | {tau} | {tau} |" in lines
    assert f"| M3 improved {MINUS} M3 basic | {STAR_UP} | {STAR_UP} | {STAR_UP} |" in lines
    zero = "+0.00 (-1.00 to 1.00)"
    assert f"| threshold {MINUS} M3 improved | {zero} | {zero} | {zero} |" in lines


def test_final_sections_runtime_rows_use_the_manifests_and_round_half_up(scores: Path, tmp_path: Path) -> None:
    out = tmp_path / "final.md"
    build_m3_final_sections.main(["--scores-root", str(scores), "--manifests",
                                  str(write_manifests(tmp_path / "m")), "--out", str(out)])
    lines = output_lines(out)
    assert "## 10. Runtime" in lines
    assert "| CRF basic | A | 1,234.6 | 1.23 | 123,457 | 0.234 |" in lines
    assert "| CRF basic | C | 3,000.3 | 1.23 | 123,457 | 0.234 |" in lines  # 3000.25 rounds up, not to even
    assert "| CRF improved | B | 4,000.0 | 2.50 | 61,728 | 0.500 |" in lines
    assert "| CRF improved | C | 6,000.5 | 2.50 | 61,728 | 0.500 |" in lines


def test_final_sections_read_the_manifests_next_to_the_scores_by_default(
        scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_manifests(scores / "crf_training_manifests")
    code, _, err = run_main(build_m3_final_sections, capsys, "--scores-root", scores, "--out", tmp_path / "x.md")
    assert (code, err) == (0, "")
    assert "| CRF basic | A | 1,234.6 |" in (tmp_path / "x.md").read_text(encoding="utf-8")


def test_final_sections_without_manifests_in_the_default_folder_name_the_missing_file(
        scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run_main(build_m3_final_sections, capsys, "--scores-root", scores, "--out", tmp_path / "x.md")
    assert code == 1
    assert_one_error_line(err, "crf_mode_a_basic_full_01.manifest.json")


def final_text(scores: Path, tmp_path: Path) -> str:
    out = tmp_path / "final.md"
    manifests = write_manifests(tmp_path / "m")
    assert build_m3_final_sections.main(["--scores-root", str(scores), "--manifests", str(manifests),
                                         "--out", str(out)]) == 0
    return out.read_text(encoding="utf-8")


def test_final_sections_quote_the_viterbi_scores_from_the_validation_files(scores: Path, tmp_path: Path) -> None:
    for mode, value, block in (("A", 0.30125, "punctuation/all"), ("B", 0.2, "punctuation/all"),
                               ("C", 0.44444, "joint/unmasked")):
        set_json(scores / "crf_validation" / f"m3_improved_{mode}.json", ("blocks", block, "macro_f1"), value)
    assert "standard Viterbi decoding scores A 30.13, B 20.00, C 44.44):" in final_text(scores, tmp_path)


def test_final_sections_leakage_prose_comes_from_the_leakage_files(scores: Path, tmp_path: Path) -> None:
    text = final_text(scores, tmp_path)
    assert "predicted all 1,234 official test utterances once" in text
    assert ("the improved CRF scores about 18 points higher than on the leak-free set; the HMM does not gain at "
            "all. Reporting the official test as-is would overstate the CRF by about 3 points.") in text


def test_final_sections_leakage_gains_are_rounded_half_up(scores: Path, tmp_path: Path) -> None:
    macro = ("blocks", "punctuation/all", "macro_f1")
    set_json(scores / "leakage" / "crf_A_test_train_video.json", macro, 0.555)  # 18.5 points over leak-free 0.37
    set_json(scores / "leakage" / "crf_A_test_official_all.json", macro, 0.415)  # 4.5 points
    set_json(scores / "leakage" / "hmm_A_test_train_video.json", macro, 0.255)  # 4.5 points over leak-free 0.21
    text = final_text(scores, tmp_path)
    assert "scores about 19 points higher than on the leak-free set; the HMM gains about 5 points. " in text
    assert "would overstate the CRF by about 5 points." in text


MACRO = ("blocks", "punctuation/all", "macro_f1")


def leakage_sentence(scores: Path, tmp_path: Path, **macro_f1: float) -> str:
    """The prose paragraph of section 8 after setting leakage macro F1 values (hmm_train, crf_train, crf_official)."""
    files = {"hmm_train": "hmm_A_test_train_video", "crf_train": "crf_A_test_train_video",
             "crf_official": "crf_A_test_official_all"}
    for name, value in macro_f1.items():
        set_json(scores / "leakage" / f"{files[name]}.json", MACRO, value)
    text = final_text(scores, tmp_path)
    start = text.index("On utterances from videos it was trained on")
    return text[start:text.index("\n", start)]


@pytest.mark.parametrize(("hmm_train", "phrase"), [
    (0.213, "the HMM does not gain at all."),    # +0.3 points rounds to 0: not "gains about 0 points"
    (0.2149, "the HMM does not gain at all."),   # +0.49
    (0.2, "the HMM does not gain at all."),      # -1 point
    (0.215, "the HMM gains about 1 point."),     # +0.5 rounds half-up to 1, singular
    (0.25, "the HMM gains about 4 points.")])
def test_the_hmm_gain_is_judged_on_the_rounded_whole_points(
        hmm_train: float, phrase: str, scores: Path, tmp_path: Path) -> None:
    assert f"; {phrase} Reporting" in leakage_sentence(scores, tmp_path, hmm_train=hmm_train)


@pytest.mark.parametrize(("crf_train", "crf_official", "trained", "official"), [
    (0.32, 0.35, "scores about 5 points lower than on the leak-free set", "would understate the CRF by about 2 points"),
    (0.37, 0.372, "scores about the same as on the leak-free set", "would not change the CRF's score"),
    (0.373, 0.366, "scores about the same as on the leak-free set", "would not change the CRF's score"),
    (0.38, 0.38, "scores about 1 point higher than on the leak-free set", "would overstate the CRF by about 1 point"),
    (0.36, 0.36, "scores about 1 point lower than on the leak-free set", "would understate the CRF by about 1 point"),
    (0.55, 0.4, "scores about 18 points higher than on the leak-free set",
     "would overstate the CRF by about 3 points")],
    ids=["loss", "zero", "tiny-either-way", "one-up", "one-down", "committed-gains"])
def test_the_crf_prose_follows_the_sign_of_the_rounded_gain(
        crf_train: float, crf_official: float, trained: str, official: str, scores: Path, tmp_path: Path) -> None:
    sentence = leakage_sentence(scores, tmp_path, crf_train=crf_train, crf_official=crf_official)
    assert sentence == (f"On utterances from videos it was trained on, the improved CRF {trained}; the HMM does not "
                        f"gain at all. Reporting the official test as-is {official}.")


def test_final_sections_hardware_comes_from_the_runtime_files_and_manifests(scores: Path, tmp_path: Path) -> None:
    text = final_text(scores, tmp_path)
    assert "Prediction speed on one laptop (Test CPU Family 9, Python 3.99.1, one process, no other jobs)" in text
    assert "measured on Member 3's machine (TESTBOX, 8 logical CPUs)" in text


def test_final_sections_runtime_prose_for_the_fixture_speeds(scores: Path, tmp_path: Path) -> None:
    assert ("predicts at about half the basic preset's speed: under 0.24 ms per utterance for basic, "
            "under 0.51 ms for improved.") in final_text(scores, tmp_path)


def test_final_sections_runtime_bounds_are_the_slowest_mode_rounded_half_up_plus_a_margin(
        scores: Path, tmp_path: Path) -> None:
    for preset, speeds in (("basic", (0.1, 0.145, 0.12)), ("improved", (0.4, 0.9, 0.45))):
        for mode, ms in zip("ABC", speeds, strict=True):
            set_json(scores / "runtime" / f"crf_{preset}_{mode}.json", ("ms_per_sequence",), ms)
    # 0.145 rounds half-up to 0.15 (half-even gives 0.14); 0.9 is the slowest improved mode; each gets +0.01
    assert "under 0.16 ms per utterance for basic, under 0.91 ms for improved." in final_text(scores, tmp_path)


def test_final_sections_quote_the_speed_ratios_when_they_are_not_about_half(scores: Path, tmp_path: Path) -> None:
    set_json(scores / "runtime" / "crf_improved_A.json", ("tokens_per_second",), 30000.0)  # 24% of basic
    assert "predicts at 24%–50% of the basic preset's speed: under" in final_text(scores, tmp_path)


@pytest.mark.parametrize(("improved_speed", "phrase"), [
    (40000.0, "about half the basic preset's speed"),    # exactly 0.4 and 0.6 are inside "about half"
    (60000.0, "about half the basic preset's speed"),
    (39000.0, "39%–50% of the basic preset's speed"),    # just outside either end: quote the range
    (61000.0, "50%–61% of the basic preset's speed"),
    (24500.0, "25%–50% of the basic preset's speed")])   # 24.5% rounds half-up (format(0.245, ".0%") gives 24%)
def test_final_sections_speed_ratio_edges(improved_speed: float, phrase: str, scores: Path, tmp_path: Path) -> None:
    set_json(scores / "runtime" / "crf_basic_A.json", ("tokens_per_second",), 100000.0)
    set_json(scores / "runtime" / "crf_improved_A.json", ("tokens_per_second",), improved_speed)
    assert f"The improved preset predicts at {phrase}: under" in final_text(scores, tmp_path)


def test_a_missing_manifest_is_reported_by_name(scores: Path, tmp_path: Path,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    empty = tmp_path / "no_manifests"
    empty.mkdir()
    code, _, err = run_main(build_m3_final_sections, capsys, "--scores-root", scores,
                            "--manifests", empty, "--out", tmp_path / "x.md")
    assert code == 1
    assert_one_error_line(err, "crf_mode_a_basic_full_01.manifest.json")
    assert str(tmp_path) not in err


# ---------------------------------------------------------------- build_ablation_table
def test_ablation_table_orders_groups_and_formats_rows(scores: Path, tmp_path: Path,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "ablation.md"
    code, stdout, _ = run_main(build_ablation_table, capsys, "--scores-root", scores, "--out", out)
    assert (code, stdout) == (0, f"wrote {out}\n")
    lines = output_lines(out)
    assert lines[0] == "## Feature ablations (validation; 50k-sequence sample, 100 iterations, frozen c1/c2)"
    assert (f"**Mode A**: full improved {cell_text('40.00', '30.00', '50.00')} primary, "
            f"{cell_text('35.00', '25.00', '45.00')} non_final") in lines
    rows = [line for line in lines if line.startswith(("| window2 |", "| casing |", "| zeta |"))]
    assert [row.split(" | ")[0] for row in rows[:3]] == ["| window2", "| casing", "| zeta"]
    assert (f"| window2 | {cell_text('38.00', '28.00', '48.00')} | +2.00 (1.00 to 3.00) * | "
            f"{cell_text('33.00', '23.00', '43.00')} | +3.00 (2.00 to 4.00) * |") in lines
    assert (f"| casing | {cell_text('36.00', '26.00', '46.00')} | +4.00 (3.00 to 5.00) * | "
            f"{cell_text('31.00', '21.00', '41.00')} | +5.00 (4.00 to 6.00) * |") in lines
    assert len(rows) == 9  # three groups in each of the three modes


def test_ablation_table_setup_text_is_taken_from_the_option(scores: Path, tmp_path: Path) -> None:
    out = tmp_path / "ablation.md"
    assert build_ablation_table.main(["--scores-root", str(scores), "--out", str(out),
                                      "--setup", "tiny setup"]) == 0
    assert output_lines(out)[0] == "## Feature ablations (validation; tiny setup)"


def test_ablation_groups_sort_known_names_first(tmp_path: Path) -> None:
    for name in ("zeta", "casing", "alpha", "window2"):
        dump(tmp_path / f"improved_minus_{name}.json", {})
    assert build_ablation_table.groups(tmp_path) == ["window2", "casing", "alpha", "zeta"]


# ---------------------------------------------------------------- one-line errors for every table builder
BUILDERS = [(build_m3_tables, [], "shortcut_A_validation.json"),
            (build_m3_final_sections, ["--manifests", "unused"], "tau_A.json"),
            (build_ablation_table, [], "improved.json")]


@pytest.mark.parametrize(("module", "extra", "missing"), BUILDERS)
def test_a_missing_scores_file_gives_exit_1_and_one_stderr_line(
        module: ModuleType, extra: list[str], missing: str, tmp_path: Path,
        capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out.md"
    code, stdout, err = run_main(module, capsys, "--scores-root", tmp_path / "nothing", *extra, "--out", out)
    assert (code, stdout) == (1, "")
    assert_one_error_line(err, missing)
    assert str(tmp_path) not in err
    assert not out.exists()


def test_malformed_json_is_reported_with_the_file_name(scores: Path, tmp_path: Path,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    (scores / "hmm" / "shortcut_A_validation.json").write_text("{not json", encoding="utf-8")
    code, _, err = run_main(build_m3_tables, capsys, "--scores-root", scores, "--out", tmp_path / "o.md")
    assert code == 1
    assert_one_error_line(err, "shortcut_A_validation.json")


def test_a_missing_key_is_reported_as_one_line(scores: Path, tmp_path: Path,
                                               capsys: pytest.CaptureFixture[str]) -> None:
    dump(scores / "hmm" / "shortcut_A_validation.json", {})
    code, _, err = run_main(build_m3_tables, capsys, "--scores-root", scores, "--out", tmp_path / "o.md")
    assert code == 1
    assert_one_error_line(err, "'blocks'")


def test_a_non_numeric_score_is_reported_as_one_line(scores: Path, tmp_path: Path,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    path = scores / "hmm" / "shortcut_A_validation.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["blocks"]["punctuation/all"]["macro_f1"] = "high"
    dump(path, document)
    code, _, err = run_main(build_m3_tables, capsys, "--scores-root", scores, "--out", tmp_path / "o.md")
    assert code == 1
    assert_one_error_line(err)


def test_an_unwritable_out_path_is_reported_by_name(scores: Path, tmp_path: Path,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    code, stdout, err = run_main(build_m3_tables, capsys, "--scores-root", scores,
                                 "--out", tmp_path / "no_such_folder" / "o.md")
    assert (code, stdout) == (1, "")
    assert_one_error_line(err, "o.md", "folder no_such_folder not found")


@pytest.mark.parametrize("module", [build_m3_tables, build_ablation_table, time_models])
def test_usage_errors_exit_with_2(module: ModuleType) -> None:
    with pytest.raises(SystemExit) as raised:
        module.main([])
    assert raised.value.code == 2


def test_split_must_be_validation_or_test(scores: Path, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        build_m3_tables.main(["--scores-root", str(scores), "--split", "test_amb", "--out", str(tmp_path / "o.md")])
    assert raised.value.code == 2


@pytest.mark.parametrize("module", [build_m3_tables, build_m3_final_sections, build_ablation_table])
def test_scores_root_defaults_to_the_committed_tree(module: ModuleType) -> None:
    args = module.build_parser().parse_args(["--out", "o.md"])
    assert args.scores_root == ROOT / "results" / "scores"


def test_reporting_sources_hold_no_personal_absolute_paths() -> None:
    folders = (ROOT / "src" / "punct_restore" / "reporting", ROOT / "scripts" / "reporting")
    for path in (path for folder in folders for path in folder.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "C:/" not in text and "C:\\" not in text and "Users" not in text, path.name


# ---------------------------------------------------------------- time_models
KEYS = ["kind", "mode", "model", "load_seconds", "sequences", "tokens", "predict_seconds", "median_seconds",
        "tokens_per_second", "ms_per_sequence", "hardware"]


def test_time_models_times_a_tiny_hmm(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model, data, out = tiny_hmm(tmp_path), tiny_data_root(tmp_path / "data"), tmp_path / "t.json"
    assert time_models.main(time_argv(model, out, data)) == 0
    assert capsys.readouterr().out.startswith("hmm A: ") and out.read_text(encoding="utf-8").endswith("}\n")
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert list(doc) == KEYS
    assert (doc["kind"], doc["mode"], doc["model"]) == ("hmm", "A", "hmm.json")  # the file name, not the path
    assert (doc["sequences"], doc["tokens"]) == (2, 5)
    assert len(doc["predict_seconds"]) == 3 and set(doc["hardware"]) == {"processor", "python"}
    median = statistics.median(doc["predict_seconds"])
    assert doc["median_seconds"] == median
    assert doc["tokens_per_second"] == 5 / median
    assert doc["ms_per_sequence"] == 1000 * median / 2


def test_time_models_times_hmm_fitting_when_asked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model, data, out = tiny_hmm(tmp_path), tiny_data_root(tmp_path / "data"), tmp_path / "t.json"
    train = write_jsonl(tmp_path / "train.jsonl", train_rows())
    ids = tmp_path / "train.ids"
    ids.write_text("t1\nt2\n", encoding="utf-8")
    fit = ["--fit-train", str(train), "--fit-ids", str(ids), "--smoothing", "0.1", "--min-token-count", "1"]
    assert time_models.main(time_argv(model, out, data, *fit)) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert list(doc) == [*KEYS, "fit_seconds", "fit_sequences"]
    assert doc["fit_sequences"] == 2 and doc["fit_seconds"] >= 0
    assert not capsys.readouterr().out.rstrip().endswith("fit -")


def test_time_models_times_a_tiny_crf(tmp_path: Path) -> None:
    records = [{"sample_id": sid, "tokens": tokens, "punct_labels": punct} for sid, tokens, punct in TRAIN]
    CRFModel(label_mode="A", max_iterations=5).fit(records).save_model(tmp_path / "crf.joblib")
    out = tmp_path / "t.json"
    argv = time_argv(tmp_path / "crf.joblib", out, tiny_data_root(tmp_path / "data"), kind="crf")
    assert time_models.main(argv) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert list(doc) == KEYS and doc["kind"] == "crf" and doc["sequences"] == 2


def test_time_models_data_release_is_selectable(tmp_path: Path) -> None:
    model, out = tiny_hmm(tmp_path), tmp_path / "t.json"
    data = tiny_data_root(tmp_path / "data", release="other-release")
    assert time_models.main(time_argv(model, out, data, "--release", "other-release")) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["sequences"] == 2


def test_time_models_lets_a_model_bug_propagate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(self: HMMModel, recs: list[dict[str, object]]) -> None:
        raise TypeError("model bug")

    monkeypatch.setattr(HMMModel, "predict", broken)
    model, data, out = tiny_hmm(tmp_path), tiny_data_root(tmp_path / "data"), tmp_path / "t.json"
    with pytest.raises(TypeError, match="model bug"):  # not relabelled as an input-JSON complaint
        time_models.main(time_argv(model, out, data))
    assert not out.exists()


def test_time_models_reports_a_missing_model_by_name(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "t.json"
    argv = time_argv(tmp_path / "gone.json", out, tiny_data_root(tmp_path / "data"))
    assert time_models.main(argv) == 1
    assert_one_error_line(capsys.readouterr().err, "gone.json")
    assert not out.exists()


def test_time_models_reports_a_folder_given_as_the_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder, out = tmp_path / "model_dir", tmp_path / "t.json"
    folder.mkdir()
    assert time_models.main(time_argv(folder, out, tiny_data_root(tmp_path / "data"))) == 1
    err = capsys.readouterr().err
    assert_one_error_line(err, "model_dir: is a folder, not a file")  # not "model file not found"
    assert str(tmp_path) not in err and not out.exists()


def test_time_models_reports_a_corrupt_model_by_name(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    assert time_models.main(time_argv(bad, tmp_path / "t.json", tiny_data_root(tmp_path / "data"))) == 1
    assert_one_error_line(capsys.readouterr().err, "broken.json")


def test_time_models_reports_malformed_validation_records_by_name(tmp_path: Path,
                                                                  capsys: pytest.CaptureFixture[str]) -> None:
    data = tiny_data_root(tmp_path / "data")
    (data / "processed" / RELEASE / "mode_A" / "validation.jsonl").write_text("{oops\n", encoding="utf-8")
    assert time_models.main(time_argv(tiny_hmm(tmp_path), tmp_path / "t.json", data)) == 1
    assert_one_error_line(capsys.readouterr().err, "validation.jsonl")


@pytest.mark.parametrize(("content", "message"), [
    (b"\xff\xfe\n", "error: validation.jsonl: not valid UTF-8\n"),
    (b'{"sample_id": "v1", "tokens": ["a"]}\n{oops\n', "error: validation.jsonl line 2: not valid JSON\n"),
    (b"[" * 100_000, "error: validation.jsonl line 1: not valid JSON\n"),
    (b'["not", "an", "object"]\n', "error: validation.jsonl line 1: not a JSON object\n"),
])
def test_time_models_words_a_bad_validation_line_the_way_the_other_commands_do(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: bytes, message: str
) -> None:
    data = tiny_data_root(tmp_path / "data")
    (data / "processed" / RELEASE / "mode_A" / "validation.jsonl").write_bytes(content)
    assert time_models.main(time_argv(tiny_hmm(tmp_path), tmp_path / "t.json", data)) == 1
    assert capsys.readouterr().err == message


@pytest.mark.parametrize("row", ['{"tokens": ["a"]}', '["not", "an", "object"]'])
def test_time_models_reports_a_validation_record_without_sample_id_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], row: str
) -> None:
    data = tiny_data_root(tmp_path / "data")
    (data / "processed" / RELEASE / "mode_A" / "validation.jsonl").write_text(row + "\n", encoding="utf-8")
    assert time_models.main(time_argv(tiny_hmm(tmp_path), tmp_path / "t.json", data)) == 1
    assert_one_error_line(capsys.readouterr().err, "validation.jsonl")


def test_time_models_reports_ids_that_match_no_record(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tiny_data_root(tmp_path / "data")
    (data / "splits" / RELEASE / "primary_eval" / "validation_mode_A.ids").write_text("zzz\n", encoding="utf-8")
    assert time_models.main(time_argv(tiny_hmm(tmp_path), tmp_path / "t.json", data)) == 1
    assert_one_error_line(capsys.readouterr().err, "validation_mode_A.ids")


def test_time_models_refuses_an_id_list_that_repeats_an_id(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tiny_data_root(tmp_path / "data")
    (data / "splits" / RELEASE / "primary_eval" / "validation_mode_A.ids").write_text("v1\nv1\n", encoding="utf-8")
    assert time_models.main(time_argv(tiny_hmm(tmp_path), tmp_path / "t.json", data)) == 1
    assert_one_error_line(capsys.readouterr().err, "Duplicate ID 'v1' found at line 2 in validation_mode_A.ids")


@pytest.mark.parametrize("extra", [
    ["--fit-train", "train.jsonl"],
    ["--fit-train", "train.jsonl", "--fit-ids", "x.ids", "--smoothing", "0.1"],
])
def test_fit_timing_needs_all_four_fit_options(tmp_path: Path, extra: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        time_models.main(time_argv(tmp_path / "m.json", tmp_path / "t.json", tmp_path, *extra))
    assert raised.value.code == 2


def test_fit_timing_is_hmm_only(tmp_path: Path) -> None:
    fit = ["--fit-train", "t.jsonl", "--fit-ids", "x.ids", "--smoothing", "0.1", "--min-token-count", "1"]
    with pytest.raises(SystemExit) as raised:
        time_models.main(time_argv(tmp_path / "m.joblib", tmp_path / "t.json", tmp_path, *fit, kind="crf"))
    assert raised.value.code == 2


@pytest.mark.parametrize(("flag", "value"), [("--kind", "svm"), ("--mode", "D")])
def test_kind_and_mode_are_restricted_choices(tmp_path: Path, flag: str, value: str) -> None:
    argv = time_argv(tmp_path / "m.json", tmp_path / "t.json", tmp_path)
    argv[argv.index(flag) + 1] = value
    with pytest.raises(SystemExit) as raised:
        time_models.main(argv)
    assert raised.value.code == 2
