"""A damaged evaluator file is one clean `error: <file>: <reason>` line in every reporting command; a bug is not hidden.

The reporting boundary (`reporting._cli.run`) turns only ValueError and OSError into an error line. So a JSON file that
is not what the reader needs must be refused where it is read, naming the file and the key, and never with the text of
Python's decoder. Each case damages one file of the tiny generated trees (tests/reporting_fixtures.py,
tests/asr_noise_table_fixtures.py) and runs a command's `main` in-process.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import reporting_fixtures
from asr_noise_table_fixtures import patch_json, v2_run, write_asr_scores
from reporting_fixtures import assert_one_error_line, dump, write_manifests
from test_length_position import argv as length_argv
from test_length_position import write_case

from punct_restore.reporting import (
    _cli,
    asr_noise_table,
    build_ablation_table,
    build_m3_final_sections,
    build_m3_tables,
    build_tables,
    length_position,
)

scores = reporting_fixtures.scores  # the fixture is defined once, there; pytest finds it by this name
RUNS = itertools.count()
DECODER_TEXT = ("Expecting", "line 1 column", "(char ")

# command name -> (the module, the options it takes besides the scores root and the outputs)
COMMANDS: dict[str, tuple[ModuleType, list[str]]] = {
    "build-m3-tables": (build_m3_tables, []),
    "build-m3-tables --split test --tau": (build_m3_tables, ["--split", "test", "--tau"]),
    "build-m3-final-sections": (build_m3_final_sections, []),
    "build-ablation-table": (build_ablation_table, []),
    "asr-noise-table": (asr_noise_table, []),
}


def run_command(name: str, root: Path, work: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    """Run the command ``name`` on the scores tree ``root`` (the manifests and outputs are in ``work``)."""
    module, options = COMMANDS[name]
    n = next(RUNS)
    if module is asr_noise_table:
        argv = ["--json-out", str(work / f"out{n}.json"), "--out", str(work / f"out{n}.md")]
    else:
        manifests = ["--manifests", str(work / "manifests")] if module is build_m3_final_sections else []
        argv = [*options, *manifests, "--out", str(work / f"out{n}.md")]
    code = module.main(["--scores-root", str(root), *argv])
    return code, capsys.readouterr().err


def files_read(name: str, root: Path, work: Path, capsys: pytest.CaptureFixture[str],
               monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The JSON files one good run of the command opens, in the order it opens them, each once."""
    opened: list[Path] = []
    real = Path.read_text

    def spy(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.suffix == ".json":
            opened.append(self)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    assert run_command(name, root, work, capsys)[0] == 0
    monkeypatch.setattr(Path, "read_text", real)
    return list(dict.fromkeys(opened))


def tree_of(name: str, work: Path) -> Path:
    """The generated scores tree the command ``name`` reads (the CRF training manifests are written beside it)."""
    write_manifests(work / "manifests")
    if COMMANDS[name][0] is asr_noise_table:
        return write_asr_scores(work / "asr_scores")
    return reporting_fixtures.write_scores_root(work / "scores")


def sampled(name: str, files: list[Path]) -> list[Path]:
    """Every file, except for the ASR tree where one reader sees 100 look-alike files: one per file name."""
    if name != "asr-noise-table":
        return files
    return list({path.name: path for path in reversed(files)}.values())


# --- whole-file damage: every file every command reads -------------------------------------------------------------


@pytest.mark.parametrize("damage", ["{}", "[]", "null", "5", '"text"'])
@pytest.mark.parametrize("name", COMMANDS)
def test_a_file_that_is_not_the_object_the_reader_needs_is_one_clean_line(
        name: str, damage: str, tmp_path: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    root = tree_of(name, tmp_path)
    for path in sampled(name, files_read(name, root, tmp_path, capsys, monkeypatch)):
        original = path.read_bytes()
        path.write_text(damage, encoding="utf-8")
        code, err = run_command(name, root, tmp_path, capsys)
        path.write_bytes(original)
        assert code == 1, f"{path.name} damaged as {damage}"
        assert_one_error_line(err, f"error: {path.name}: ")
        assert str(tmp_path) not in err and not any(text in err for text in DECODER_TEXT), err


@pytest.mark.parametrize("content", [b"", b"{not json", b"\xff\xfe not text", b"[" * 100_000],
                         ids=["empty", "bad", "not-utf8", "nested"])
@pytest.mark.parametrize("name", COMMANDS)
def test_a_file_that_is_not_json_is_named_without_the_text_of_the_decoder(
        name: str, content: bytes, tmp_path: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    root = tree_of(name, tmp_path)
    for path in sampled(name, files_read(name, root, tmp_path, capsys, monkeypatch)):
        original = path.read_bytes()
        path.write_bytes(content)
        code, err = run_command(name, root, tmp_path, capsys)
        path.write_bytes(original)
        assert code == 1, path.name
        assert_one_error_line(err, f"error: {path.name}: not valid ")
        assert not any(text in err for text in DECODER_TEXT), err


# --- a missing key or a value of the wrong kind: the key is named -------------------------------------------------

Edit = Callable[[dict[str, Any]], None]


def delete(*keys: str | int) -> Edit:
    """An edit that removes the entry at the path ``keys`` (a number is a list position)."""
    def edit(doc: dict[str, Any]) -> None:
        node: Any = doc
        for key in keys[:-1]:
            node = node[key]
        del node[keys[-1]]
    return edit


def put(value: object, *keys: str | int) -> Edit:
    """An edit that sets the entry at the path ``keys`` to ``value``."""
    def edit(doc: dict[str, Any]) -> None:
        node: Any = doc
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return edit


def edit_file(path: Path, edit: Edit) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    edit(document)
    dump(path, document)


P = "punctuation/all"
NF = "punctuation/non_final"
TABLES = "build-m3-tables"
FINAL = "build-m3-final-sections"
ABLATION = "build-ablation-table"
# (command, file below the scores root, the damage, the whole error line)
KEYS = [
    (TABLES, "hmm/shortcut_A_validation.json", delete("blocks"), "shortcut_A_validation.json: missing blocks"),
    (TABLES, "crf_validation/m3_basic_A.json", delete("blocks", P, "micro"),
     f"m3_basic_A.json: missing blocks.{P}.micro"),
    (TABLES, "crf_validation/m3_basic_A.json", put("high", "blocks", P, "accuracy"),
     f"m3_basic_A.json: blocks.{P}.accuracy is not a score"),
    (TABLES, "crf_validation/m3_basic_A.json", put(None, "blocks", P, "macro_f1"),
     f"m3_basic_A.json: blocks.{P}.macro_f1 is not a score"),
    (TABLES, "crf_validation/m3_basic_A.json", put(5, "blocks", P, "macro_f1_ci"),
     f"m3_basic_A.json: blocks.{P}.macro_f1_ci is not a JSON object"),
    (TABLES, "crf_validation/m3_basic_A.json", put({"low": 0.1}, "blocks", P, "macro_f1_ci"),
     f"m3_basic_A.json: missing blocks.{P}.macro_f1_ci.high"),
    (TABLES, "crf_validation/m3_improved_C.json", delete("blocks", "casing/unmasked_non_initial"),
     "m3_improved_C.json: missing blocks.casing/unmasked_non_initial"),
    (TABLES, "crf_validation/m3_basic_B.json", delete("blocks", P, "per_class", "DASH"),
     f"m3_basic_B.json: missing blocks.{P}.per_class.DASH"),
    (TABLES, "crf_validation/m3_basic_B.json", put("many", "blocks", P, "per_class", "COMMA", "support"),
     f"m3_basic_B.json: blocks.{P}.per_class.COMMA.support is not a whole number"),
    (TABLES, "crf_validation/m3_improved_B.json", put([[1, 2, 3], "x", []], "blocks", P, "confusion"),
     f"m3_improved_B.json: blocks.{P}.confusion.1 is not a JSON list"),
    (TABLES, "crf_validation/m3_improved_B.json", put([[1, 2, 3], [4, 5, 6]], "blocks", P, "confusion"),
     f"m3_improved_B.json: blocks.{P}.confusion is not 3 rows of 3 counts"),
    (TABLES, "crf_validation/m3_improved_B.json", put([[1, 2, 3], [4, 5], [6, 7, 8]], "blocks", P, "confusion"),
     f"m3_improved_B.json: blocks.{P}.confusion is not 3 rows of 3 counts"),
    (TABLES, "crf_validation/m3_basic_A.json", put([[1, 2], [3, 4], [5, 6]], "blocks", P, "confusion"),
     f"m3_basic_A.json: blocks.{P}.confusion is not 2 rows of 2 counts"),
    (TABLES, "crf_validation/m3_basic_C.json", put([[1, 2]], "blocks", "casing/unmasked", "confusion"),
     "m3_basic_C.json: blocks.casing/unmasked.confusion is not 2 rows of 2 counts"),
    (TABLES, "crf_validation/m3_basic_vs_hmm_B.json", delete("blocks", NF, "difference_ci"),
     f"m3_basic_vs_hmm_B.json: missing blocks.{NF}.difference_ci"),
    (TABLES, "crf_validation/m3_basic_vs_hmm_B.json", put("x", "blocks", P, "difference"),
     f"m3_basic_vs_hmm_B.json: blocks.{P}.difference is not a score"),
    (TABLES, "crf_validation/m3_improved_A.errors.json", delete("pair_counts"),
     "m3_improved_A.errors.json: missing pair_counts"),
    (TABLES, "crf_validation/m3_improved_A.errors.json", put("many", "errors_total"),
     "m3_improved_A.errors.json: errors_total is not a whole number"),
    (TABLES, "crf_validation/m3_improved_A.errors.json", put({}, "pair_counts"),
     "m3_improved_A.errors.json: pair_counts is not a JSON list"),
    (TABLES, "crf_validation/m3_improved_A.errors.json", put("x", "pair_counts", 0),
     "m3_improved_A.errors.json: pair_counts.0 is not a JSON object"),
    (FINAL, "threshold/tau_A.json", delete("grid"), "tau_A.json: missing grid"),
    (FINAL, "threshold/tau_A.json", put("x", "tau"), "tau_A.json: tau is not a number"),
    (FINAL, "threshold/tau_A.json", put(None, "grid", 1, "macro_f1"), "tau_A.json: grid.1.macro_f1 is not a score"),
    (FINAL, "threshold/tau_B.json", put([{"tau": 0.3, "macro_f1": 0.4}], "grid"),
     "tau_B.json: the grid does not have the taus of tau_A.json"),
    (FINAL, "leakage/crf_A_test_leakfree.json", delete("counts"), "crf_A_test_leakfree.json: missing counts"),
    (FINAL, "leakage/crf_A_test_leakfree.json", put("many", "counts", "sequences"),
     "crf_A_test_leakfree.json: counts.sequences is not a whole number"),
    (FINAL, "leakage/hmm_A_test_quality.json", delete("blocks", P, "macro_f1"),
     f"hmm_A_test_quality.json: missing blocks.{P}.macro_f1"),
    (FINAL, "hmm/shortcut_B_test_amb.json", delete("blocks"), "shortcut_B_test_amb.json: missing blocks"),
    (FINAL, "crf_test/m3_improved_vs_hmm_C_test_amb.json", put(None, "blocks", "joint/unmasked", "difference_ci"),
     "m3_improved_vs_hmm_C_test_amb.json: blocks.joint/unmasked.difference_ci is not a JSON object"),
    (ABLATION, "ablations/mode_A/improved.json", delete("blocks", NF, "macro_f1"),
     f"improved.json: missing blocks.{NF}.macro_f1"),
    (ABLATION, "ablations/mode_A/improved_minus_casing.json", put(None, "blocks", P, "macro_f1"),
     f"improved_minus_casing.json: blocks.{P}.macro_f1 is not a score"),
    (ABLATION, "ablations/mode_B/full_vs_improved_minus_window2.json", delete("blocks", P, "difference"),
     f"full_vs_improved_minus_window2.json: missing blocks.{P}.difference"),
    (FINAL, "../manifests/crf_mode_a_basic_full_01.manifest.json", delete("training_duration_seconds"),
     "crf_mode_a_basic_full_01.manifest.json: missing training_duration_seconds"),
    (FINAL, "../manifests/crf_mode_a_basic_full_01.manifest.json", put("slow", "training_duration_seconds"),
     "crf_mode_a_basic_full_01.manifest.json: training_duration_seconds is not a non-negative number"),
]


@pytest.mark.parametrize(("name", "relative", "edit", "message"), KEYS, ids=[f"{k[0]}-{k[3]}" for k in KEYS])
def test_a_missing_key_or_a_value_of_the_wrong_kind_is_named(
        name: str, relative: str, edit: Edit, message: str, scores: Path, tmp_path: Path,
        capsys: pytest.CaptureFixture[str]) -> None:
    write_manifests(tmp_path / "manifests")
    edit_file(tmp_path / "scores" / relative, edit)  # `scores` is tmp_path/scores; ../manifests is the other tree
    code, err = run_command(name, scores, tmp_path, capsys)
    assert code == 1
    assert err == f"error: {message}\n"


# (file below the ASR scores root, the damage, the whole error line)
ASR_KEYS = [
    ("hmm/hmm_B_test.json", delete("blocks"), "hmm_B_test.json: missing blocks"),
    ("crf_test/m3_basic_C_test.json", delete("blocks", "joint/unmasked", "macro_f1"),
     "m3_basic_C_test.json: missing blocks.joint/unmasked.macro_f1"),
    ("crf_test/m3_basic_A_test.json", put({}, "blocks", P, "macro_f1_ci"),
     f"m3_basic_A_test.json: missing blocks.{P}.macro_f1_ci.low"),
    ("crf_test/m3_basic_A_test.json", put(5, "blocks", P, "macro_f1_ci"),
     f"m3_basic_A_test.json: blocks.{P}.macro_f1_ci is not a JSON object"),
    ("crf_test/m3_basic_A_test.json", delete("label_mode"), "m3_basic_A_test.json: missing label_mode"),
    ("asr_noise_v2/v2/wer0.10/seed2027/summary.json", delete("cost_share", "swap"),
     "summary.json: missing cost_share.swap"),
    ("asr_noise_v2/v2/wer0.10/seed2027/summary.json", delete("wer"), "summary.json: missing wer"),
    ("asr_noise_v2/v2/wer0.10/seed2027/summary.json", put([1, 2], "edits"), "summary.json: edits is not a JSON object"),
    ("asr_noise_v2/v2/wer0.10/seed2027/summary.json", put(None, "cost_share"),
     "summary.json: cost_share is not a JSON object"),
    ("asr_noise/wer0.10/summary.json", delete("tokens_before"), "summary.json: missing tokens_before"),
    ("asr_noise/wer0.10/summary.json", delete("edits", "swap"), "summary.json: missing edits.swap"),
]


@pytest.mark.parametrize(("relative", "edit", "message"), ASR_KEYS, ids=[k[2] for k in ASR_KEYS])
def test_asr_noise_table_names_a_missing_key_or_a_value_of_the_wrong_kind(
        relative: str, edit: Edit, message: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = write_asr_scores(tmp_path / "asr_scores")
    edit_file(root / relative, edit)
    code, err = run_command("asr-noise-table", root, tmp_path, capsys)
    assert code == 1
    assert err == f"error: {message}\n"
    assert not list(tmp_path.glob("out*"))


def test_a_simulator_summary_of_the_wrong_kind_keeps_its_named_complaint(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = write_asr_scores(tmp_path / "asr_scores")
    patch_json(v2_run(root, "0.10", 2027) / "summary.json", budget="many")
    code, err = run_command("asr-noise-table", root, tmp_path, capsys)
    assert (code, err) == (1, "error: summary.json: budget must be a positive integer\n")


# --- a number the tables cannot round ------------------------------------------------------------------------------
# Infinity, NaN, 1e999 and an integer of hundreds of digits are JSON that Python reads but no number a table can print:
# pct and half_up raise decimal.InvalidOperation, a float conversion raises OverflowError. A finite number outside
# 0..1 (1e30) crashes pct the same way. Each is one named line, never a traceback.

NOT_FINITE = {"Infinity": "Infinity", "-Infinity": "-Infinity", "NaN": "NaN", "1e999": "1e999",
              "-1e999": "-1e999", "400-digit": "9" * 400, "negative 400-digit": "-" + "9" * 400}
OUTSIDE_SCORE = {"1e30": "1e30", "1.5": "1.5", "-0.01": "-0.01", "2": "2"}
OUTSIDE_DIFFERENCE = {"1e30": "1e30", "1.5": "1.5", "-1.01": "-1.01", "-2": "-2"}
Row = tuple[str, str, tuple[str, ...], str]  # command, file below the scores root, path to the number, end of the line


def number_file(relative: str) -> str:
    return relative.rsplit("/", 1)[-1]


def put_raw(path: Path, keys: tuple[str | int, ...], raw: str) -> None:
    """Set the entry at ``keys`` to the JSON text ``raw``: json.dumps cannot write 1e999 or a 400-digit number."""
    edit_file(path, put("__raw__", *keys))
    path.write_text(path.read_text(encoding="utf-8").replace('"__raw__"', raw), encoding="utf-8")


def is_score(name: str, relative: str, *keys: str) -> Row:
    """A number a table shows as a percentage: 0..1 (a difference: -1..1), else ``<key> is not a score``."""
    return name, relative, keys, f"{'.'.join(keys)} is not a score"


def is_difference(keys: tuple[str, ...]) -> bool:
    return "difference" in keys or "difference_ci" in keys


CRF_A = "crf_validation/m3_basic_A.json"
CRF_B = "crf_validation/m3_basic_B.json"
CRF_VS_HMM_B = "crf_validation/m3_basic_vs_hmm_B.json"
NUMBER_ROWS: list[Row] = [
    is_score(TABLES, "hmm/shortcut_A_validation.json", "blocks", P, "macro_f1"),
    is_score(TABLES, CRF_A, "blocks", P, "macro_f1_ci", "low"),
    is_score(TABLES, CRF_A, "blocks", P, "macro_f1_ci", "high"),
    is_score(TABLES, CRF_A, "blocks", P, "micro", "precision"),
    is_score(TABLES, CRF_A, "blocks", P, "micro", "recall"),
    is_score(TABLES, CRF_A, "blocks", P, "micro", "f1"),
    is_score(TABLES, CRF_A, "blocks", P, "accuracy"),
    is_score(TABLES, CRF_B, "blocks", P, "per_class", "COMMA", "precision"),
    is_score(TABLES, CRF_B, "blocks", P, "per_class", "COMMA", "recall"),
    is_score(TABLES, CRF_B, "blocks", P, "per_class", "DASH", "f1"),
    is_score(TABLES, "crf_validation/m3_improved_C.json", "blocks", "casing/all", "macro_f1"),
    is_score(TABLES, CRF_VS_HMM_B, "blocks", P, "difference"),
    is_score(TABLES, CRF_VS_HMM_B, "blocks", NF, "difference_ci", "low"),
    is_score(TABLES, CRF_VS_HMM_B, "blocks", NF, "difference_ci", "high"),
    is_score(FINAL, "threshold/tau_B.json", "grid", "2", "macro_f1"),
    is_score(FINAL, "crf_validation/m3_improved_C.json", "blocks", "joint/unmasked", "macro_f1"),
    is_score(FINAL, "leakage/crf_A_test_leakfree.json", "blocks", P, "macro_f1"),
    is_score(FINAL, "leakage/hmm_A_test_train_video.json", "blocks", P, "macro_f1_ci", "high"),
    is_score(FINAL, "hmm/shortcut_A_test_amb.json", "blocks", P, "macro_f1"),
    is_score(FINAL, "crf_test/m3_improved_vs_hmm_C_test_amb.json", "blocks", "joint/unmasked", "difference"),
    is_score(FINAL, "threshold/tau_vs_improved_A_test_amb.json", "blocks", P, "difference_ci", "low"),
    is_score(ABLATION, "ablations/mode_A/improved.json", "blocks", P, "macro_f1"),
    is_score(ABLATION, "ablations/mode_B/improved_minus_casing.json", "blocks", NF, "macro_f1_ci", "low"),
    is_score(ABLATION, "ablations/mode_B/full_vs_improved_minus_window2.json", "blocks", P, "difference"),
    is_score(ABLATION, "ablations/mode_C/full_vs_improved_minus_window2.json", "blocks", NF, "difference_ci", "high"),
    (FINAL, "threshold/tau_A.json", ("tau",), "tau is not a number"),
    (FINAL, "runtime/crf_basic_A.json", ("median_seconds",), "median_seconds must be a positive number"),
    (FINAL, "runtime/crf_improved_B.json", ("tokens_per_second",), "tokens_per_second must be a positive number"),
    (FINAL, "runtime/crf_improved_C.json", ("ms_per_sequence",), "ms_per_sequence must be a positive number"),
    (FINAL, "../manifests/crf_mode_b_improved_full_01.manifest.json", ("training_duration_seconds",),
     "training_duration_seconds is not a non-negative number"),
]
SCORE_ROWS = [row for row in NUMBER_ROWS if row[3].endswith("is not a score")]
ROW_IDS = [f"{row[0]}-{row[1].rsplit('/', 1)[-1]}-{row[2][-1]}" for row in NUMBER_ROWS]


def run_with_number(row: Row, raw: str, scores: Path, tmp_path: Path,
                    capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    name, relative, keys, _ = row
    write_manifests(tmp_path / "manifests")
    put_raw(tmp_path / "scores" / relative, tuple(int(k) if k.isdigit() else k for k in keys), raw)
    return run_command(name, scores, tmp_path, capsys)


def named_line(row: Row) -> str:
    return f"error: {row[1].rsplit('/', 1)[-1]}: {row[3]}\n"


@pytest.mark.parametrize("raw", NOT_FINITE.values(), ids=list(NOT_FINITE))
@pytest.mark.parametrize("row", NUMBER_ROWS, ids=ROW_IDS)
def test_a_number_that_is_not_finite_is_one_named_line_in_every_table_command(
        row: Row, raw: str, scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_with_number(row, raw, scores, tmp_path, capsys) == (1, named_line(row))


OUTSIDE = [(row, raw) for row in SCORE_ROWS
           for raw in (OUTSIDE_DIFFERENCE if is_difference(row[2]) else OUTSIDE_SCORE).values()]


@pytest.mark.parametrize(("row", "raw"), OUTSIDE, ids=[f"{ROW_IDS[NUMBER_ROWS.index(r)]}={v}" for r, v in OUTSIDE])
def test_a_finite_number_outside_the_range_of_a_score_is_not_a_score(
        row: Row, raw: str, scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_with_number(row, raw, scores, tmp_path, capsys) == (1, named_line(row))


ENDS = [(row, raw) for row in SCORE_ROWS for raw in (("-1", "1") if is_difference(row[2]) else ("0", "1"))]


@pytest.mark.parametrize(("row", "raw"), ENDS, ids=[f"{ROW_IDS[NUMBER_ROWS.index(r)]}={v}" for r, v in ENDS])
def test_the_ends_of_the_range_are_scores(
        row: Row, raw: str, scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, err = run_with_number(row, raw, scores, tmp_path, capsys)
    assert (code, err) == (0, "")


def test_two_speeds_whose_ratio_is_no_number_are_refused_by_name(
        scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Each speed is a finite number, yet 1e300 over 1e-300 is Infinity, which the speed sentence cannot round."""
    put_raw(tmp_path / "scores" / "runtime" / "crf_improved_A.json", ("tokens_per_second",), "1e300")
    put_raw(tmp_path / "scores" / "runtime" / "crf_basic_A.json", ("tokens_per_second",), "1e-300")
    write_manifests(tmp_path / "manifests")
    assert run_command(FINAL, scores, tmp_path, capsys) == (
        1, "error: crf_improved_A.json: tokens_per_second cannot be compared with that of crf_basic_A.json\n")


@pytest.mark.parametrize("raw", ["-1", "-0.5", "-1e-9"])
def test_a_negative_fit_time_is_refused_and_zero_is_not(
        raw: str, scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    row = NUMBER_ROWS[-1]
    assert run_with_number(row, raw, scores, tmp_path, capsys) == (1, named_line(row))
    assert run_with_number(row, "0", scores, tmp_path, capsys)[0] == 0


# (file below the ASR scores root, the path to the number, the field the line names)
ASR_NUMBERS = [
    ("hmm/hmm_B_test.json", ("blocks", P, "macro_f1"), "macro_f1"),
    ("crf_test/m3_basic_A_test.json", ("blocks", P, "macro_f1_ci", "high"), "macro_f1_ci.high"),
    ("crf_test/m3_basic_C_test.json", ("blocks", "joint/unmasked", "macro_f1_ci", "low"), "macro_f1_ci.low"),
    ("asr_noise_v2/v2/wer0.10/seed2027/summary.json", ("wer",), "wer"),
    ("asr_noise_v2/v2/wer0.10/seed2027/summary.json", ("achieved_wer",), "achieved_wer"),
    ("asr_noise_v2/v2/wer0.10/seed2027/summary.json", ("cost_share", "swap"), "cost_share.swap"),
    ("asr_noise/wer0.10/summary.json", ("achieved_wer",), "achieved_wer"),
    ("asr_noise/wer0.10/summary.json", ("cost_share", "deletion"), "cost_share.deletion"),
]
ASR_IDS = [f"{number_file(relative)}-{'.'.join(keys)}" for relative, keys, _ in ASR_NUMBERS]


def run_asr_with_number(index: int, raw: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    relative, keys, _ = ASR_NUMBERS[index]
    root = write_asr_scores(tmp_path / "asr_scores")
    put_raw(root / relative, keys, raw)
    code, err = run_command("asr-noise-table", root, tmp_path, capsys)
    assert not list(tmp_path.glob("out*"))  # nothing is written for a refused input
    return code, err


@pytest.mark.parametrize("raw", NOT_FINITE.values(), ids=list(NOT_FINITE))
@pytest.mark.parametrize("index", range(len(ASR_NUMBERS)), ids=ASR_IDS)
def test_asr_noise_table_refuses_a_number_that_is_not_finite_by_name(
        index: int, raw: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    relative, _, field = ASR_NUMBERS[index]
    assert run_asr_with_number(index, raw, tmp_path, capsys) == (
        1, f"error: {number_file(relative)}: {field} must be a finite number\n")


# (the simulator summary's path to the count, how the line calls it)
ASR_COUNTS = [(("tokens_before",), "a positive integer"), (("tokens_after",), "a positive integer"),
              (("budget",), "a positive integer"), (("edits", "deletion"), "a non-negative integer"),
              (("edits", "swap"), "a non-negative integer")]


@pytest.mark.parametrize("raw", ["9" * 400, "1" + "0" * 309], ids=["400-digit", "310-digit"])
@pytest.mark.parametrize(("keys", "kind"), ASR_COUNTS, ids=[".".join(keys) for keys, _ in ASR_COUNTS])
def test_asr_noise_table_refuses_a_count_too_big_for_a_float_by_name(
        keys: tuple[str, ...], kind: str, raw: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A count is averaged and shown as a float: with 310 digits it is Infinity, which half_up cannot round."""
    root = write_asr_scores(tmp_path / "asr_scores")
    put_raw(root / "asr_noise_v2/v2/wer0.10/seed2027/summary.json", keys, raw)
    assert run_command("asr-noise-table", root, tmp_path, capsys) == (
        1, f"error: summary.json: {'.'.join(keys)} must be {kind}\n")
    assert not list(tmp_path.glob("out*"))


UNIT_FIELDS = [i for i, (_, _, field) in enumerate(ASR_NUMBERS) if field != "wer"]  # wer has its own check


@pytest.mark.parametrize("raw", ["1e30", "1.5", "-0.01"])
@pytest.mark.parametrize("index", UNIT_FIELDS, ids=[ASR_IDS[i] for i in UNIT_FIELDS])
def test_asr_noise_table_refuses_a_share_or_score_outside_0_to_1_by_name(
        index: int, raw: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """1e30 reaches pct, which cannot round it; a score, an achieved WER and a cost share are fractions."""
    relative, _, field = ASR_NUMBERS[index]
    assert run_asr_with_number(index, raw, tmp_path, capsys) == (
        1, f"error: {number_file(relative)}: {field} must be between 0 and 1\n")


# --- the boundary --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bug", [KeyError("k"), TypeError("t"), ZeroDivisionError("z"), AttributeError("a")])
def test_a_bug_in_the_task_is_not_relabelled_as_a_bad_input_file(bug: Exception) -> None:
    def task() -> None:
        raise bug

    with pytest.raises(type(bug)):
        _cli.run(task)


def test_a_malformed_file_is_named_by_load_without_the_decoder_text(tmp_path: Path) -> None:
    for content, reason in ((b"{not json", "not valid JSON"), (b"\xff\xfe", "not valid UTF-8"),
                            (b"[" * 100_000, "not valid JSON")):
        (tmp_path / "bad.json").write_bytes(content)
        with pytest.raises(ValueError) as caught:
            build_tables.load(tmp_path / "bad.json")
        assert str(caught.value) == f"bad.json: {reason}"


def test_load_checks_a_schema_when_it_is_given_one(tmp_path: Path) -> None:
    path = tmp_path / "doc.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    assert build_tables.load(path, {"a": float}) == {"a": 1}
    with pytest.raises(ValueError, match=r"^doc\.json: missing b$"):
        build_tables.load(path, {"b": float})


# --- length-position: the evaluator names a damaged input file ----------------------------------------------------


@pytest.mark.parametrize("damage", [b"{}\n", b"[]\n", b"null\n", b"5\n", b""],
                         ids=["object", "list", "null", "number", "empty"])
@pytest.mark.parametrize("which", ["gold", "ids", "first", "perfect"])
def test_length_position_names_a_damaged_input_file(
        which: str, damage: bytes, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    case = write_case(tmp_path / "case")
    path = {"gold": case.gold, "ids": case.ids, **case.predictions}[which]
    path.write_bytes(damage)
    code = length_position.main(length_argv(case, tmp_path / "lp.json"))
    err = capsys.readouterr().err
    assert code == 1
    assert_one_error_line(err, f"error: {path.name}")
    assert str(tmp_path) not in err and not any(text in err for text in DECODER_TEXT), err


def test_a_tau_grid_with_more_taus_than_mode_a_still_builds_as_on_main(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tree_of(FINAL, tmp_path)
    edit_file(root / "threshold" / "tau_B.json", lambda doc: doc["grid"].append({"tau": 0.99, "macro_f1": 0.01}))
    code, err = run_command(FINAL, root, tmp_path, capsys)
    assert (code, err) == (0, "")


def test_a_tau_grid_that_swaps_one_of_mode_a_taus_is_refused(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tree_of(FINAL, tmp_path)
    taus = [point["tau"] for point in json.loads((root / "threshold" / "tau_A.json").read_text("utf-8"))["grid"]]
    swapped = [{"tau": tau, "macro_f1": 0.4} for tau in taus[1:]] + [{"tau": 0.999, "macro_f1": 0.4}]
    edit_file(root / "threshold" / "tau_B.json", put(swapped, "grid"))  # same size as mode A's grid, one tau differs
    code, err = run_command(FINAL, root, tmp_path, capsys)
    assert code == 1 and "tau_B.json: the grid does not have the taus of tau_A.json" in err
