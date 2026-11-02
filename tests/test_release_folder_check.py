"""One release-folder check for the commands that read ``<data root>/processed/<release>``.

``file_io.require_release`` is the one helper; simulate-asr-noise, build-leakage-subsets, build-word-clusters,
find-gold-noise and run-ablations call it before they read anything, and time-models calls it before it reads the
validation records. The four table commands (build-m3-tables, build-m3-final-sections, build-ablation-table and
asr-noise-table) check their scores folder. A missing release says what to run; a missing data root keeps the
``require_folder`` wording. No message shows a folder above the one that is named.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from reporting_fixtures import RELEASE as FIXTURE_RELEASE
from reporting_fixtures import tiny_data_root, tiny_hmm

from punct_restore import file_io
from punct_restore.cli import (
    build_leakage_subsets,
    build_word_clusters,
    find_gold_noise,
    run_ablations,
    simulate_asr_noise,
)
from punct_restore.reporting import (
    asr_noise_table,
    build_ablation_table,
    build_m3_final_sections,
    build_m3_tables,
    time_models,
)

Main = Callable[[list[str]], int]
RELEASE = "rel-v9"


def _argv(command: str, root: Path, tmp_path: Path) -> list[str]:
    """The command line of ``command`` with ``root`` as its data root and ``RELEASE`` as its release."""
    common = ["--data-root", str(root), "--release", RELEASE]
    out = str(tmp_path / "out")
    return {
        "simulate-asr-noise": [*common, "--split", "test", "--wer", "0.1", "--out", out],
        "build-leakage-subsets": [*common, "--out", out],
        "build-word-clusters": [*common, "--k", "3", "--out", out + ".json"],
        "find-gold-noise": [*common, "--out", out],
        "run-ablations": [*common, "--mode", "A", "--out", out],
    }[command]


MAINS: dict[str, Main] = {
    "simulate-asr-noise": simulate_asr_noise.main,
    "build-leakage-subsets": build_leakage_subsets.main,
    "build-word-clusters": build_word_clusters.main,
    "find-gold-noise": find_gold_noise.main,
    "run-ablations": run_ablations.main,
}
COMMANDS = list(MAINS)


# --- the helper -----------------------------------------------------------------------------------------------------


def test_the_helper_returns_the_processed_folder_of_the_release(tmp_path: Path) -> None:
    (tmp_path / "data" / "processed" / RELEASE).mkdir(parents=True)
    assert file_io.require_release(tmp_path / "data", RELEASE) == tmp_path / "data" / "processed" / RELEASE


def test_the_helper_names_the_release_and_the_data_root_by_name_and_says_what_to_run(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    with pytest.raises(FileNotFoundError) as caught:
        file_io.require_release(tmp_path / "data", RELEASE)
    assert str(caught.value) == f"{RELEASE}: release folder not found in data; run recap prepare-data build first"


def test_the_helper_refuses_a_release_that_is_a_file_like_a_missing_one(tmp_path: Path) -> None:
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "processed" / RELEASE).write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="release folder not found in data"):
        file_io.require_release(tmp_path / "data", RELEASE)


def test_the_helper_keeps_the_require_folder_wording_for_the_data_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as missing:
        file_io.require_release(tmp_path / "nowhere", RELEASE)
    assert str(missing.value) == "nowhere: data root not found"
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError) as a_file:
        file_io.require_release(tmp_path / "notes.txt", RELEASE)
    assert str(a_file.value) == "notes.txt: is a file, not a folder"


# --- the five commands ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("command", COMMANDS)
def test_a_missing_release_is_one_line_that_says_to_build_it(
        command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "data"
    (root / "splits" / RELEASE).mkdir(parents=True)  # only the release's processed folder is missing
    assert MAINS[command](_argv(command, root, tmp_path)) == 1
    captured = capsys.readouterr()
    assert captured.err == f"error: {RELEASE}: release folder not found in data; run recap prepare-data build first\n"
    assert str(tmp_path) not in captured.err
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("command", COMMANDS)
def test_a_missing_data_root_keeps_the_data_root_wording(
        command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert MAINS[command](_argv(command, tmp_path / "nowhere", tmp_path)) == 1
    assert capsys.readouterr().err == "error: nowhere: data root not found\n"


@pytest.mark.parametrize("command", COMMANDS)
def test_a_data_root_that_is_a_file_is_refused_by_name(
        command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    assert MAINS[command](_argv(command, tmp_path / "notes.txt", tmp_path)) == 1
    assert capsys.readouterr().err == "error: notes.txt: is a file, not a folder\n"


# --- the four table commands: the scores folder --------------------------------------------------------------------

TABLE_MAINS: dict[str, Main] = {
    "build-m3-tables": build_m3_tables.main,
    "build-m3-final-sections": build_m3_final_sections.main,
    "build-ablation-table": build_ablation_table.main,
    "asr-noise-table": asr_noise_table.main,
}


def _table_argv(command: str, scores: Path, tmp_path: Path) -> list[str]:
    if command == "asr-noise-table":
        return ["--scores-root", str(scores), "--json-out", str(tmp_path / "t.json"), "--out", str(tmp_path / "t.md")]
    return ["--scores-root", str(scores), "--out", str(tmp_path / "t.md")]


def _nothing_written(tmp_path: Path) -> bool:
    return not (tmp_path / "t.json").exists() and not (tmp_path / "t.md").exists()


@pytest.mark.parametrize("command", TABLE_MAINS)
def test_a_missing_scores_folder_is_named_by_name(
        command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert TABLE_MAINS[command](_table_argv(command, tmp_path / "nowhere", tmp_path)) == 1
    assert capsys.readouterr().err == "error: nowhere: scores folder not found\n"
    assert _nothing_written(tmp_path)


@pytest.mark.parametrize("command", TABLE_MAINS)
def test_a_scores_folder_that_is_a_file_is_refused_by_name(
        command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    assert TABLE_MAINS[command](_table_argv(command, tmp_path / "notes.txt", tmp_path)) == 1
    assert capsys.readouterr().err == "error: notes.txt: is a file, not a folder\n"
    assert _nothing_written(tmp_path)


# --- time-models: the data root and the release ---------------------------------------------------------------------


def _time_argv(model: Path, tmp_path: Path, data_root: Path, release: str = FIXTURE_RELEASE) -> list[str]:
    return ["--kind", "hmm", "--model", str(model), "--mode", "A", "--out", str(tmp_path / "t.json"),
            "--data-root", str(data_root), "--release", release]


def test_time_models_says_what_to_run_for_a_release_that_is_not_built(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tiny_data_root(tmp_path / "data")
    assert time_models.main(_time_argv(tiny_hmm(tmp_path), tmp_path, data, release=RELEASE)) == 1
    expected = f"error: {RELEASE}: release folder not found in data; run recap prepare-data build first\n"
    assert capsys.readouterr().err == expected
    assert not (tmp_path / "t.json").exists()


def test_time_models_names_a_missing_data_root_and_a_data_root_that_is_a_file(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = tiny_hmm(tmp_path)
    assert time_models.main(_time_argv(model, tmp_path, tmp_path / "nowhere")) == 1
    assert capsys.readouterr().err == "error: nowhere: data root not found\n"
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    assert time_models.main(_time_argv(model, tmp_path, tmp_path / "notes.txt")) == 1
    assert capsys.readouterr().err == "error: notes.txt: is a file, not a folder\n"
    assert not (tmp_path / "t.json").exists()
