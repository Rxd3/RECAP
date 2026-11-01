"""One release-folder check for the commands that read ``<data root>/processed/<release>``.

``file_io.require_release`` is the one helper; simulate-asr-noise, build-leakage-subsets, build-word-clusters,
find-gold-noise and run-ablations call it before they read anything, and asr-noise-table checks its scores folder.
A missing release says what to run; a missing data root keeps the ``require_folder`` wording. Neither message shows
a folder above the one that is named.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from punct_restore import file_io
from punct_restore.cli import (
    build_leakage_subsets,
    build_word_clusters,
    find_gold_noise,
    run_ablations,
    simulate_asr_noise,
)
from punct_restore.reporting import asr_noise_table

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


# --- asr-noise-table: the scores folder -----------------------------------------------------------------------------


def _table_argv(scores: Path, tmp_path: Path) -> list[str]:
    return ["--scores-root", str(scores), "--json-out", str(tmp_path / "t.json"), "--out", str(tmp_path / "t.md")]


def test_a_missing_scores_folder_is_named_by_name(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert asr_noise_table.main(_table_argv(tmp_path / "nowhere", tmp_path)) == 1
    assert capsys.readouterr().err == "error: nowhere: scores folder not found\n"
    assert not (tmp_path / "t.json").exists() and not (tmp_path / "t.md").exists()


def test_a_scores_folder_that_is_a_file_is_refused_by_name(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    assert asr_noise_table.main(_table_argv(tmp_path / "notes.txt", tmp_path)) == 1
    assert capsys.readouterr().err == "error: notes.txt: is a file, not a folder\n"
