"""Tests for punct_restore.cli.find_gold_noise on a tiny generated data root: no real data.

The fixture has three videos in validation, one in test, one in test_amb and one in the default-train list. The
expected numbers in the tests below were counted by hand from ``VALIDATION`` / ``TEST`` / ``TRAIN``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from test_gold_noise import make_record

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "find_gold_noise.py"  # the stub: subprocess runs
CLI_MODULE = ROOT / "src" / "punct_restore" / "cli" / "find_gold_noise.py"  # the code: in-process loads
RELEASE = "sponspeech-m1-v2"
SRT = "Well. 506 00:18:45,000 -- 00:18:44,999 Oh no."

# (sample id, video, text the tokens were made from, original text or None); LISTED holds the scored ids.
VALIDATION = [
    ("val-0", "vidA", "Jeff: hello there. Bob: hi.", None),          # 2 labels
    ("val-1", "vidA", "Jeff: ok then. Bob: sure.", None),             # 2 labels
    ("val-2", "vidA", "Yeah, Jeff: it was fine.", None),              # a label only because Jeff recurs
    ("val-3", "vidA", "Yeah, Bob: no. Tip: one here.", None),         # Bob recurs only twice; Tip is no name
    ("val-4", "vidB", "Well. 506 -- Oh no.", SRT),                    # cue residue
    ("val-5", "vidA", "Jeff: extra.", None),                          # not listed: still counts for the inventory
    ("val-6", "732fql_jhJQ", "Pat: hi there.", None),                 # label and auto-caption video: two rows
    ("val-7", "732fql_jhJQ", "hello there friend.", None),
    ("val-8", "vidB", "We saw it here: all good.", None),             # a colon label that is not a speaker label
]
VALIDATION_LISTED = [i for i, *_ in VALIDATION if i != "val-5"]
TEST = [("tst-0", "vidC", "Sadhguru: hello.", None), ("tst-1", "vidC", "plain words here.", None)]
TEST_AMB = [("amb-0", "vidD", "plain text only", None)]
TRAIN = [
    ("trn-0", "vidT", "Mia: a b.", None), ("trn-1", "vidT", "Mia: c d.", None), ("trn-2", "vidT", "Mia: e f.", None),
    ("trn-3", "vidT", "go on, Mia: right.", None),                    # a label only because Mia recurs
    ("trn-4", "vidT", "Mia: more.", None),                            # not listed
    ("trn-5", "vidT", "Well. 61 -- Oh no.", "Well. 61 00:03:10,8 -- 00:03:13,66 Oh no."),
]
TRAIN_LISTED = ["trn-0", "trn-1", "trn-2", "trn-3", "trn-5"]


def write_jsonl(path: Path, rows: list[tuple[str, str, str, str | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(make_record(text, original=original, group=group, sid=sid)) for sid, group, text, original
             in rows]
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8", newline="\n")   # a blank line between records


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8", newline="\n")


def build_root(base: Path) -> Path:
    root = base / "data"
    mode = root / "processed" / RELEASE / "mode_B"
    primary = root / "splits" / RELEASE / "primary_eval"
    for name, rows in (("validation", VALIDATION), ("test", TEST), ("test_amb", TEST_AMB), ("train", TRAIN)):
        write_jsonl(mode / f"{name}.jsonl", rows)
    write_ids(primary / "validation_mode_B.ids", VALIDATION_LISTED)
    write_ids(primary / "test_mode_B.ids", [i for i, *_ in TEST])
    write_ids(primary / "test_amb_mode_B.ids", [i for i, *_ in TEST_AMB])
    write_ids(root / "splits" / RELEASE / "default_train" / "mode_B.ids", TRAIN_LISTED)
    return root


@pytest.fixture(scope="module")
def mod() -> Iterator[ModuleType]:
    name = "find_gold_noise_under_test"
    spec = importlib.util.spec_from_file_location(name, CLI_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules[name]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return build_root(tmp_path)


def argv_for(root: Path, out: Path) -> list[str]:
    return ["--data-root", str(root), "--release", RELEASE, "--mode", "B", "--out", str(out)]


def run(mod: ModuleType, capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str, str]:
    code = mod.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def summary(out: Path, name: str) -> dict[str, Any]:
    return json.loads((out / f"{name}_B_summary.json").read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_one_error_line(err: str) -> None:
    assert err.startswith("error: ") and err.count("\n") == 1 and err.endswith("\n"), err


@pytest.fixture
def done(mod: ModuleType, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    out = tmp_path / "out"
    code, _stdout, err = run(mod, capsys, argv_for(root, out))
    assert code == 0 and err == "", err
    return out


# --- what is written -------------------------------------------------------------------------------------------------

def test_writes_a_table_and_a_summary_per_eval_split_and_only_a_summary_for_train(done: Path) -> None:
    assert sorted(p.name for p in done.iterdir()) == [
        "default_train_B_summary.json",
        "test_B_noisy.tsv", "test_B_summary.json",
        "test_amb_B_noisy.tsv", "test_amb_B_summary.json",
        "validation_B_noisy.tsv", "validation_B_summary.json"]


def test_validation_table_is_headerless_in_input_order_with_one_row_per_rule(done: Path) -> None:
    text = (done / "validation_B_noisy.tsv").read_bytes().decode("utf-8")
    assert text == "".join(f"{i}\t{rule}\n" for i, rule in [
        ("val-0", "speaker_label"), ("val-1", "speaker_label"), ("val-2", "speaker_label"),
        ("val-4", "cue_number"), ("val-6", "speaker_label"), ("val-6", "auto_caption_video"),
        ("val-7", "auto_caption_video")])


def test_validation_summary_counts(done: Path) -> None:
    got = summary(done, "validation")
    assert got["records_per_rule"] == {"speaker_label": 4, "cue_number": 1, "auto_caption_video": 2}
    assert got["records"] == 8 and got["records_flagged"] == 6
    assert got["gold_colon_labels"] == 9 and got["speaker_label_colons"] == 6
    assert got["speaker_label_colon_share"] == pytest.approx(6 / 9, abs=1e-6)
    assert (got["release"], got["mode"], got["split"]) == (RELEASE, "B", "validation")


def test_summary_records_the_sha256_of_every_input(done: Path, root: Path) -> None:
    mode = Path("processed", RELEASE, "mode_B")
    assert summary(done, "validation")["input_sha256"] == {
        (mode / "validation.jsonl").as_posix(): sha(root / mode / "validation.jsonl"),
        f"splits/{RELEASE}/primary_eval/validation_mode_B.ids":
            sha(root / "splits" / RELEASE / "primary_eval" / "validation_mode_B.ids")}
    assert set(summary(done, "default_train")["input_sha256"]) == {
        f"processed/{RELEASE}/mode_B/train.jsonl", f"splits/{RELEASE}/default_train/mode_B.ids"}


def test_test_and_test_amb_summaries(done: Path) -> None:
    got = summary(done, "test")
    assert got["records"] == 2 and got["records_flagged"] == 1
    assert got["records_per_rule"] == {"speaker_label": 1, "cue_number": 0, "auto_caption_video": 0}
    assert (got["gold_colon_labels"], got["speaker_label_colons"], got["speaker_label_colon_share"]) == (1, 1, 1.0)
    assert (done / "test_B_noisy.tsv").read_text(encoding="utf-8") == "tst-0\tspeaker_label\n"
    amb = summary(done, "test_amb")
    assert amb["records"] == 1 and amb["records_flagged"] == 0 and amb["gold_colon_labels"] == 0
    assert amb["speaker_label_colon_share"] is None            # no colon labels: no share
    assert (done / "test_amb_B_noisy.tsv").read_bytes() == b""


def test_default_train_summary_counts_the_listed_records_with_the_whole_file_inventory(done: Path) -> None:
    got = summary(done, "default_train")
    assert got["split"] == "default_train" and got["records"] == 5
    assert got["records_per_rule"] == {"speaker_label": 4, "cue_number": 1, "auto_caption_video": 0}
    assert got["records_flagged"] == 5
    assert (got["gold_colon_labels"], got["speaker_label_colons"], got["speaker_label_colon_share"]) == (4, 4, 1.0)


def test_stdout_is_one_line_per_split(mod: ModuleType, root: Path, tmp_path: Path,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _err = run(mod, capsys, argv_for(root, tmp_path / "out"))
    assert code == 0
    lines = out.splitlines()
    assert [line.split()[0] for line in lines] == ["validation", "test", "test_amb", "default_train"]


# --- refusals --------------------------------------------------------------------------------------------------------

def refused(mod: ModuleType, capsys: pytest.CaptureFixture[str], argv: list[str], tmp_path: Path) -> str:
    code, out, err = run(mod, capsys, argv)
    assert code == 1 and out == ""
    assert_one_error_line(err)
    assert str(tmp_path) not in err and "Traceback" not in err
    return err


def test_existing_out_is_refused_and_left_alone(mod: ModuleType, root: Path, tmp_path: Path,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("keep", encoding="utf-8")
    err = refused(mod, capsys, argv_for(root, out), tmp_path)
    assert "out" in err and "already exists" in err
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_missing_input_names_the_file_and_leaves_nothing(mod: ModuleType, root: Path, tmp_path: Path,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    (root / "splits" / RELEASE / "primary_eval" / "test_amb_mode_B.ids").unlink()
    err = refused(mod, capsys, argv_for(root, tmp_path / "out"), tmp_path)
    assert "test_amb_mode_B.ids" in err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data"]       # no out, no temporary sibling


def test_listed_id_that_is_not_in_the_file_is_refused(mod: ModuleType, root: Path, tmp_path: Path,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    write_ids(root / "splits" / RELEASE / "primary_eval" / "test_mode_B.ids", ["tst-0", "nope-9"])
    err = refused(mod, capsys, argv_for(root, tmp_path / "out"), tmp_path)
    assert "test_mode_B.ids" in err and "nope-9" in err


def test_invalid_json_names_file_and_line(mod: ModuleType, root: Path, tmp_path: Path,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    path = root / "processed" / RELEASE / "mode_B" / "test.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{broken\n", encoding="utf-8", newline="\n")
    err = refused(mod, capsys, argv_for(root, tmp_path / "out"), tmp_path)
    assert "test.jsonl line 4: not valid JSON" in err                  # record, blank, record, then the bad line


@pytest.mark.parametrize(("bad_line", "reason"), [("[" * 50_000, "not valid JSON"), ("[1, 2]", "not a JSON object")])
def test_a_line_that_is_nested_too_deeply_or_not_an_object_names_file_and_line(
        mod: ModuleType, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], bad_line: str,
        reason: str) -> None:
    path = root / "processed" / RELEASE / "mode_B" / "test.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + bad_line + "\n", encoding="utf-8", newline="\n")
    err = refused(mod, capsys, argv_for(root, tmp_path / "out"), tmp_path)
    assert f"test.jsonl line 4: {reason}" in err


def test_record_without_normalized_text_names_file_and_line(mod: ModuleType, root: Path, tmp_path: Path,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    path = root / "processed" / RELEASE / "mode_B" / "test_amb.jsonl"
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    del record["metadata"]["normalized_text"]
    path.write_text(json.dumps(record) + "\n", encoding="utf-8", newline="\n")
    err = refused(mod, capsys, argv_for(root, tmp_path / "out"), tmp_path)
    assert "test_amb.jsonl line 1" in err and "normalized_text" in err


def test_duplicate_sample_id_is_refused(mod: ModuleType, root: Path, tmp_path: Path,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    path = root / "processed" / RELEASE / "mode_B" / "test_amb.jsonl"
    line = path.read_text(encoding="utf-8").splitlines()[0]
    path.write_text(line + "\n" + line + "\n", encoding="utf-8", newline="\n")
    err = refused(mod, capsys, argv_for(root, tmp_path / "out"), tmp_path)
    assert "test_amb.jsonl" in err and "amb-0" in err and "twice" in err


def test_non_utf8_input_is_refused(mod: ModuleType, root: Path, tmp_path: Path,
                                   capsys: pytest.CaptureFixture[str]) -> None:
    (root / "processed" / RELEASE / "mode_B" / "test.jsonl").write_bytes(b"\xff\xfe\x00")
    err = refused(mod, capsys, argv_for(root, tmp_path / "out"), tmp_path)
    assert "test.jsonl" in err and "UTF-8" in err


def test_duplicate_ids_in_a_list_are_refused(mod: ModuleType, root: Path, tmp_path: Path,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    write_ids(root / "splits" / RELEASE / "primary_eval" / "test_mode_B.ids", ["tst-0", "tst-0"])
    err = refused(mod, capsys, argv_for(root, tmp_path / "out"), tmp_path)
    assert "test_mode_B.ids" in err and "Duplicate ID 'tst-0'" in err


def test_non_utf8_id_list_is_refused(mod: ModuleType, root: Path, tmp_path: Path,
                                     capsys: pytest.CaptureFixture[str]) -> None:
    (root / "splits" / RELEASE / "primary_eval" / "test_mode_B.ids").write_bytes(b"\xff\xfe\x00")
    err = refused(mod, capsys, argv_for(root, tmp_path / "out"), tmp_path)
    assert "test_mode_B.ids" in err and "UTF-8" in err


def test_a_failed_final_rename_leaves_no_out_and_no_temporary_folder(
        mod: ModuleType, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"

    def deny(src: object, dst: object) -> None:
        raise PermissionError(13, "Permission denied", str(out))

    monkeypatch.setattr(mod.os, "replace", deny)
    err = refused(mod, capsys, argv_for(root, out), tmp_path)
    assert err == "error: out: Permission denied\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data"]


def nothing_else_in(tmp_path: Path, *kept: str) -> bool:
    return sorted(p.name for p in tmp_path.iterdir()) == sorted(["data", *kept])


def test_data_root_that_is_a_file_is_refused_before_scoring(mod: ModuleType, tmp_path: Path,
                                                            capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    err = refused(mod, capsys, argv_for(tmp_path / "notes.txt", tmp_path / "out"), tmp_path)
    assert err == "error: data root notes.txt is not a folder\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.txt"]


def test_data_root_that_is_missing_is_refused_before_scoring(mod: ModuleType, tmp_path: Path,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    err = refused(mod, capsys, argv_for(tmp_path / "nowhere", tmp_path / "out"), tmp_path)
    assert err == "error: data root nowhere is not a folder\n"
    assert list(tmp_path.iterdir()) == []


def test_out_whose_parent_is_a_file_is_refused_before_scoring(mod: ModuleType, root: Path, tmp_path: Path,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "blocker").write_text("x", encoding="utf-8")
    err = refused(mod, capsys, argv_for(root, tmp_path / "blocker" / "out"), tmp_path)
    assert err == "error: output parent blocker is not a folder\n"
    assert nothing_else_in(tmp_path, "blocker")


def test_out_whose_parent_is_missing_is_refused_and_not_created(mod: ModuleType, root: Path, tmp_path: Path,
                                                                capsys: pytest.CaptureFixture[str]) -> None:
    err = refused(mod, capsys, argv_for(root, tmp_path / "later" / "out"), tmp_path)
    assert err == "error: output parent later is not a folder\n"
    assert nothing_else_in(tmp_path)


def test_checks_run_before_any_input_is_read(mod: ModuleType, root: Path, tmp_path: Path,
                                             capsys: pytest.CaptureFixture[str],
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object) -> None:
        raise AssertionError("scoring started")

    monkeypatch.setattr(mod, "process_split", fail)
    refused(mod, capsys, argv_for(root, tmp_path / "later" / "out"), tmp_path)


@pytest.mark.parametrize("relative", [
    "processed/sponspeech-m1-v2/mode_B/validation.jsonl",
    "splits/sponspeech-m1-v2/primary_eval/validation_mode_B.ids",
])
def test_an_input_that_is_a_folder_says_so_not_permission_denied(
        mod: ModuleType, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], relative: str) -> None:
    path = root / relative
    path.unlink()
    path.mkdir()
    err = refused(mod, capsys, argv_for(root, tmp_path / "out"), tmp_path)
    assert err == f"error: {path.name}: is a folder, not a file\n"
    assert nothing_else_in(tmp_path)


def test_a_denied_rename_of_the_temporary_folder_still_says_permission_denied(
        mod: ModuleType, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The source of the rename is the temporary folder: two paths, so it is not 'a folder, not a file'."""
    def deny(src: object, dst: object) -> None:
        raise PermissionError(13, "Permission denied", str(src), None, str(dst))

    monkeypatch.setattr(mod.os, "replace", deny)
    err = refused(mod, capsys, argv_for(root, tmp_path / "out"), tmp_path)
    assert err.endswith(": Permission denied\n") and "folder" not in err
    assert nothing_else_in(tmp_path)


def test_a_missing_id_is_found_without_reading_the_file_a_third_time(
        mod: ModuleType, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    write_ids(root / "splits" / RELEASE / "primary_eval" / "test_mode_B.ids", ["tst-0", "nope-9"])
    calls = []
    real = mod._records

    def counting(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
        calls.append(path.name)
        return real(path)

    monkeypatch.setattr(mod, "_records", counting)
    err = refused(mod, capsys, argv_for(root, tmp_path / "out"), tmp_path)
    assert "nope-9" in err
    assert calls.count("test.jsonl") == 2                       # pass 1 (inventory) and pass 2 (scoring)


@pytest.mark.parametrize("extra", [[], ["--mode", "A"], ["--bogus"]])
def test_usage_errors_exit_2(mod: ModuleType, root: Path, tmp_path: Path, extra: list[str]) -> None:
    """No --out, a mode other than B and an unknown option are usage errors, not input errors."""
    argv = ["--data-root", str(root), *extra]
    if extra[:1] != []:
        argv += ["--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit) as caught:
        mod.main(argv)
    assert caught.value.code == 2
    assert not (tmp_path / "out").exists()


# --- as a user runs it -----------------------------------------------------------------------------------------------

def run_script(root: Path, out: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    """The stub in a fresh interpreter without PYTHONPATH (the installed package is found in site-packages)."""
    environ = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, str(SCRIPT), *argv_for(root, out)], cwd=cwd, env=environ,
                          capture_output=True, text=True, check=False)


def test_runs_as_a_subprocess_without_pythonpath(root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    first = run_script(root, out, tmp_path)
    assert first.returncode == 0 and first.stderr == "", first.stderr
    assert (out / "validation_B_noisy.tsv").is_file()
    second = run_script(root, out, tmp_path)                 # --out exists now
    assert second.returncode == 1 and second.stderr.startswith("error: ") and second.stderr.count("\n") == 1


def test_the_script_and_module_have_no_marker_or_bom() -> None:
    marker = "TO" + "DO"                                     # split so this file does not match itself
    for path in (SCRIPT, CLI_MODULE, ROOT / "src" / "punct_restore" / "data" / "gold_noise.py"):
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf") and marker not in raw.decode("utf-8")
