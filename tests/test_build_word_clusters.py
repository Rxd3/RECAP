"""punct_restore.cli.build_word_clusters on a tiny fake data root (release ``rel``, mode A)."""

from __future__ import annotations

import builtins
import fnmatch
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from punct_restore.cli import build_word_clusters as cli
from punct_restore.features.word_clusters import load_cluster_map

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_word_clusters.py"
REL = "rel"
SENTENCES = [
    ["the", "cat", "sat"],
    ["the", "dog", "sat"],
    ["we", "run", "fast"],
    ["we", "walk", "fast"],
]
# basename patterns of the files the CLI must never open (the tmp_path folders contain "test", so no substrings)
FORBIDDEN = ("validation.jsonl", "test.jsonl", "test_amb*", "*validation*.ids", "test*.ids")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _jsonl(records: list[dict]) -> str:
    return "".join(json.dumps(record) + "\n" for record in records)


def _record(sample_id: str, tokens: list[str]) -> dict:
    return {"sample_id": sample_id, "tokens": tokens, "metadata": {}}


def make_root(tmp_path: Path) -> Path:
    """Fake data root: 40 listed train records, one unlisted train record, and bait validation/test files."""
    root = tmp_path / "data"
    listed = [_record(f"s:{index}", tokens) for index, tokens in enumerate(SENTENCES * 10)]
    unlisted = _record("s:held", ["zzzword"] * 10)
    mode_dir = root / "processed" / REL / "mode_A"
    _write(mode_dir / "train.jsonl", _jsonl([unlisted, *listed]))
    for name in ("validation.jsonl", "test.jsonl", "test_amb.jsonl"):
        _write(mode_dir / name, _jsonl([_record("s:eval", ["baitword"] * 10)]))
    splits = root / "splits" / REL
    _write(splits / "default_train" / "mode_A.ids", "".join(f"{r['sample_id']}\n" for r in listed))
    for name in ("primary_eval/validation_A.ids", "primary_eval/test_A.ids", "leakage_controlled/test.ids"):
        _write(splits / name, "s:eval\n")
    return root


def args_for(root: Path, out: Path, *extra: str) -> list[str]:
    return ["--data-root", str(root), "--release", REL, "--mode", "A", "--k", "2", "--out", str(out), *extra]


def run(root: Path, out: Path, *extra: str) -> int:
    return cli.main(args_for(root, out, *extra))


def watch_opens(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every opened basename from now on; opening a validation/test file fails the test on the spot."""
    seen: list[str] = []
    real_open = io.open

    def guarded(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            name = os.path.basename(os.fsdecode(file))
            seen.append(name)
            if any(fnmatch.fnmatch(name, pattern) for pattern in FORBIDDEN):
                raise AssertionError(f"the CLI opened a forbidden file: {name}")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded)
    monkeypatch.setattr(io, "open", guarded)
    return seen


# ---------------------------------------------------------------- success

def test_builds_a_map_from_the_listed_train_records_only(tmp_path, capsys):
    root = make_root(tmp_path)
    out = tmp_path / "map.json"
    assert run(root, out, "--seed", "5", "--min-count", "3", "--dims", "20", "--window", "2") == 0
    loaded = load_cluster_map(out)
    ids_bytes = (root / "splits" / REL / "default_train" / "mode_A.ids").read_bytes()
    assert loaded.train_ids_sha256 == hashlib.sha256(ids_bytes).hexdigest()
    assert (loaded.release, loaded.mode) == (REL, "A")
    assert (loaded.k, loaded.seed, loaded.min_count, loaded.dims, loaded.window) == (2, 5, 3, 7, 2)  # dims 20 -> 8 - 1
    assert set(loaded.clusters) == {"the", "cat", "dog", "sat", "we", "run", "walk", "fast"}
    assert loaded.vocabulary_size == 8
    assert loaded.clusters["cat"] == loaded.clusters["dog"] != loaded.clusters["run"] == loaded.clusters["walk"]
    printed = capsys.readouterr()
    assert re.fullmatch(r"vocabulary_size=8 k=2 seconds=\d+\.\d\n", printed.out)
    assert printed.err == ""


def test_defaults_are_the_documented_ones(tmp_path):
    out = tmp_path / "map.json"
    assert run(make_root(tmp_path), out) == 0
    loaded = load_cluster_map(out)
    assert (loaded.seed, loaded.min_count, loaded.window) == (2026, 5, 2)
    assert cli.build_parser().parse_args(["--k", "2", "--out", "x"]).dims == 100


@pytest.mark.parametrize(("asked", "used"), [("3", 3), ("7", 7), ("8", 7), ("100", 7)])
def test_records_the_svd_dimensions_actually_used(tmp_path, asked, used):
    out = tmp_path / "map.json"  # the fixture vocabulary has 8 words, so at most 7 SVD components exist
    assert run(make_root(tmp_path), out, "--dims", asked) == 0
    assert load_cluster_map(out).dims == used


def test_never_opens_validation_or_test_files(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    opened = watch_opens(monkeypatch)
    assert run(root, tmp_path / "map.json") == 0
    assert "train.jsonl" in opened and "mode_A.ids" in opened
    assert [name for name in opened if any(fnmatch.fnmatch(name, p) for p in FORBIDDEN)] == []


def test_output_has_lf_newlines_and_the_same_bytes_every_run(tmp_path):
    root = make_root(tmp_path)
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    assert run(root, first) == 0 and run(root, second) == 0
    data = first.read_bytes()
    assert data == second.read_bytes()
    assert b"\r" not in data and data.endswith(b"\n")


def test_creates_the_missing_parent_folder(tmp_path):
    out = tmp_path / "maps" / "map.json"
    assert run(make_root(tmp_path), out) == 0
    assert out.is_file()


# ---------------------------------------------------------------- refusals

def _assert_one_line_error(capsys, tmp_path, fragment: str) -> None:
    printed = capsys.readouterr()
    lines = printed.err.splitlines()
    assert printed.out == "" and len(lines) == 1
    assert lines[0].startswith("error: ") and fragment in lines[0]
    assert str(tmp_path) not in lines[0] and "Traceback" not in printed.err


def test_refuses_to_overwrite_an_existing_out_before_reading_anything(tmp_path, capsys, monkeypatch):
    root = make_root(tmp_path)
    out = tmp_path / "map.json"
    out.write_text("keep me", encoding="utf-8")
    opened = watch_opens(monkeypatch)
    assert run(root, out) == 1
    _assert_one_line_error(capsys, tmp_path, "map.json already exists")
    assert out.read_text(encoding="utf-8") == "keep me"
    assert "train.jsonl" not in opened


def _drop_ids(root: Path) -> None:
    (root / "splits" / REL / "default_train" / "mode_A.ids").unlink()


def _drop_train(root: Path) -> None:
    (root / "processed" / REL / "mode_A" / "train.jsonl").unlink()


def _add_unknown_id(root: Path) -> None:
    path = root / "splits" / REL / "default_train" / "mode_A.ids"
    _write(path, path.read_text(encoding="utf-8") + "s:ghost\n")


def _duplicate_id(root: Path) -> None:
    path = root / "splits" / REL / "default_train" / "mode_A.ids"
    _write(path, path.read_text(encoding="utf-8") + "s:0\n")


def _empty_ids(root: Path) -> None:
    _write(root / "splits" / REL / "default_train" / "mode_A.ids", "\n")


def _append_train(root: Path, text: str) -> None:
    path = root / "processed" / REL / "mode_A" / "train.jsonl"
    _write(path, path.read_text(encoding="utf-8") + text)


NOT_UTF8 = bytes([0xFF, 0xFE])  # a UTF-8 decoder rejects both bytes


def _ids_not_utf8(root: Path) -> None:
    path = root / "splits" / REL / "default_train" / "mode_A.ids"
    path.write_bytes(path.read_bytes() + b"s:" + NOT_UTF8 + b"\n")


def _train_not_utf8(root: Path) -> None:
    path = root / "processed" / REL / "mode_A" / "train.jsonl"
    path.write_bytes(path.read_bytes() + b'{"sample_id": "s:bad", "tokens": ["' + NOT_UTF8 + b'"]}\n')


BROKEN = {
    "missing ids file": (_drop_ids, "mode_A.ids"),
    "missing train file": (_drop_train, "train.jsonl"),
    "id not in train": (_add_unknown_id, "train.jsonl: 1 ID"),
    "duplicate id in ids file": (_duplicate_id, "Duplicate ID 's:0' found at line 41 in mode_A.ids"),
    "empty ids file": (_empty_ids, "mode_A.ids: ID list is empty"),
    "ids file not utf-8": (_ids_not_utf8, "mode_A.ids: not valid UTF-8"),
    "train file not utf-8": (_train_not_utf8, "train.jsonl: not valid UTF-8"),
    "invalid json line": (lambda r: _append_train(r, "{oops\n"), "train.jsonl line 42: not valid JSON"),
    "record without sample_id": (lambda r: _append_train(r, '{"tokens": ["a"]}\n'),
                                 "train.jsonl line 42: sample_id must be a nonempty string"),
    "record that is not an object": (lambda r: _append_train(r, "[1, 2]\n"), "train.jsonl line 42: not a JSON object"),
    "record nested too deeply": (lambda r: _append_train(r, "[" * 50_000 + "\n"),
                                 "train.jsonl line 42: not valid JSON"),
    "duplicate listed record": (
        lambda r: _append_train(r, _jsonl([_record("s:0", ["a"])])), "train.jsonl line 42: duplicate"),
}


@pytest.mark.parametrize(("break_it", "fragment"), list(BROKEN.values()), ids=list(BROKEN))
def test_broken_inputs_fail_with_one_line_naming_files_only(tmp_path, capsys, break_it, fragment):
    root = make_root(tmp_path)
    break_it(root)
    out = tmp_path / "map.json"
    assert run(root, out) == 1
    _assert_one_line_error(capsys, tmp_path, fragment)
    assert not out.exists()


@pytest.mark.parametrize("record", [
    '{"sample_id": "s:bad"}',
    '{"sample_id": "s:bad", "tokens": "a b"}',
    '{"sample_id": "s:bad", "tokens": ["a", 3]}',
], ids=["no tokens", "tokens is a string", "token is an int"])
def test_a_listed_record_with_bad_tokens_is_rejected(tmp_path, capsys, record):
    root = make_root(tmp_path)
    path = root / "splits" / REL / "default_train" / "mode_A.ids"
    _write(path, path.read_text(encoding="utf-8") + "s:bad\n")
    _append_train(root, record + "\n")
    assert run(root, tmp_path / "map.json") == 1
    _assert_one_line_error(capsys, tmp_path, "train.jsonl line 42: tokens must be a list of strings")


def test_k_larger_than_the_vocabulary_fails_cleanly(tmp_path, capsys):
    out = tmp_path / "map.json"
    assert run(make_root(tmp_path), out, "--k", "99") == 1  # the later --k wins
    _assert_one_line_error(capsys, tmp_path, "vocabulary")
    assert not out.exists()


def test_invalid_parameters_fail_with_exit_one(tmp_path, capsys):
    assert run(make_root(tmp_path), tmp_path / "map.json", "--dims", "1") == 1
    _assert_one_line_error(capsys, tmp_path, "dims must be")


def test_a_seed_above_the_scikit_learn_limit_fails_with_exit_one(tmp_path, capsys):
    out = tmp_path / "map.json"
    assert run(make_root(tmp_path), out, "--seed", "4294967296") == 1
    _assert_one_line_error(capsys, tmp_path, "seed must be an integer in 0..4294967295")
    assert not out.exists()


def test_an_unwritable_out_parent_fails_cleanly(tmp_path, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder", encoding="utf-8")
    assert run(make_root(tmp_path), blocker / "map.json") == 1
    _assert_one_line_error(capsys, tmp_path, "blocker")


def test_a_failed_write_leaves_no_partial_file(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "map_to_json", lambda cluster_map: "half \ud800 map")  # not encodable as UTF-8
    out = tmp_path / "map.json"
    assert run(make_root(tmp_path), out) == 1
    _assert_one_line_error(capsys, tmp_path, "error: ")
    assert not out.exists()


@pytest.mark.parametrize("bad", [
    ["--k", "2"],  # no --out
    ["--out", "map.json"],  # no --k
    ["--k", "two", "--out", "map.json"],
    ["--k", "2", "--out", "map.json", "--mode", "D"],
])
def test_usage_errors_exit_with_two(bad):
    with pytest.raises(SystemExit) as caught:
        cli.main(bad)
    assert caught.value.code == 2


# ---------------------------------------------------------------- real process

def _subprocess(root: Path, out: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args_for(root, out)],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=300, check=False,
    )


def test_runs_as_a_script_without_pythonpath(tmp_path):
    root = make_root(tmp_path)
    out = tmp_path / "map.json"
    result = _subprocess(root, out, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("vocabulary_size=8 k=2 seconds=") and result.stderr == ""
    assert load_cluster_map(out).vocabulary_size == 8


def test_a_failing_script_run_prints_one_error_line(tmp_path):
    root = make_root(tmp_path)
    out = tmp_path / "map.json"
    out.write_text("keep me", encoding="utf-8")
    result = _subprocess(root, out, cwd=tmp_path)
    assert result.returncode == 1 and result.stdout == ""
    assert result.stderr.splitlines() == ["error: map.json already exists; refusing to overwrite"]
