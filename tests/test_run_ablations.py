"""Tests for scripts/run_ablations.py: seeded-sample CRF feature ablations (validation only)."""

from __future__ import annotations

import builtins
import fnmatch
import io
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import scripts.run_ablations as run_ablations

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "rel"

_SEQUENCES: dict[str, tuple[list[str], list[str]]] = {
    "t1": (["hello", "everyone", "welcome", "to", "the", "show"], ["O", "O", "O", "O", "O", "PERIOD"]),
    "t2": (["did", "you", "see", "it"], ["O", "O", "O", "QUESTION"]),
    "t3": (["well", "um", "i", "mean", "it", "works"], ["COMMA", "O", "O", "COMMA", "O", "PERIOD"]),
    "t4": (["yes", "we", "can", "go"], ["COMMA", "O", "O", "PERIOD"]),
    "t5": (["what", "is", "this"], ["O", "O", "QUESTION"]),
    "t6": (["thank", "you", "very", "much"], ["O", "O", "O", "PERIOD"]),
    "v1": (["so", "you", "know", "it", "works"], ["O", "COMMA", "O", "O", "PERIOD"]),
    "v2": (["are", "you", "ready"], ["O", "O", "QUESTION"]),
    "v3": (["we", "are", "done", "here"], ["O", "O", "O", "PERIOD"]),
}
TRAIN_IDS = ["t1", "t2", "t3", "t4", "t5", "t6"]
VAL_IDS = ["v1", "v2", "v3"]


def _row(sample_id: str, split: str) -> dict[str, Any]:
    tokens, punct = _SEQUENCES[sample_id]
    return {
        "sample_id": sample_id,
        "split": split,
        "label_mode": "A",
        "tokens": tokens,
        "punct_labels": punct,
        "case_labels": ["LOWER"] * len(tokens),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")


def _write_ids(path: Path, ids: list[str]) -> None:
    """Write an ID file as bytes with CRLF line endings and a trailing blank line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\r\n".join(i.encode() for i in ids) + b"\r\n\r\n")


def build_root(
    tmp_path: Path,
    *,
    train_listed: list[str] = TRAIN_IDS,
    train_present: list[str] = TRAIN_IDS,
    val_listed: list[str] = VAL_IDS,
    val_present: list[str] = VAL_IDS,
) -> Path:
    """Tiny data root for release 'rel', mode A, including test files that must never be opened."""
    root = tmp_path / "data"
    proc = root / "processed" / RELEASE / "mode_A"
    splits = root / "splits" / RELEASE
    _write_jsonl(proc / "train.jsonl", [_row(i, "train") for i in train_present])
    _write_jsonl(proc / "validation.jsonl", [_row(i, "validation") for i in val_present])
    _write_jsonl(proc / "test.jsonl", [_row(i, "test") for i in VAL_IDS])
    _write_ids(splits / "default_train" / "mode_A.ids", train_listed)
    _write_ids(splits / "primary_eval" / "validation_mode_A.ids", val_listed)
    _write_ids(splits / "primary_eval" / "test_mode_A.ids", VAL_IDS)
    _write_ids(splits / "primary_eval" / "test_amb_mode_A.ids", VAL_IDS)
    return root


def _argv(root: Path, out: Path, *extra: str) -> list[str]:
    return ["--mode", "A", "--out", str(out), "--data-root", str(root), "--release", RELEASE, *extra]


def _forbidden(path: object) -> bool:
    if not isinstance(path, (str, os.PathLike)):
        return False
    name = os.path.basename(os.fsdecode(path))
    return (
        name == "test.jsonl"
        or name.startswith("test_amb")
        or fnmatch.fnmatch(name, "test_mode_*.ids")
        or fnmatch.fnmatch(name, "test_amb_mode_*.ids")
    )


def forbid_test_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if anything opens a held-out test file (by basename, not by 'test' substring).

    Install it after the data root is built: building the fixture itself writes those files.
    """
    real_open = io.open

    def guarded(file: Any, *args: Any, **kwargs: Any) -> Any:
        if _forbidden(file):
            raise AssertionError(f"test data must never be opened: {os.path.basename(os.fsdecode(file))}")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded)
    monkeypatch.setattr(io, "open", guarded)


def test_variants_are_the_eight_expected_names_in_order():
    assert list(run_ablations.VARIANTS) == [
        "improved",
        "improved_minus_window2",
        "improved_minus_ngrams",
        "improved_minus_question",
        "improved_minus_clause",
        "improved_minus_casing",
        "improved_minus_position",
        "improved_plus_filler",
    ]
    assert run_ablations.VARIANTS["improved"] == {}
    assert run_ablations.VARIANTS["improved_minus_window2"] == {"window_size": 1}
    assert run_ablations.VARIANTS["improved_plus_filler"] == {"use_filler_cues": True}
    assert run_ablations.SEED == 2026


def test_run_ablations_reads_crlf_ids_and_never_opens_test(tmp_path, monkeypatch):
    root = build_root(tmp_path)
    out = tmp_path / "out"
    forbid_test_files(monkeypatch)
    held_out = root / "processed" / RELEASE / "mode_A" / "test.jsonl"
    with pytest.raises(AssertionError, match="never be opened"), open(held_out, encoding="utf-8"):
        pass  # pragma: no cover - the guard raises before a file object exists

    code = run_ablations.main(
        _argv(root, out, "--sample", "4", "--variants", "improved,improved_plus_filler")
    )

    assert code == 0
    sample_bytes = (out / "mode_A" / "sample.ids").read_bytes()
    assert b"\r" not in sample_bytes
    sample_ids = sample_bytes.decode("utf-8").split("\n")
    assert sample_ids.pop() == ""  # file ends with exactly one newline
    assert sample_ids == sorted(random.Random(2026).sample(sorted(TRAIN_IDS), 4))
    assert len(sample_ids) == 4
    for variant in ("improved", "improved_plus_filler"):
        path = out / "mode_A" / variant / "validation.jsonl"
        preds = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [p["sample_id"] for p in preds] == VAL_IDS
        assert {p["model_name"] for p in preds} == {f"crf_{variant}"}
        assert {p["label_mode"] for p in preds} == {"A"}
        assert all(len(p["predicted_labels"]) == len(_SEQUENCES[p["sample_id"]][0]) for p in preds)
    assert not (out / "mode_A" / "improved_minus_ngrams").exists()


def test_progress_line_per_variant_goes_to_stdout(tmp_path, capsys):
    root = build_root(tmp_path)
    code = run_ablations.main(_argv(root, tmp_path / "out", "--sample", "3", "--variants", "improved"))
    captured = capsys.readouterr()
    assert code == 0
    assert re.fullmatch(r"A improved: \d+s\n", captured.out)
    assert captured.err == ""


@pytest.mark.parametrize("variants", ["bogus", "improved,improved", "improved,", ""])
def test_bad_variants_are_an_argparse_error(tmp_path, variants):
    root = build_root(tmp_path)
    with pytest.raises(SystemExit) as exc:
        run_ablations.main(_argv(root, tmp_path / "out", "--variants", variants))
    assert exc.value.code == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("sample", ["0", "-5", "many"])
def test_non_positive_sample_is_an_argparse_error(tmp_path, sample):
    root = build_root(tmp_path)
    with pytest.raises(SystemExit) as exc:
        run_ablations.main(_argv(root, tmp_path / "out", "--sample", sample))
    assert exc.value.code == 2


def test_existing_output_dir_fails_before_training(tmp_path, monkeypatch, capsys):
    root = build_root(tmp_path)
    out = tmp_path / "out"
    (out / "mode_A").mkdir(parents=True)

    def boom(**_: Any) -> None:
        raise AssertionError("trained although the output directory exists")

    monkeypatch.setattr(run_ablations, "CRFModel", boom)
    code = run_ablations.main(_argv(root, out, "--sample", "3"))

    err = capsys.readouterr().err
    assert code == 1
    assert len(err.splitlines()) == 1 and err.startswith("error: ")
    assert str(tmp_path) not in err
    assert list((out / "mode_A").iterdir()) == []


def test_output_dir_created_while_loading_still_fails_before_training(tmp_path, monkeypatch, capsys):
    root = build_root(tmp_path)
    out = tmp_path / "out"
    real_load_inputs = run_ablations.load_inputs

    def load_then_lose_the_race(args: Any) -> Any:
        loaded = real_load_inputs(args)
        (out / "mode_A").mkdir(parents=True)  # another run claims the folder meanwhile
        return loaded

    def boom(**_: Any) -> None:
        raise AssertionError("trained although the output directory exists")

    monkeypatch.setattr(run_ablations, "load_inputs", load_then_lose_the_race)
    monkeypatch.setattr(run_ablations, "CRFModel", boom)

    assert run_ablations.main(_argv(root, out, "--sample", "3")) == 1
    assert capsys.readouterr().err.splitlines() == [
        "error: mode_A already exists under --out; remove it or pick another --out"
    ]


def test_existing_output_dir_fails_before_any_input_is_read(tmp_path, capsys):
    root = build_root(tmp_path)
    (root / "processed" / RELEASE / "mode_A" / "train.jsonl").unlink()
    out = tmp_path / "out"
    (out / "mode_A").mkdir(parents=True)

    code = run_ablations.main(_argv(root, out, "--sample", "3"))

    assert code == 1
    assert capsys.readouterr().err.splitlines() == [
        "error: mode_A already exists under --out; remove it or pick another --out"
    ]


def test_sample_larger_than_train_ids_is_an_error(tmp_path, capsys):
    root = build_root(tmp_path)
    out = tmp_path / "out"
    code = run_ablations.main(_argv(root, out, "--sample", "7"))
    err = capsys.readouterr().err
    assert code == 1
    assert len(err.splitlines()) == 1 and err.startswith("error: ")
    assert not out.exists()


def test_validation_id_missing_from_jsonl_is_an_error(tmp_path, capsys):
    root = build_root(tmp_path, val_listed=VAL_IDS + ["v9"])
    out = tmp_path / "out"
    code = run_ablations.main(_argv(root, out, "--sample", "3"))
    err = capsys.readouterr().err
    assert code == 1
    assert len(err.splitlines()) == 1 and "validation.jsonl" in err
    assert str(tmp_path) not in err
    assert not out.exists()


def test_train_id_missing_from_jsonl_is_an_error(tmp_path, capsys):
    root = build_root(tmp_path, train_present=TRAIN_IDS[:-1])
    code = run_ablations.main(_argv(root, tmp_path / "out", "--sample", "6"))
    err = capsys.readouterr().err
    assert code == 1
    assert len(err.splitlines()) == 1 and "train.jsonl" in err


def test_missing_input_file_is_one_friendly_line(tmp_path, capsys):
    root = build_root(tmp_path)
    (root / "splits" / RELEASE / "default_train" / "mode_A.ids").unlink()
    code = run_ablations.main(_argv(root, tmp_path / "out", "--sample", "3"))
    err = capsys.readouterr().err
    assert code == 1
    assert len(err.splitlines()) == 1 and err.startswith("error: ")
    assert "mode_A.ids" in err
    assert str(tmp_path) not in err


def test_malformed_jsonl_is_one_friendly_line(tmp_path, capsys):
    root = build_root(tmp_path)
    proc = root / "processed" / RELEASE / "mode_A"
    (proc / "validation.jsonl").write_text('{"sample_id": "v1"\n', encoding="utf-8")
    code = run_ablations.main(_argv(root, tmp_path / "out", "--sample", "3"))
    err = capsys.readouterr().err
    assert code == 1
    assert len(err.splitlines()) == 1 and "validation.jsonl" in err
    assert str(tmp_path) not in err


def test_jsonl_row_without_sample_id_is_an_error(tmp_path, capsys):
    root = build_root(tmp_path)
    proc = root / "processed" / RELEASE / "mode_A"
    (proc / "train.jsonl").write_text('{"tokens": ["a"]}\n', encoding="utf-8")
    code = run_ablations.main(_argv(root, tmp_path / "out", "--sample", "3"))
    err = capsys.readouterr().err
    assert code == 1
    assert len(err.splitlines()) == 1 and "train.jsonl" in err


@pytest.mark.parametrize("bad_id", [123, "", None, ["a"], True])
def test_jsonl_sample_id_must_be_a_non_empty_string(tmp_path, capsys, bad_id):
    root = build_root(tmp_path)
    proc = root / "processed" / RELEASE / "mode_A"
    _write_jsonl(proc / "train.jsonl", [{"sample_id": bad_id}])
    code = run_ablations.main(_argv(root, tmp_path / "out", "--sample", "3"))
    assert code == 1
    assert capsys.readouterr().err.splitlines() == [
        "error: train.jsonl line 1: sample_id must be a non-empty string"
    ]


@pytest.mark.parametrize(("repeated", "line"), [("v1", 4), ("v9", 5)])  # selected / unselected ID
def test_jsonl_duplicate_sample_id_is_an_error(tmp_path, capsys, repeated, line):
    root = build_root(tmp_path)
    extra = {"sample_id": repeated, "tokens": ["a"]}
    rows = [_row(i, "validation") for i in VAL_IDS] + [extra]
    if repeated == "v9":  # not in the ID file: still checked, the first copy sits on line 1
        rows.insert(0, extra)
    _write_jsonl(root / "processed" / RELEASE / "mode_A" / "validation.jsonl", rows)
    code = run_ablations.main(_argv(root, tmp_path / "out", "--sample", "3"))
    assert code == 1
    assert capsys.readouterr().err.splitlines() == [
        f"error: validation.jsonl line {line}: duplicate sample_id {repeated!r}"
    ]


def test_script_runs_standalone_and_has_no_personal_paths():
    script = ROOT / "scripts" / "run_ablations.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    )
    assert result.returncode == 0, result.stderr
    assert "--data-root" in result.stdout and "--release" in result.stdout
    assert "--code" not in result.stdout
    source = script.read_text(encoding="utf-8")
    assert "Asus" not in source and not re.search(r"[A-Za-z]:[/\\]", source)
