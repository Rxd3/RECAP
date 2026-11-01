"""Tests for punct_restore.cli.run_ablations: bad options, missing or damaged inputs and a taken output folder."""

from __future__ import annotations

from typing import Any

import pytest
from run_ablations_fixtures import RELEASE, TRAIN_IDS, VAL_IDS, _argv, _row, _write_jsonl, build_root

from punct_restore.cli import run_ablations


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


def test_a_repeated_id_in_an_id_list_is_one_friendly_line(tmp_path, capsys):
    root = build_root(tmp_path)
    ids = root / "splits" / RELEASE / "default_train" / "mode_A.ids"
    ids.write_bytes(ids.read_bytes() + ids.read_bytes().splitlines()[0] + b"\r\n")
    code = run_ablations.main(_argv(root, tmp_path / "out", "--sample", "3"))
    err = capsys.readouterr().err
    assert code == 1
    assert len(err.splitlines()) == 1 and err.startswith("error: Duplicate ID ") and "in mode_A.ids" in err
    assert str(tmp_path) not in err
    assert not (tmp_path / "out").exists()


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


@pytest.mark.parametrize(("bad_line", "reason"), [
    ('{"sample_id": "v1"', "not valid JSON"),
    ("[" * 50_000, "not valid JSON"),  # json gives up with a RecursionError
    ("[1, 2]", "not a JSON object"),
])
def test_a_bad_jsonl_line_names_the_file_and_line(tmp_path, capsys, bad_line, reason):
    root = build_root(tmp_path)
    path = root / "processed" / RELEASE / "mode_A" / "validation.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + bad_line + "\n", encoding="utf-8", newline="\n")
    code = run_ablations.main(_argv(root, tmp_path / "out", "--sample", "3"))
    assert code == 1
    assert capsys.readouterr().err.splitlines() == [f"error: validation.jsonl line 4: {reason}"]


def test_a_jsonl_file_that_is_not_utf8_names_the_file(tmp_path, capsys):
    root = build_root(tmp_path)
    (root / "processed" / RELEASE / "mode_A" / "train.jsonl").write_bytes(b"\xe9\xff\n")
    code = run_ablations.main(_argv(root, tmp_path / "out", "--sample", "3"))
    assert code == 1
    assert capsys.readouterr().err.splitlines() == ["error: train.jsonl: not valid UTF-8"]


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
        "error: train.jsonl line 1: sample_id must be a nonempty string"
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
