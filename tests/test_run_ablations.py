"""Tests for punct_restore.cli.run_ablations: seeded-sample CRF feature ablations (validation only)."""

from __future__ import annotations

import builtins
import fnmatch
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from punct_restore.cli import run_ablations
from punct_restore.features.crf_features import FeatureConfig
from punct_restore.features.word_clusters import ClusterMap, map_to_json

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "rel"
CLUSTERS_VARIANT = "improved_plus_clusters"

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


def _row(sample_id: str, split: str, mode: str = "A") -> dict[str, Any]:
    tokens, punct = _SEQUENCES[sample_id]
    case = ["FIRST_UPPER"] + ["LOWER"] * (len(tokens) - 1) if mode == "C" else ["LOWER"] * len(tokens)
    return {
        "sample_id": sample_id,
        "split": split,
        "label_mode": mode,
        "tokens": tokens,
        "punct_labels": punct,
        "case_labels": case,
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
    mode: str = "A",
) -> Path:
    """Tiny data root for release 'rel' and one mode, including test files that must never be opened."""
    root = tmp_path / "data"
    proc = root / "processed" / RELEASE / f"mode_{mode}"
    splits = root / "splits" / RELEASE
    _write_jsonl(proc / "train.jsonl", [_row(i, "train", mode) for i in train_present])
    _write_jsonl(proc / "validation.jsonl", [_row(i, "validation", mode) for i in val_present])
    _write_jsonl(proc / "test.jsonl", [_row(i, "test", mode) for i in VAL_IDS])
    _write_ids(splits / "default_train" / f"mode_{mode}.ids", train_listed)
    _write_ids(splits / "primary_eval" / f"validation_mode_{mode}.ids", val_listed)
    _write_ids(splits / "primary_eval" / f"test_mode_{mode}.ids", VAL_IDS)
    _write_ids(splits / "primary_eval" / f"test_amb_mode_{mode}.ids", VAL_IDS)
    return root


def _argv(root: Path, out: Path, *extra: str, mode: str = "A") -> list[str]:
    return ["--mode", mode, "--out", str(out), "--data-root", str(root), "--release", RELEASE, *extra]


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


def test_variants_are_the_thirteen_expected_names_in_order():
    assert list(run_ablations.VARIANTS) == [
        "improved",
        "improved_minus_window2",
        "improved_minus_ngrams",
        "improved_minus_question",
        "improved_minus_clause",
        "improved_minus_casing",
        "improved_minus_position",
        "improved_plus_filler",
        "improved_plus_question2",
        "improved_plus_clusters",
        "gap_improved",
        "case_improved",
        "factored",
    ]
    assert run_ablations.VARIANTS["improved"] == {}
    assert run_ablations.VARIANTS["improved_minus_window2"] == {"window_size": 1}
    assert run_ablations.VARIANTS["improved_plus_filler"] == {"use_filler_cues": True}
    assert run_ablations.VARIANTS["improved_plus_question2"] == {"use_question_lookback": True}
    assert run_ablations.VARIANTS["improved_plus_clusters"] == {"use_word_clusters": True}  # map added at run time
    assert run_ablations.VARIANTS["gap_improved"] == {}  # the improved features, no removed group
    assert run_ablations.VARIANTS["case_improved"] == {} and run_ablations.VARIANTS["factored"] == {}
    assert run_ablations.SEED == 2026


def test_run_ablations_reads_crlf_ids_and_never_opens_test(tmp_path, monkeypatch):
    root = build_root(tmp_path)
    out = tmp_path / "out"
    forbid_test_files(monkeypatch)
    held_out = root / "processed" / RELEASE / "mode_A" / "test.jsonl"
    with pytest.raises(AssertionError, match="never be opened"), open(held_out, encoding="utf-8"):
        pass  # pragma: no cover - the guard raises before a file object exists

    code = run_ablations.main(
        _argv(root, out, "--sample", "4", "--variants", "improved,improved_plus_filler,improved_plus_question2")
    )

    assert code == 0
    sample_bytes = (out / "mode_A" / "sample.ids").read_bytes()
    assert b"\r" not in sample_bytes
    sample_ids = sample_bytes.decode("utf-8").split("\n")
    assert sample_ids.pop() == ""  # file ends with exactly one newline
    assert sample_ids == sorted(random.Random(2026).sample(sorted(TRAIN_IDS), 4))
    assert len(sample_ids) == 4
    for variant in ("improved", "improved_plus_filler", "improved_plus_question2"):
        path = out / "mode_A" / variant / "validation.jsonl"
        preds = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [p["sample_id"] for p in preds] == VAL_IDS
        assert {p["model_name"] for p in preds} == {f"crf_{variant}"}
        assert {p["label_mode"] for p in preds} == {"A"}
        assert all(len(p["predicted_labels"]) == len(_SEQUENCES[p["sample_id"]][0]) for p in preds)
    assert not (out / "mode_A" / "improved_minus_ngrams").exists()


def test_gap_improved_runs_the_gap_classifier_not_the_crf(tmp_path, monkeypatch):
    root = build_root(tmp_path)
    out = tmp_path / "out"

    def boom(**_: Any) -> None:
        raise AssertionError("gap_improved must not build a CRFModel")

    monkeypatch.setattr(run_ablations, "CRFModel", boom)
    code = run_ablations.main(_argv(root, out, "--sample", "4", "--variants", "gap_improved"))

    assert code == 0
    path = out / "mode_A" / "gap_improved" / "validation.jsonl"
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    preds = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    assert [p["sample_id"] for p in preds] == VAL_IDS
    assert {p["model_name"] for p in preds} == {"gap_improved"}
    assert {p["label_mode"] for p in preds} == {"A"}
    assert all(len(p["predicted_labels"]) == len(_SEQUENCES[p["sample_id"]][0]) for p in preds)
    assert all(label in {"O", "COMMA", "PERIOD", "QUESTION"} for p in preds for label in p["predicted_labels"])


def test_gap_classifier_is_built_with_the_improved_features_and_explicit_hyperparameters(tmp_path, monkeypatch):
    root = build_root(tmp_path)
    built: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    real = run_ablations.GapClassifier

    def spy(*args: Any, **kwargs: Any) -> Any:
        built.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(run_ablations, "GapClassifier", spy)
    code = run_ablations.main(_argv(root, tmp_path / "out", "--sample", "4", "--variants", "gap_improved"))

    assert code == 0
    assert built == [
        (
            ("A",),
            {"feature_config": FeatureConfig.improved(), "C": 1.0, "max_iter": 200, "model_name": "gap_improved"},
        )
    ]


def test_gap_improved_and_improved_share_the_sample_and_the_output_layout(tmp_path):
    root = build_root(tmp_path)
    out = tmp_path / "out"
    code = run_ablations.main(_argv(root, out, "--sample", "4", "--variants", "improved,gap_improved"))
    assert code == 0
    models = {}
    for variant in ("improved", "gap_improved"):
        lines = (out / "mode_A" / variant / "validation.jsonl").read_text(encoding="utf-8").splitlines()
        models[variant] = {json.loads(line)["model_name"] for line in lines}
    assert models == {"improved": {"crf_improved"}, "gap_improved": {"gap_improved"}}
    assert (out / "mode_A" / "sample.ids").is_file()


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


def test_casing_helper_mode_is_not_an_ablation_mode(tmp_path, capsys):
    root = build_root(tmp_path)
    with pytest.raises(SystemExit) as exc:
        run_ablations.main(_argv(root, tmp_path / "out", mode="CASE"))
    assert exc.value.code == 2
    assert "--mode" in capsys.readouterr().err
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


def test_help_says_what_mode_does_and_that_the_cluster_variant_needs_a_cluster_map(capsys):
    with pytest.raises(SystemExit) as exit_info:
        run_ablations.parse_args(["--help"])
    assert exit_info.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "label mode to train and predict" in text
    assert f"{CLUSTERS_VARIANT} runs only with --cluster-map" in text


def test_script_runs_standalone_and_has_no_personal_paths():
    stub = ROOT / "scripts" / "run_ablations.py"
    module = ROOT / "src" / "punct_restore" / "cli" / "run_ablations.py"
    result = subprocess.run(
        [sys.executable, str(stub), "--help"],
        capture_output=True,
        text=True,
        check=False,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    )
    assert result.returncode == 0, result.stderr
    assert "--data-root" in result.stdout and "--release" in result.stdout
    assert "--cluster-map" in result.stdout
    assert "--code" not in result.stdout
    sources = [path.read_text(encoding="utf-8") for path in (stub, module)]
    assert not any(re.search(r"[A-Za-z]:[/\\]", source) for source in sources)


# --- mode C only variants: case_improved (CASE CRF) and factored (mode A CRF + CASE -> joint labels) -------------

C_ONLY = ["case_improved", "factored"]
A_LABELS = {"O", "COMMA", "PERIOD", "QUESTION"}
CASING = {"LOWER", "FIRST_UPPER", "ALL_UPPER"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_default_variants_leave_out_the_mode_c_only_ones_for_a_and_b_and_keep_them_in_order_for_c(tmp_path):
    root = build_root(tmp_path)
    today = [v for v in run_ablations.VARIANTS if v != CLUSTERS_VARIANT]  # no --cluster-map: defaults unchanged
    assert len(today) == 12
    for mode in ("A", "B"):
        chosen = run_ablations.parse_args(_argv(root, tmp_path / "out", mode=mode)).variants
        assert chosen == [v for v in today if v not in C_ONLY] and len(chosen) == 10
    chosen = run_ablations.parse_args(_argv(root, tmp_path / "out", mode="C")).variants
    assert chosen == today
    assert chosen.index("case_improved") < chosen.index("factored")


def test_explicit_variants_are_kept_as_given(tmp_path):
    root = build_root(tmp_path)
    args = run_ablations.parse_args(_argv(root, tmp_path / "out", "--variants", "improved,case_improved,factored",
                                          mode="C"))
    assert args.variants == ["improved", "case_improved", "factored"]


def test_mode_c_runs_improved_case_improved_and_factored_with_consistent_files(tmp_path, capsys, monkeypatch):
    root = build_root(tmp_path, mode="C")
    out = tmp_path / "out"
    forbid_test_files(monkeypatch)

    code = run_ablations.main(
        _argv(root, out, "--sample", "4", "--variants", "improved,case_improved,factored", mode="C")
    )

    captured = capsys.readouterr()
    assert code == 0 and captured.err == ""
    assert re.fullmatch(r"C improved: \d+s\nC case_improved: \d+s\nC factored: \d+s\n", captured.out)
    base = out / "mode_C"
    assert sorted(path.relative_to(base).as_posix() for path in base.rglob("*") if path.is_file()) == [
        "case_improved/validation.jsonl",
        "factored/punct_A.jsonl",
        "factored/validation.jsonl",
        "improved/validation.jsonl",
        "sample.ids",
    ]
    lengths = {sid: len(_SEQUENCES[sid][0]) for sid in VAL_IDS}
    expected_files = {
        "improved/validation.jsonl": ("C", "crf_improved"),
        "case_improved/validation.jsonl": ("CASE", "crf_case_improved"),
        "factored/punct_A.jsonl": ("A", "crf_improved_on_C_sample"),
        "factored/validation.jsonl": ("C", "crf_factored"),
    }
    rows = {}
    for name, (label_mode, model_name) in expected_files.items():
        rows[name] = _read_jsonl(base / name)
        assert [r["sample_id"] for r in rows[name]] == VAL_IDS, name
        assert {r["label_mode"] for r in rows[name]} == {label_mode}, name
        assert {r["model_name"] for r in rows[name]} == {model_name}, name
        assert all(len(r["predicted_labels"]) == lengths[r["sample_id"]] for r in rows[name]), name
    assert {lbl for r in rows["case_improved/validation.jsonl"] for lbl in r["predicted_labels"]} <= CASING
    assert {lbl for r in rows["factored/punct_A.jsonl"] for lbl in r["predicted_labels"]} <= A_LABELS
    for joint, punct, case in zip(
        rows["factored/validation.jsonl"],
        rows["factored/punct_A.jsonl"],
        rows["case_improved/validation.jsonl"],
        strict=False,
    ):
        assert joint["predicted_labels"] == [
            f"{p}|{c}" for p, c in zip(punct["predicted_labels"], case["predicted_labels"], strict=False)
        ]


def test_case_improved_and_factored_build_the_documented_crfs_on_the_same_sample(tmp_path, monkeypatch):
    root = build_root(tmp_path, mode="C")
    built: list[tuple[dict[str, Any], list[str]]] = []
    real = run_ablations.CRFModel

    def spy(**kwargs: Any) -> Any:
        model = real(**kwargs)
        real_fit = model.fit

        def fit(records: Any, *args: Any, **kw: Any) -> Any:
            built.append((kwargs, [r["sample_id"] for r in records]))
            return real_fit(records, *args, **kw)

        model.fit = fit
        return model

    monkeypatch.setattr(run_ablations, "CRFModel", spy)
    argv = _argv(root, tmp_path / "out", "--sample", "4", "--variants", "case_improved,factored", mode="C")
    assert run_ablations.main(argv) == 0

    sample = (tmp_path / "out" / "mode_C" / "sample.ids").read_text(encoding="utf-8").split()
    common = {"c1": 0.1, "c2": 0.1, "max_iterations": 100, "feature_config": FeatureConfig.improved()}
    assert built == [
        ({**common, "label_mode": "CASE", "model_name": "crf_case_improved"}, sample),
        ({**common, "label_mode": "A", "model_name": "crf_improved_on_C_sample"}, sample),
    ]


ONLY_MODE_C = "is only valid with --mode C"
CASE_FIRST = "factored needs case_improved earlier in --variants"


@pytest.mark.parametrize(
    ("mode", "variants", "message"),
    [
        ("C", "factored", CASE_FIRST),
        ("C", "improved,factored", CASE_FIRST),
        ("C", "factored,case_improved", CASE_FIRST),
        ("A", "case_improved", "case_improved " + ONLY_MODE_C),
        ("A", "improved,factored", "factored " + ONLY_MODE_C),
        ("A", "case_improved,factored", "case_improved " + ONLY_MODE_C),
        ("B", "case_improved", "case_improved " + ONLY_MODE_C),
        ("B", "factored", "factored " + ONLY_MODE_C),
    ],
)
def test_mode_c_only_variants_are_an_argparse_error_elsewhere_or_without_case_first(
        tmp_path, capsys, mode, variants, message):
    root = build_root(tmp_path)
    with pytest.raises(SystemExit) as exc:
        run_ablations.main(_argv(root, tmp_path / "out", "--variants", variants, mode=mode))
    assert exc.value.code == 2
    assert f"--variants: {message}" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_factored_fails_with_one_line_when_the_case_predictions_are_unusable(tmp_path, capsys, monkeypatch):
    root = build_root(tmp_path, mode="C")
    real = run_ablations.run_variant

    def run_then_empty_the_case_file(mode: str, variant: str, *args: Any, **kwargs: Any) -> None:
        real(mode, variant, *args, **kwargs)
        if variant == "case_improved":
            (args[-1] / "validation.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(run_ablations, "run_variant", run_then_empty_the_case_file)
    code = run_ablations.main(
        _argv(root, tmp_path / "out", "--sample", "3", "--variants", "case_improved,factored", mode="C")
    )

    err = capsys.readouterr().err
    assert code == 1
    assert len(err.splitlines()) == 1 and err.startswith("error: ") and "validation.jsonl" in err
    assert str(tmp_path) not in err
    assert not (tmp_path / "out" / "mode_C" / "factored" / "validation.jsonl").exists()


# --- improved_plus_clusters: opt-in word-cluster variant, fed by --cluster-map -----------------------------------

TINY_CLUSTERS = {"hello": 1, "everyone": 1, "did": 2, "you": 2, "see": 3, "it": 3, "we": 4, "are": 5}


def ids_sha(root: Path, mode: str = "A") -> str:
    return hashlib.sha256((root / "splits" / RELEASE / "default_train" / f"mode_{mode}.ids").read_bytes()).hexdigest()


def fake_cluster_loader(
    monkeypatch,
    root: Path,
    *,
    mode: str = "A",
    release: str = RELEASE,
    sha: str | None = None,
    clusters: dict[str, int] | None = None,
) -> list[Any]:
    """Replace the cluster-map loader with a fake map (by default one that matches the fixture's release and
    default_train IDs); returns the list of paths it was asked to load."""
    asked: list[Any] = []
    loaded = SimpleNamespace(
        mode=mode,
        release=release,
        train_ids_sha256=ids_sha(root) if sha is None else sha,
        clusters=TINY_CLUSTERS if clusters is None else clusters,
    )

    def load(path: Any) -> Any:
        asked.append(path)
        return loaded

    monkeypatch.setattr(run_ablations, "load_cluster_map", load)
    return asked


def test_clusters_variant_trains_with_the_loaded_map_and_is_written_like_the_others(tmp_path, monkeypatch):
    root = build_root(tmp_path)
    out = tmp_path / "out"
    asked = fake_cluster_loader(monkeypatch, root)
    built: list[dict[str, Any]] = []
    real = run_ablations.CRFModel

    def spy(**kwargs: Any) -> Any:
        built.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(run_ablations, "CRFModel", spy)
    forbid_test_files(monkeypatch)

    code = run_ablations.main(
        _argv(root, out, "--sample", "4", "--cluster-map", "tiny.json", "--variants", f"improved,{CLUSTERS_VARIANT}")
    )

    assert code == 0
    assert asked == [Path("tiny.json")]
    expected = FeatureConfig.improved(use_word_clusters=True, word_clusters=TINY_CLUSTERS)
    assert [(b["feature_config"], b["model_name"]) for b in built] == [
        (FeatureConfig.improved(), "crf_improved"),
        (expected, "crf_improved_plus_clusters"),
    ]
    assert {b["c1"] for b in built} == {0.1} and {b["c2"] for b in built} == {0.1}
    assert {b["max_iterations"] for b in built} == {100} and {b["label_mode"] for b in built} == {"A"}
    base = out / "mode_A"
    preds = _read_jsonl(base / CLUSTERS_VARIANT / "validation.jsonl")
    assert [p["sample_id"] for p in preds] == VAL_IDS
    assert {p["model_name"] for p in preds} == {"crf_improved_plus_clusters"}
    assert {p["label_mode"] for p in preds} == {"A"}
    assert all(len(p["predicted_labels"]) == len(_SEQUENCES[p["sample_id"]][0]) for p in preds)
    assert sorted(path.name for path in base.iterdir()) == ["improved", CLUSTERS_VARIANT, "sample.ids"]
    assert (base / "sample.ids").read_text(encoding="utf-8") == sample_text_for(TRAIN_IDS, 4)


def sample_text_for(ids: list[str], size: int) -> str:
    return "\n".join(sorted(random.Random(run_ablations.SEED).sample(sorted(ids), size))) + "\n"


def test_clusters_variant_shares_the_sample_with_the_other_variants(tmp_path, monkeypatch):
    root = build_root(tmp_path)
    fake_cluster_loader(monkeypatch, root)
    out = tmp_path / "out"
    argv = _argv(root, out, "--sample", "4", "--cluster-map", "m.json", "--variants", f"{CLUSTERS_VARIANT},improved")
    assert run_ablations.main(argv) == 0
    sample = (out / "mode_A" / "sample.ids").read_text(encoding="utf-8")
    assert sample == sample_text_for(TRAIN_IDS, 4)


def test_clusters_variant_without_a_cluster_map_is_an_argparse_error(tmp_path, capsys, monkeypatch):
    root = build_root(tmp_path)
    asked = fake_cluster_loader(monkeypatch, root)
    with pytest.raises(SystemExit) as exc:
        run_ablations.main(_argv(root, tmp_path / "out", "--variants", f"improved,{CLUSTERS_VARIANT}"))
    assert exc.value.code == 2
    assert f"--variants: {CLUSTERS_VARIANT} needs --cluster-map" in capsys.readouterr().err
    assert asked == [] and not (tmp_path / "out").exists()


def test_cluster_map_with_a_different_mode_is_one_line_error_before_any_training(tmp_path, capsys, monkeypatch):
    root = build_root(tmp_path)
    out = tmp_path / "out"
    fake_cluster_loader(monkeypatch, root, mode="B")

    def boom(**_: Any) -> None:
        raise AssertionError("trained although the cluster map is for another mode")

    monkeypatch.setattr(run_ablations, "CRFModel", boom)
    monkeypatch.setattr(run_ablations, "GapClassifier", boom)

    code = run_ablations.main(
        _argv(root, out, "--sample", "3", "--cluster-map", str(tmp_path / "secret" / "m.json"),
              "--variants", CLUSTERS_VARIANT)
    )

    assert code == 1
    assert capsys.readouterr().err.splitlines() == ["error: --cluster-map was built for mode B, not --mode A"]
    assert not out.exists()


def test_cluster_map_file_that_matches_release_and_ids_is_accepted(tmp_path):
    root = build_root(tmp_path)
    path = tmp_path / "map.json"
    cluster_map = ClusterMap(
        k=2, seed=1, min_count=1, dims=2, window=1, vocabulary_size=2, train_ids_sha256=ids_sha(root),
        release=RELEASE, mode="A", clusters={"hello": 0, "you": 1},
    )
    path.write_text(map_to_json(cluster_map), encoding="utf-8", newline="\n")

    assert run_ablations.load_clusters(path, "A", RELEASE, root) == {"hello": 0, "you": 1}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"release": "other"}, "error: --cluster-map was built for release 'other', not --release 'rel'"),
        ({"sha": "0" * 64}, "error: --cluster-map does not match the default_train IDs of --release 'rel' mode A"),
    ],
    ids=["wrong release", "wrong ids sha"],
)
def test_cluster_map_from_another_release_or_ids_is_one_line_error_before_any_training(
    tmp_path, capsys, monkeypatch, changes, message
):
    root = build_root(tmp_path)
    out = tmp_path / "out"
    fake_cluster_loader(monkeypatch, root, **changes)

    def boom(**_: Any) -> None:
        raise AssertionError("trained although the cluster map does not belong to this release")

    monkeypatch.setattr(run_ablations, "CRFModel", boom)
    monkeypatch.setattr(run_ablations, "GapClassifier", boom)

    code = run_ablations.main(
        _argv(root, out, "--sample", "3", "--cluster-map", str(tmp_path / "secret" / "m.json"),
              "--variants", CLUSTERS_VARIANT)
    )

    err = capsys.readouterr().err
    assert code == 1
    assert err.splitlines() == [message] and str(tmp_path) not in err
    assert not out.exists()


def test_cluster_map_load_failure_is_one_friendly_line(tmp_path, capsys, monkeypatch):
    root = build_root(tmp_path)

    def broken(path: Any) -> Any:
        raise FileNotFoundError(2, "No such file or directory", str(path))

    monkeypatch.setattr(run_ablations, "load_cluster_map", broken)
    missing = tmp_path / "deep" / "missing_map.json"

    code = run_ablations.main(
        _argv(root, tmp_path / "out", "--sample", "3", "--cluster-map", str(missing), "--variants", CLUSTERS_VARIANT)
    )

    err = capsys.readouterr().err
    assert code == 1
    assert err.splitlines() == ["error: missing_map.json: No such file or directory"]
    assert str(tmp_path) not in err


def test_cluster_map_given_adds_the_variant_to_the_defaults_in_variant_order(tmp_path):
    root = build_root(tmp_path)
    everything = list(run_ablations.VARIANTS)
    for mode in ("A", "B"):
        chosen = run_ablations.parse_args(_argv(root, tmp_path / "out", "--cluster-map", "m.json", mode=mode)).variants
        assert chosen == [v for v in everything if v not in C_ONLY] and len(chosen) == 11
        assert chosen.index(CLUSTERS_VARIANT) == chosen.index("improved_plus_question2") + 1
    chosen = run_ablations.parse_args(_argv(root, tmp_path / "out", "--cluster-map", "m.json", mode="C")).variants
    assert chosen == everything and len(chosen) == 13


def test_explicit_variants_without_the_clusters_one_ignore_the_map_in_the_list(tmp_path):
    root = build_root(tmp_path)
    args = run_ablations.parse_args(_argv(root, tmp_path / "out", "--cluster-map", "m.json", "--variants", "improved"))
    assert args.variants == ["improved"] and args.cluster_map == Path("m.json")


def test_cluster_map_defaults_to_none(tmp_path):
    root = build_root(tmp_path)
    assert run_ablations.parse_args(_argv(root, tmp_path / "out")).cluster_map is None


def test_the_docstring_documents_the_cluster_variant_and_option():
    assert "improved_plus_clusters" in run_ablations.__doc__ and "--cluster-map" in run_ablations.__doc__
