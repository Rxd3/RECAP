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

from punct_restore.features.crf_features import FeatureConfig
from scripts import run_ablations

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


def test_variants_are_the_twelve_expected_names_in_order():
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
        "gap_improved",
        "case_improved",
        "factored",
    ]
    assert run_ablations.VARIANTS["improved"] == {}
    assert run_ablations.VARIANTS["improved_minus_window2"] == {"window_size": 1}
    assert run_ablations.VARIANTS["improved_plus_filler"] == {"use_filler_cues": True}
    assert run_ablations.VARIANTS["improved_plus_question2"] == {"use_question_lookback": True}
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
    assert not re.search(r"[A-Za-z]:[/\\]", source)


# --- mode C only variants: case_improved (CASE CRF) and factored (mode A CRF + CASE -> joint labels) -------------

C_ONLY = ["case_improved", "factored"]
A_LABELS = {"O", "COMMA", "PERIOD", "QUESTION"}
CASING = {"LOWER", "FIRST_UPPER", "ALL_UPPER"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_default_variants_leave_out_the_mode_c_only_ones_for_a_and_b_and_keep_them_in_order_for_c(tmp_path):
    root = build_root(tmp_path)
    everything = list(run_ablations.VARIANTS)
    for mode in ("A", "B"):
        chosen = run_ablations.parse_args(_argv(root, tmp_path / "out", mode=mode)).variants
        assert chosen == [v for v in everything if v not in C_ONLY] and len(chosen) == 10
    chosen = run_ablations.parse_args(_argv(root, tmp_path / "out", mode="C")).variants
    assert chosen == everything
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
        rows["factored/validation.jsonl"], rows["factored/punct_A.jsonl"], rows["case_improved/validation.jsonl"]
    ):
        assert joint["predicted_labels"] == [
            f"{p}|{c}" for p, c in zip(punct["predicted_labels"], case["predicted_labels"])
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

    def run_then_empty_the_case_file(mode: str, variant: str, *args: Any) -> None:
        real(mode, variant, *args)
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
