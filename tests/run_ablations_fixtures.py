"""Shared fixtures of the run_ablations tests: a tiny data root for release 'rel', held-out test files included.

Used by test_run_ablations.py, test_run_ablations_errors.py and test_malformed_inputs.py.
"""

from __future__ import annotations

import builtins
import fnmatch
import io
import json
import os
from pathlib import Path
from typing import Any

import pytest

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
