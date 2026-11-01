"""Tiny hand-written fixtures for the Member 5 evaluation tests (no real data, no network).

Every builder returns fresh objects, so a test may change its own copy without touching other tests.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_models.py"  # the stub: subprocess runs use the entry point
MODULE_PATH = REPO_ROOT / "src" / "punct_restore" / "cli" / "evaluate_models.py"  # the code: in-process loads
BOM = b"\xef\xbb\xbf"
BAD_UTF8_LINE = b"bad \xe9\xff\n"  # 0xE9 starts a 3-byte UTF-8 sequence that 0xFF breaks
FILE_NAMES = {"gold": "gold.jsonl", "ids": "list.ids", "predictions": "preds.jsonl", "masks": "masks.jsonl"}


def corrupt_line_two(path: Path) -> None:
    """Keep the first line of the file and make line 2 invalid UTF-8."""
    path.write_bytes(path.read_bytes().splitlines(keepends=True)[0] + BAD_UTF8_LINE)


def run_cli(cli: ModuleType, capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str, str]:
    """cli.main(argv) in-process: (exit code, stdout, stderr)."""
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def assert_one_error_line(err: str) -> None:
    """Spec 8: one friendly stderr line, no traceback."""
    assert err.startswith("error: "), repr(err)
    assert err.endswith("\n") and err.count("\n") == 1, repr(err)
    assert "\r" not in err and "Traceback" not in err, repr(err)


def cli_argv(command: str, case: Case, out: Path, *extra: str, predictions: bool = True) -> list[str]:
    """[command, --gold, --ids, (--predictions), (--masks), *extra, --out] for a written case."""
    argv = [command, "--gold", str(case.gold), "--ids", str(case.ids)]
    if predictions:
        argv += ["--predictions", str(case.predictions)]
    if case.masks is not None:
        argv += ["--masks", str(case.masks)]
    return argv + [*extra, "--out", str(out)]


def gold_row(sample_id: str, mode: str, tokens: Sequence[str], punct: Sequence[str], case: Sequence[str],
             group_id: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"sample_id": sample_id, "label_mode": mode, "tokens": list(tokens),
                           "punct_labels": list(punct), "case_labels": list(case)}
    if group_id is not None:
        row["metadata"] = {"group_id": group_id}
    return row


def pred_row(sample_id: str, mode: str, labels: Sequence[str], model: str = "m") -> dict[str, Any]:
    return {"sample_id": sample_id, "model_name": model, "label_mode": mode, "predicted_labels": list(labels)}


def mask_row(sample_id: str, positions: Sequence[Any], reason: str | None = None) -> dict[str, Any]:
    return {"sample_id": sample_id, "excluded_positions": list(positions), "reason": reason}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any] | str]) -> Path:
    """One JSON object per line; a str row is written verbatim (blank lines, bad JSON, wrong shapes)."""
    lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def write_ids(path: Path, ids: Iterable[str]) -> Path:
    path.write_text("".join(f"{sample_id}\n" for sample_id in ids), encoding="utf-8")
    return path


# Mode A/B/C share tokens: s1 is a wh-question, s2 a plain statement, s3 a one-token record outside the ID list.
TOKENS = {"s1": ("what", "is", "it"), "s2": ("well", "it", "rains"), "s3": ("ok",)}
CASE = {"s1": ("FIRST_UPPER", "LOWER", "LOWER"), "s2": ("FIRST_UPPER", "LOWER", "LOWER"), "s3": ("ALL_UPPER",)}
_STANDARD = {"s1": ("O", "O", "QUESTION"), "s2": ("COMMA", "O", "PERIOD"), "s3": ("PERIOD",)}
PUNCT = {"A": _STANDARD, "B": {"s1": ("O", "O", "QUESTION"), "s2": ("COLON", "O", "EXCLAMATION"), "s3": ("PERIOD",)},
         "C": _STANDARD}
GROUPS = {"s1": "g1", "s2": "g1", "s3": None}
SCORED = ("s1", "s2")


def gold_rows(mode: str) -> list[dict[str, Any]]:
    """Valid gold for s1, s2, s3 in the given mode (s1 and s2 carry group_id g1, s3 has no metadata)."""
    return [gold_row(sid, mode, TOKENS[sid], PUNCT[mode][sid], CASE[sid], GROUPS[sid]) for sid in TOKENS]


def joint(punct: Sequence[str], case: Sequence[str]) -> list[str]:
    return [f"{p}|{c}" for p, c in zip(punct, case, strict=False)]


def perfect_preds(mode: str, ids: Sequence[str] = SCORED, model: str = "m") -> list[dict[str, Any]]:
    """Predictions equal to gold for the given IDs (joint labels for mode C)."""
    labels = {sid: joint(PUNCT[mode][sid], CASE[sid]) if mode == "C" else PUNCT[mode][sid] for sid in TOKENS}
    return [pred_row(sid, mode, labels[sid], model) for sid in ids]


def all_masks(positions: dict[str, list[int]] | None = None) -> list[dict[str, Any]]:
    """One mask entry per gold record (s1, s2, s3); unspecified IDs get no excluded positions."""
    chosen = positions or {}
    return [mask_row(sid, chosen.get(sid, [])) for sid in TOKENS]


@dataclass(frozen=True)
class Case:
    gold: Path
    ids: Path
    predictions: Path
    masks: Path | None


def write_case(folder: Path, gold: Iterable[dict[str, Any] | str], ids: Iterable[str],
               predictions: Iterable[dict[str, Any] | str],
               masks: Iterable[dict[str, Any] | str] | None = None) -> Case:
    """gold.jsonl, list.ids, preds.jsonl and (when given) masks.jsonl inside `folder`."""
    return Case(gold=write_jsonl(folder / "gold.jsonl", gold), ids=write_ids(folder / "list.ids", ids),
                predictions=write_jsonl(folder / "preds.jsonl", predictions),
                masks=write_jsonl(folder / "masks.jsonl", masks) if masks is not None else None)


def valid_case(folder: Path, mode: str) -> Case:
    """A known-good case: gold s1-s3, IDs s1 and s2, perfect predictions for s1 and s2, masks for mode C."""
    return write_case(folder, gold_rows(mode), SCORED, perfect_preds(mode),
                      all_masks() if mode == "C" else None)
