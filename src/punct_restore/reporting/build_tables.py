"""Shared Markdown helpers for the reporting commands.

The table numbers the commands print come from JSON files written by `recap evaluate-models`; a few fixed
descriptions in the section prose are written into build_m3_final_sections.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from punct_restore.evaluation.report import pct as pct  # the one percent format; the tables import it from here
from punct_restore.file_io import read_json, read_object, require_object

MODES = ("A", "B", "C")
PRIMARY = {"A": "punctuation/all", "B": "punctuation/all", "C": "joint/unmasked"}


# What the table builders read of one block of an evaluator file (see file_io.require_object for the kinds).
CELL: dict[str, Any] = {"macro_f1": float}  # cell(); a macro_f1_ci, when there is one, is checked by read_scores
ROW: dict[str, Any] = {**CELL, "micro": {"precision": float, "recall": float, "f1": float}, "accuracy": float}
DIFF: dict[str, Any] = {"difference": float, "difference_ci": {"low": float, "high": float}}  # diff_cell()
CONFUSION: dict[str, Any] = {"labels": list[str], "confusion": list[list[int]]}  # matrix()
PER_CLASS: dict[str, Any] = {"precision": float, "recall": float, "f1": float, "support": int}


def load(path: str | Path, schema: Mapping[str, Any] | None = None) -> Any:
    """The JSON file at ``path``; with a ``schema`` it must be an object that holds those keys.

    Raises:
        OSError: If the file cannot be read.
        ValueError: One line that starts with the file name: not valid UTF-8 or JSON, and with a ``schema`` also not
            a JSON object, a key that is missing (``missing <key>``) or one of the wrong kind
            (``<key> is not <kind>``); a nested key is named by its dotted path.
    """
    path = Path(path)
    return read_json(path) if schema is None else read_object(path, schema)


def read_scores(path: Path, blocks: Mapping[str, Mapping[str, Any]], extra: Mapping[str, Any] | None = None
                ) -> dict[str, Any]:
    """The evaluator score or compare file at ``path``, checked for the keys a table reads.

    ``blocks`` maps each block name to the schema of the keys read from it (``CELL``, ``ROW``, ``DIFF``, ...); ``extra``
    adds top-level keys. A block with a ``macro_f1`` may hold a ``macro_f1_ci`` (``null`` or none is "n/a"); if it
    does, that holds a ``low`` and a ``high``. Raises ValueError as ``load`` does.
    """
    doc = read_object(path, {**(extra or {}), "blocks": blocks})
    for name, schema in blocks.items():
        interval = doc["blocks"][name].get("macro_f1_ci") if "macro_f1" in schema else None
        if interval is not None:
            require_object(path.name, interval, {"low": float, "high": float}, f"blocks.{name}.macro_f1_ci")
    return doc


def read_block(path: Path, name: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Block ``name`` of the evaluator file at ``path``, checked against ``schema`` (see ``read_scores``)."""
    block: dict[str, Any] = read_scores(path, {name: schema})["blocks"][name]
    return block


def ci(block: dict[str, Any], key: str = "macro_f1_ci") -> str:
    c = block.get(key)
    return f"{pct(c['low'])}–{pct(c['high'])}" if c else "n/a"


def cell(block: dict[str, Any]) -> str:
    return f"{pct(block['macro_f1'])} ({ci(block)})"


def diff_cell(block: dict[str, Any]) -> str:
    c = block["difference_ci"]
    sign = "+" if block["difference"] >= 0 else ""
    excl = "" if c["low"] <= 0 <= c["high"] else " *"
    return f"{sign}{pct(block['difference'])} ({pct(c['low'])} to {pct(c['high'])}){excl}"


def half_up(x: float | str | Decimal, places: int = 0, *, grouping: bool = False) -> str:
    """``x`` rounded half-up to ``places`` decimals, like ``pct``: the shortest decimal of a float, not its binary
    value.

    ``grouping=True`` adds thousands separators (``f"{x:,.{places}f}"`` without the binary-float tie errors).
    """
    value = Decimal(repr(x)) if isinstance(x, float) else Decimal(x)
    return format(value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP), "," if grouping else "")
