"""Shared Markdown helpers for the reporting scripts.

Every number the scripts print comes from a JSON file written by scripts/evaluate_models.py.
"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

MODES = ("A", "B", "C")
PRIMARY = {"A": "punctuation/all", "B": "punctuation/all", "C": "joint/unmasked"}


def load(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:  # malformed JSON or bytes that are not UTF-8: say which file
        raise ValueError(f"{Path(path).name}: {exc}") from exc


def pct(x: float | str) -> str:
    # Round the stored 6-dp value half-up so 0.67985 prints 67.99, not the float artefact 67.98.
    return str((Decimal(str(x)) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


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
