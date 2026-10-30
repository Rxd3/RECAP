"""Shipped code carries no scaffold leftovers (spec W1: remove scaffold-only placeholders)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "TO" + "DO"  # split so this file does not match itself


def test_no_todo_markers_in_src_scripts_or_tests() -> None:
    files = [path for folder in ("src", "scripts", "tests") for path in (ROOT / folder).rglob("*.py")]
    hits = [
        f"{path.relative_to(ROOT).as_posix()}:{number}"
        for path in files
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if MARKER in line
    ]
    assert hits == []
