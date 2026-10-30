"""Shipped code carries no scaffold leftovers (spec W1: remove scaffold-only placeholders)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "TO" + "DO"  # split so this file does not match itself


def _python_files() -> list[Path]:
    return [path for folder in ("src", "scripts", "tests") for path in (ROOT / folder).rglob("*.py")]


def test_no_python_file_starts_with_a_byte_order_mark() -> None:
    """A UTF-8 BOM imports fine but breaks ast.parse of the raw text and byte-level tools."""
    with_bom = [path.relative_to(ROOT).as_posix() for path in _python_files()
                if path.read_bytes().startswith(b"\xef\xbb\xbf")]
    assert with_bom == []


def test_no_todo_markers_in_src_scripts_or_tests() -> None:
    files = _python_files()
    hits = [
        f"{path.relative_to(ROOT).as_posix()}:{number}"
        for path in files
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if MARKER in line
    ]
    assert hits == []
