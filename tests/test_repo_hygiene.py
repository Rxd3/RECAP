"""Shipped code carries no scaffold leftovers (spec W1) and no import-path edits or bare sibling imports."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MARKER = "TO" + "DO"  # split so this file does not match itself
SYS_PATH = "sys." + "path"  # split so this file does not match itself
# The old bare module names, listed one by one: `asr_noise_\w+` would also match the test helper module
# `asr_noise_table_fixtures`, which the tests folder legitimately imports by its bare name.
_SIBLING = r"(?:_cli|build_tables|asr_noise_(?:findings|report|stats|table))\b"
BARE_SIBLING = re.compile(rf"^\s*(?:from\s+{_SIBLING}\s+import\b|import\s+{_SIBLING})", re.MULTILINE)
# Until Task 4 these two checks cover src/, scripts/reporting/ and the reporting tests only; Task 4 widens them to
# src/, scripts/, tests/.
REPORTING_TESTS = (
    "asr_noise_table_fixtures.py", "reporting_fixtures.py", "test_asr_noise_findings.py", "test_asr_noise_table.py",
    "test_crf_convergence.py", "test_length_position.py", "test_reporting.py", "test_reporting_safety.py",
    "test_stubs.py", "test_repo_hygiene.py", "test_cli_dispatcher.py",
)


def _python_files() -> list[Path]:
    return [path for folder in ("src", "scripts", "tests") for path in (ROOT / folder).rglob("*.py")]


def _import_scope() -> list[Path]:
    return [
        *(ROOT / "src").rglob("*.py"),
        *(ROOT / "scripts" / "reporting").glob("*.py"),
        *(ROOT / "tests" / name for name in REPORTING_TESTS),
    ]


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def test_no_python_file_starts_with_a_byte_order_mark() -> None:
    """A UTF-8 BOM imports fine but breaks ast.parse of the raw text and byte-level tools."""
    with_bom = [_relative(path) for path in _python_files() if path.read_bytes().startswith(b"\xef\xbb\xbf")]
    assert with_bom == []


def test_no_todo_markers_in_src_scripts_or_tests() -> None:
    files = _python_files()
    hits = [
        f"{_relative(path)}:{number}"
        for path in files
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if MARKER in line
    ]
    assert hits == []


def test_no_file_edits_sys_path() -> None:
    hits = [
        f"{_relative(path)}:{number}"
        for path in _import_scope()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if SYS_PATH in line
    ]
    assert hits == []


def _bare_sibling_hits(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [f"{_relative(path)}:{text.count(chr(10), 0, match.end()) + 1}" for match in BARE_SIBLING.finditer(text)]


def test_no_bare_sibling_imports() -> None:
    assert [hit for path in _import_scope() for hit in _bare_sibling_hits(path)] == []


@pytest.mark.parametrize(
    "line",
    ["from _cli import run", "from build_tables import pct", "from asr_noise_stats import (\n",
     "    import asr_noise_findings", "import _cli"],
)
def test_the_bare_sibling_pattern_catches_the_old_forms(line: str) -> None:
    assert BARE_SIBLING.search(line)


@pytest.mark.parametrize(
    "line",
    ["from punct_restore.reporting import _cli", "from punct_restore.reporting.build_tables import pct",
     "from punct_restore.data.asr_noise import V2_WEIGHTS",
     "    from punct_restore.reporting import asr_noise_findings", "import asr_noise",
     "from asr_noise_table_fixtures import report"],
)
def test_the_bare_sibling_pattern_leaves_package_imports_alone(line: str) -> None:
    assert BARE_SIBLING.search(line) is None
