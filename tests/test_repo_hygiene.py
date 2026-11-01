"""Shipped code carries no scaffold leftovers (spec W1) and no import-path edits or bare sibling imports."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MARKER = "TO" + "DO"  # split so this file does not match itself
SYS_PATH = "sys." + "path"  # split so this file does not match itself
# The old bare module names (the former scripts and scripts/reporting files), listed one by one: `asr_noise_\w+`
# would also match the test helper module `asr_noise_table_fixtures`, which the tests import by its bare name.
_COMMAND_SCRIPTS = (
    "build_leakage_subsets", "build_word_clusters", "evaluate_models", "factor_predictions", "find_gold_noise",
    "package_hmm_handoff", "predict_hmm", "prepare_data", "project_joint", "run_ablations", "run_demo",
    "run_hmm_experiments", "simulate_asr_noise", "train_crf", "train_hidden_event_lm", "train_hmm", "tune_threshold",
    "verify_hmm_experiments",
)
_SIBLING = rf"(?:_cli|build_tables|asr_noise_(?:findings|report|stats|table)|{'|'.join(_COMMAND_SCRIPTS)})\b"
BARE_SIBLING = re.compile(rf"^\s*(?:from\s+{_SIBLING}\s+import\b|import\s+{_SIBLING})", re.MULTILINE)
# A command module is imported from the package, never from the scripts folder (its stub runs the command).
FROM_SCRIPTS = re.compile(r"^\s*(?:from\s+scripts(?:\.\w+)*\s+import\b|import\s+scripts\b)", re.MULTILINE)


def _python_files() -> list[Path]:
    return [path for folder in ("src", "scripts", "tests") for path in (ROOT / folder).rglob("*.py")]


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
        for path in _python_files()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if SYS_PATH in line
    ]
    assert hits == []


def _hits(path: Path, pattern: re.Pattern[str]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [f"{_relative(path)}:{text.count(chr(10), 0, match.end()) + 1}" for match in pattern.finditer(text)]


def test_no_bare_sibling_imports() -> None:
    assert [hit for path in _python_files() for hit in _hits(path, BARE_SIBLING)] == []


@pytest.mark.parametrize(
    "line",
    ["from _cli import run", "from build_tables import pct", "from asr_noise_stats import (\n",
     "    import asr_noise_findings", "import _cli", "from run_hmm_experiments import frozen, now",
     "import run_hmm_experiments as workflow", "    import train_crf"],
)
def test_the_bare_sibling_pattern_catches_the_old_forms(line: str) -> None:
    assert BARE_SIBLING.search(line)


@pytest.mark.parametrize(
    "line",
    ["from punct_restore.reporting import _cli", "from punct_restore.reporting.build_tables import pct",
     "from punct_restore.data.asr_noise import V2_WEIGHTS",
     "    from punct_restore.reporting import asr_noise_findings", "import asr_noise",
     "from asr_noise_table_fixtures import report", "from punct_restore.cli.run_hmm_experiments import frozen",
     "from punct_restore.cli import train_crf", "from test_hmm_cli import ROOT", "import train_crf_helpers"],
)
def test_the_bare_sibling_pattern_leaves_package_imports_alone(line: str) -> None:
    assert BARE_SIBLING.search(line) is None


def test_nothing_imports_from_the_scripts_folder() -> None:
    assert [hit for path in _python_files() for hit in _hits(path, FROM_SCRIPTS)] == []


@pytest.mark.parametrize(
    "line", ["from scripts import run_demo", "import scripts.train_crf as train_crf", "from scripts.reporting import _cli",
             "    from scripts import tune_threshold"],
)
def test_the_scripts_import_pattern_catches_the_old_forms(line: str) -> None:
    assert FROM_SCRIPTS.search(line)


@pytest.mark.parametrize("line", ["from punct_restore.cli import run_demo", "import scripts_helper", "SCRIPTS = ROOT"])
def test_the_scripts_import_pattern_leaves_other_imports_alone(line: str) -> None:
    assert FROM_SCRIPTS.search(line) is None
