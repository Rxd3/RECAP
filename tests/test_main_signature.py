"""Every command entry point in cli/ and reporting/ is exactly `main(argv: list[str] | None = None) -> int`."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from punct_restore.paths import REPO_ROOT as ROOT

PACKAGE = ROOT / "src" / "punct_restore"
MODULES = sorted(path for folder in ("cli", "reporting") for path in (PACKAGE / folder).glob("*.py"))
CANONICAL = "def main(argv: list[str] | None = None) -> int:\n    return 0\n"


def main_signatures(source: str) -> list[str]:
    """The parameters and return annotation (as source text) of every module-level `main` in `source`."""
    return [
        f"({ast.unparse(node.args)}) -> {ast.unparse(node.returns) if node.returns else '<missing>'}"
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]


EXPECTED = main_signatures(CANONICAL)


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def test_every_main_has_the_one_command_signature() -> None:
    found = {_relative(path): main_signatures(path.read_text(encoding="utf-8")) for path in MODULES}
    wrong = {name: signatures for name, signatures in found.items() if signatures and signatures != EXPECTED}
    assert wrong == {}


def test_the_check_sees_the_command_modules() -> None:
    """A moved or renamed folder must not turn the check above into an empty loop."""
    with_main = [_relative(path) for path in MODULES if main_signatures(path.read_text(encoding="utf-8"))]
    assert len(with_main) >= 25
    assert "src/punct_restore/cli/simulate_asr_noise.py" in with_main
    assert "src/punct_restore/reporting/time_models.py" in with_main


@pytest.mark.parametrize(
    ("source", "matches"),
    [
        (CANONICAL, True),
        (CANONICAL.replace("list[str]", "Sequence[str]"), False),
        (CANONICAL.replace("list[str] | None = None", "list[str] | None"), False),
        (CANONICAL.replace("argv: list[str] | None = None", "args: list[str] | None = None"), False),
        (CANONICAL.replace("argv: list[str] | None = None", "argv=None"), False),
        (CANONICAL.replace(" -> int", ""), False),
        (CANONICAL.replace("-> int", "-> None"), False),
        (CANONICAL.replace("None = None", "None = None, *, strict: bool = False"), False),
    ],
)
def test_the_check_reads_the_whole_signature(source: str, matches: bool) -> None:
    assert (main_signatures(source) == EXPECTED) is matches


@pytest.mark.parametrize(
    "source", ["def other() -> int:\n    return 0\n", "class C:\n    def main(self):\n        pass\n"]
)
def test_the_check_ignores_functions_that_are_not_module_level_main(source: str) -> None:
    assert main_signatures(source) == []
