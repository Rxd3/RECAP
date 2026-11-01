"""Size and nesting limits for shipped code (src/ and scripts/): short functions, short files, shallow nesting.

Function length runs from the first decorator (or the `def`) to the last line: decorators, docstring and body all
count (spec section 5).
Nesting is the number of `if/for/while/with/try/match` blocks around a statement inside one function: a statement
directly in the function body is depth 0, `elif` and `else` stay at the depth of their `if`, and a nested function
starts again at 0 (it is checked on its own).
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from punct_restore.paths import REPO_ROOT as ROOT

MAX_FUNCTION_LINES = 50
MAX_FILE_LINES = 400
MAX_NESTING = 4

FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
BLOCKS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.TryStar, ast.Match)


def _shipped_files() -> list[Path]:
    return sorted(path for folder in ("src", "scripts") for path in (ROOT / folder).rglob("*.py"))


def _functions(tree: ast.AST) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    return (node for node in ast.walk(tree) if isinstance(node, FUNCTIONS))


def _child_blocks(node: ast.stmt) -> list[list[ast.stmt]]:
    """The statement lists one level inside a block; an `elif` is flattened into its `if`.

    An `elif` is a lone If in `orelse` at the same column as its `if`; an `if` written inside `else:` is indented
    further and counts as one more level.
    """
    if isinstance(node, ast.If):
        orelse = node.orelse
        if len(orelse) == 1 and isinstance(orelse[0], ast.If) and orelse[0].col_offset == node.col_offset:
            return [node.body, *_child_blocks(node.orelse[0])]
        return [node.body, node.orelse]
    if isinstance(node, (ast.Try, ast.TryStar)):
        return [node.body, *(handler.body for handler in node.handlers), node.orelse, node.finalbody]
    if isinstance(node, ast.Match):
        return [case.body for case in node.cases]
    return [node.body, getattr(node, "orelse", [])]


def _depth(statements: list[ast.stmt], level: int = 0) -> int:
    """Deepest nesting level reached by any statement in `statements` (nested functions excluded)."""
    deepest = level if statements else 0
    for node in statements:
        if isinstance(node, BLOCKS):
            deepest = max(deepest, *(_depth(block, level + 1) for block in _child_blocks(node)))
    return deepest


def _first_line(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])


def function_lengths(source: str) -> list[tuple[int, str, int]]:
    tree = ast.parse(source)
    return [(node.lineno, node.name, node.end_lineno - _first_line(node) + 1) for node in _functions(tree)]


def function_nesting(source: str) -> list[tuple[int, str, int]]:
    return [(node.lineno, node.name, _depth(node.body)) for node in _functions(ast.parse(source))]


def _offenders(measure: Callable[[str], list[tuple[int, str, int]]], limit: int) -> list[str]:
    hits = []
    for path in _shipped_files():
        relative = path.relative_to(ROOT).as_posix()
        hits += [f"{relative}:{line} {name} ({n})" for line, name, n in measure(path.read_text(encoding="utf-8"))
                 if n > limit]
    return hits


def test_functions_are_short() -> None:
    assert _offenders(function_lengths, MAX_FUNCTION_LINES) == []


def test_nesting_is_shallow() -> None:
    assert _offenders(function_nesting, MAX_NESTING) == []


def test_files_are_short() -> None:
    long_files = []
    for path in _shipped_files():
        count = len(path.read_text(encoding="utf-8").splitlines())
        if count > MAX_FILE_LINES:
            long_files.append(f"{path.relative_to(ROOT).as_posix()}:1 <file> ({count})")
    assert long_files == []


def _function_of(body_lines: int) -> str:
    return "def f():\n" + "    x = 1\n" * (body_lines - 1)


def _nested(levels: int, header: str = "if x:") -> str:
    lines = ["def f():"] + [f"{'    ' * (level + 1)}{header}" for level in range(levels)]
    return "\n".join([*lines, f"{'    ' * (levels + 1)}pass"]) + "\n"


def test_the_checker_flags_a_long_function() -> None:
    assert function_lengths(_function_of(MAX_FUNCTION_LINES)) == [(1, "f", 50)]
    assert function_lengths(_function_of(MAX_FUNCTION_LINES + 1)) == [(1, "f", 51)]
    assert function_lengths("@decorator\ndef f():\n    pass\n") == [(2, "f", 3)]
    assert function_lengths("@a\n@b(\n    1,\n)\ndef f():\n    pass\n") == [(5, "f", 6)]


def test_the_checker_flags_deep_nesting() -> None:
    assert function_nesting(_nested(MAX_NESTING)) == [(1, "f", 4)]
    assert function_nesting(_nested(MAX_NESTING + 1)) == [(1, "f", 5)]
    assert function_nesting(_nested(5, "for i in x:")) == [(1, "f", 5)]
    assert function_nesting(_nested(5, "with x:")) == [(1, "f", 5)]
    assert function_nesting("def f():\n    return 1\n") == [(1, "f", 0)]


@pytest.mark.parametrize(
    ("source", "depth"),
    [
        ("def f():\n    if a:\n        pass\n    elif b:\n        pass\n    elif c:\n        pass\n    else:\n"
         "        pass\n", 1),
        ("def f():\n    try:\n        pass\n    except E:\n        if a:\n            pass\n    finally:\n"
         "        pass\n", 2),
        ("def f():\n    match x:\n        case 1:\n            if a:\n                pass\n", 2),
        ("def f():\n    while a:\n        pass\n    else:\n        if b:\n            pass\n", 2),
        ("def f():\n    if a:\n        pass\n    else:\n        if b:\n            pass\n", 2),
        ("def f():\n    if a:\n        pass\n    else:\n        if b:\n            if c:\n                pass\n", 3),
    ],
)
def test_elif_else_except_and_case_are_counted_once(source: str, depth: int) -> None:
    assert function_nesting(source) == [(1, "f", depth)]


def test_a_nested_function_starts_again_at_zero() -> None:
    source = ("def outer():\n    if a:\n        def inner():\n            if b:\n                pass\n"
              "        return inner\n")
    assert function_nesting(source) == [(1, "outer", 1), (3, "inner", 1)]
