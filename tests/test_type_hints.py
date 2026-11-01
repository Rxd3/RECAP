"""Every function in src/ annotates all of its parameters and its return type (`self` and `cls` excepted)."""

from __future__ import annotations

import ast

import pytest

from punct_restore.paths import REPO_ROOT as ROOT

FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    args = node.args
    every = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    every += [arg for arg in (args.vararg, args.kwarg) if arg is not None]
    return every


def missing_annotations(source: str) -> list[tuple[int, str, str]]:
    """`(line, function, what)` for every unannotated parameter or return in `source`."""
    missing = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, FUNCTIONS):
            continue
        params = _parameters(node)
        if params and params[0].arg in ("self", "cls") and params[0] in [*node.args.posonlyargs, *node.args.args]:
            params = params[1:]
        missing += [(node.lineno, node.name, f"parameter {arg.arg}") for arg in params if arg.annotation is None]
        if node.returns is None:
            missing.append((node.lineno, node.name, "return"))
    return missing


def test_every_function_in_src_is_fully_annotated() -> None:
    hits = [f"{path.relative_to(ROOT).as_posix()}:{line} {name}: {what}"
            for path in sorted((ROOT / "src").rglob("*.py"))
            for line, name, what in missing_annotations(path.read_text(encoding="utf-8"))]
    assert hits == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def f(a: int) -> int:\n    return a\n", []),
        ("def f(a) -> int:\n    return 1\n", [(1, "f", "parameter a")]),
        ("def f(a: int):\n    return 1\n", [(1, "f", "return")]),
        ("def f(*args, **kwargs) -> None:\n    pass\n", [(1, "f", "parameter args"), (1, "f", "parameter kwargs")]),
        ("def f(a: int, /, b: int, *, c) -> None:\n    pass\n", [(1, "f", "parameter c")]),
        ("class C:\n    def m(self, a: int) -> None:\n        pass\n", []),
        ("class C:\n    @classmethod\n    def m(cls) -> None:\n        pass\n", []),
        ("def outer() -> None:\n    def inner(x):\n        pass\n",
         [(2, "inner", "parameter x"), (2, "inner", "return")]),
        ("async def f(a: int):\n    pass\n", [(1, "f", "return")]),
    ],
)
def test_the_annotation_check_sees_every_parameter_kind(source: str, expected: list[tuple[int, str, str]]) -> None:
    assert missing_annotations(source) == expected
