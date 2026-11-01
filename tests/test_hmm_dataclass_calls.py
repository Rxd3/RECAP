"""The small record classes of the HMM code are built with named arguments, so a call reads without the class open."""

from __future__ import annotations

import ast
from pathlib import Path

from punct_restore.paths import REPO_ROOT as ROOT

HMM_FILES = sorted(
    [*(ROOT / "src/punct_restore/models").glob("hmm*.py"), *(ROOT / "src/punct_restore/cli").glob("*hmm*.py")]
)


def _record_classes(tree: ast.AST) -> set[str]:
    """Names of the classes in `tree` that are dataclasses or NamedTuples."""
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        decorators = {ast.unparse(d.func if isinstance(d, ast.Call) else d) for d in node.decorator_list}
        bases = {ast.unparse(base) for base in node.bases}
        if "dataclass" in decorators or "NamedTuple" in bases:
            names.add(node.name)
    return names


def _positional_calls(path: Path, names: set[str]) -> list[str]:
    """`file:line Class(...)` for every call in `path` that passes an argument to one of `names` by position."""
    return [
        f"{path.name}:{node.lineno} {node.func.id}(...)"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names and node.args
    ]


def test_every_record_class_of_the_hmm_code_is_built_with_named_arguments() -> None:
    names = set().union(*(_record_classes(ast.parse(path.read_text(encoding="utf-8"))) for path in HMM_FILES))
    assert {"Study", "_RunFacts", "_TrainingRules", "_PairSpec", "_Prediction"} <= names  # the scan sees them
    offenders = [hit for path in HMM_FILES for hit in _positional_calls(path, names)]
    assert offenders == []
