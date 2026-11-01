"""Which files of one package a set of modules imports, directly or through other modules of that package.

Only import statements are read (`import a.b`, `from a.b import c`, relative forms, and imports inside functions).
A module named in a statement is a file of the closure; so is `a.b.c` when `c` is itself a module. The `__init__.py`
of the parent packages that Python runs first is not followed unless a statement names that package.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path


def _module_file(dotted: str, src: Path) -> Path | None:
    """The file of the module `dotted` under `src`, or None when it is not there (a name, or another package)."""
    base = src.joinpath(*dotted.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _imported_names(path: Path, src: Path, package: str) -> set[str]:
    """Every dotted name of `package` that the import statements in `path` may refer to (modules or attributes)."""
    folder = path.parent.relative_to(src).parts
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            anchor = list(folder[: len(folder) - node.level + 1]) if node.level else []
            module = ".".join([*anchor, *([node.module] if node.module else [])])
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
    return {name for name in names if name.split(".")[0] == package}


def import_closure(roots: Iterable[Path], src: Path, package: str) -> set[Path]:
    """The root files and every file of `package` (under `src`) they import, directly or indirectly."""
    found: set[Path] = set()
    pending = list(roots)
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        for name in _imported_names(path, src, package):
            target = _module_file(name, src)
            if target is not None:
                pending.append(target)
    return found
