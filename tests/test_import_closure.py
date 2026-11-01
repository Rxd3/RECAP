"""The import-closure helper the provenance test relies on: it must follow every kind of import inside a package."""

from __future__ import annotations

from pathlib import Path

from import_closure import import_closure


def _tree(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / "pkg" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _names(root: Path, found: set[Path]) -> set[str]:
    return {path.relative_to(root / "pkg").as_posix() for path in found}


def test_it_follows_absolute_imports_through_several_levels(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"a.py": "from pkg.b import x\n", "b.py": "import pkg.c\n", "c.py": "", "unused.py": ""})
    assert _names(root, import_closure([root / "pkg/a.py"], root, "pkg")) == {"a.py", "b.py", "c.py"}


def test_it_follows_relative_imports_from_a_module_and_from_a_package_init(tmp_path: Path) -> None:
    root = _tree(tmp_path, {
        "sub/__init__.py": "from .one import x\nfrom .. import top\n",
        "sub/one.py": "from ..top import y\nfrom . import two\n",
        "sub/two.py": "",
        "top.py": "",
        "unused.py": "",
    })
    found = import_closure([root / "pkg/sub/__init__.py"], root, "pkg")
    assert _names(root, found) == {"sub/__init__.py", "sub/one.py", "sub/two.py", "top.py"}


def test_it_follows_a_submodule_imported_by_name_from_its_package(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"a.py": "from pkg.models import one\n", "models/__init__.py": "", "models/one.py": ""})
    found = import_closure([root / "pkg/a.py"], root, "pkg")
    assert _names(root, found) == {"a.py", "models/__init__.py", "models/one.py"}


def test_it_follows_an_import_written_inside_a_function(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"a.py": "def f():\n    from pkg.b import x\n", "b.py": ""})
    assert _names(root, import_closure([root / "pkg/a.py"], root, "pkg")) == {"a.py", "b.py"}


def test_it_ignores_other_packages_and_survives_an_import_cycle(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"a.py": "import json\nimport yaml\nfrom pkg.b import x\n", "b.py": "from pkg.a import y\n"})
    assert _names(root, import_closure([root / "pkg/a.py"], root, "pkg")) == {"a.py", "b.py"}


def test_a_parent_package_init_is_not_followed_for_a_dotted_module_import(tmp_path: Path) -> None:
    """`from pkg.models.one import x` names the module `one`; the `__init__.py` it runs first is not part of it."""
    root = _tree(tmp_path, {"a.py": "from pkg.models.one import x\n", "models/__init__.py": "", "models/one.py": ""})
    assert _names(root, import_closure([root / "pkg/a.py"], root, "pkg")) == {"a.py", "models/one.py"}
