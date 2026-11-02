"""Lazy package exports (PEP 562): a package whose names load their submodule only when first asked for.

``punct_restore.data`` and ``punct_restore.evaluation`` export names from many submodules. Importing the package must
not import them all, or every command that needs one small module (say ``data.labeling``) would load the evaluator
and the release builder too, and the HMM provenance (`hmm_experiment_identity.SOURCE_FILES`) would hash them.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping
from typing import Any


def lazy_exports(package: str, exports: Mapping[str, str]) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """The module-level ``__getattr__`` and ``__dir__`` of ``package``, for ``exports`` (name -> its submodule).

    An exported name is taken from its submodule on first use and kept on the package, so the next access is a plain
    attribute. A submodule (``package.labeling``) is imported when asked for, as it was an attribute while the package
    imported everything at once. Any other name is an AttributeError.
    """
    def missing(name: str) -> AttributeError:
        return AttributeError(f"module {package!r} has no attribute {name!r}")

    def module_getattr(name: str) -> Any:
        if name in exports:
            value = getattr(importlib.import_module(f".{exports[name]}", package), name)
        elif name.startswith("_"):
            raise missing(name)
        else:
            try:
                value = importlib.import_module(f".{name}", package)
            except ModuleNotFoundError as exc:
                if exc.name != f"{package}.{name}":
                    raise  # the submodule exists but needs a module that is not installed: say so
                raise missing(name) from None
        setattr(sys.modules[package], name, value)
        return value

    def module_dir() -> list[str]:
        return sorted({*vars(sys.modules[package]), *exports})

    return module_getattr, module_dir
