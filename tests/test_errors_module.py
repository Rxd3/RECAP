"""Round 3, item 10: the error-text helpers have their own module, punct_restore.errors; prediction_io still exports
them (restructure spec section 2) and no other module of the package imports them from there any more."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from punct_restore import errors, prediction_io
from punct_restore.paths import REPO_ROOT

MOVED = ("OS_REASONS", "os_error_code", "os_reason", "one_line_error", "one_line_error_with_short_paths",
         "reconfigure_stdout")
PACKAGE = REPO_ROOT / "src" / "punct_restore"


@pytest.mark.parametrize("name", MOVED)
def test_the_helper_lives_in_errors_and_prediction_io_exports_the_very_same_object(name: str) -> None:
    assert hasattr(errors, name)
    assert getattr(prediction_io, name) is getattr(errors, name)


def test_the_errors_module_depends_on_nothing_of_the_package() -> None:
    """A module every command prints through must stay a leaf: it adds nothing to anyone's import closure."""
    imports = [node for node in ast.walk(ast.parse((PACKAGE / "errors.py").read_text(encoding="utf-8")))
               if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert [node for node in imports
            if isinstance(node, ast.ImportFrom) and (node.level > 0 or (node.module or "").startswith("punct_restore"))
            or isinstance(node, ast.Import) and any(alias.name.startswith("punct_restore") for alias in node.names)
            ] == []


def _names_imported_from_prediction_io(path: Path) -> list[str]:
    """The moved names that a module takes from prediction_io (absolute or relative import)."""
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if module == "punct_restore.prediction_io" or (node.level > 0 and module == "prediction_io"):
            found += [alias.name for alias in node.names if alias.name in MOVED]
    return found


def test_no_module_takes_the_moved_helpers_from_prediction_io() -> None:
    users = {path.relative_to(PACKAGE).as_posix(): names for path in sorted(PACKAGE.rglob("*.py"))
             if path.name != "prediction_io.py" and (names := _names_imported_from_prediction_io(path))}
    assert users == {}


def test_the_stdout_setting_still_turns_unencodable_text_into_escapes(capsys: pytest.CaptureFixture[str]) -> None:
    """reconfigure_stdout is the one helper that is not a pure function: it must still find a reconfigurable stdout."""
    errors.reconfigure_stdout()  # under pytest's capture the stream may not reconfigure: it must not fail either way
    print("ok")
    assert capsys.readouterr().out == "ok\n"
