"""Shared plumbing for the reporting command lines: safe output files and one friendly stderr line.

Output files are only ever created (never replaced), the one-line ``error: ...`` goes to stderr with exit
code 1, and no message shows a folder name or lets a console that cannot print a character crash the script.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from punct_restore.prediction_io import (
    check_new_output_paths,
    one_line_error,
    reconfigure_stdout,
    write_new_file,
)


def check_out(out: Path, inputs: Mapping[str, Path | None] | None = None) -> None:
    """Refuse an ``--out`` that is one of ``inputs`` (flag name -> file), a folder, in a folder that is missing or
    cannot be written, or a file that already exists. Call it before reading anything; it raises a one-line
    ValueError."""
    check_new_output_paths({"--out": out}, inputs or {})
    if not out.parent.exists():
        raise ValueError(f"{out.name}: folder {out.parent.name or out.parent} not found")
    if out.exists():
        raise ValueError(f"{out.name} already exists")


def write_new_text(out: Path, text: str) -> None:
    """Create ``out`` holding ``text`` as UTF-8 with the platform newline; an existing file is never replaced."""
    write_new_file(out, text, newline=None)


def name_only(text: str, path: Path) -> str:
    """``text`` with every spelling of ``path`` (as given, resolved, repr-escaped, slashes) cut to its file name."""
    for spelling in (str(path.resolve()), str(path)):  # the resolved form is longer: replace it first
        for form in (spelling, spelling.replace("\\", "\\\\"), Path(spelling).as_posix()):
            text = text.replace(form, path.name)
    return text


def _fail(message: str) -> int:
    print("error: " + " ".join(message.splitlines()), file=sys.stderr)
    return 1


def run(task: Callable[[], None], *, json_errors: bool = True) -> int:
    """Run ``task``; a missing file or malformed input becomes ``error: <message>`` and exit code 1.

    Files are named by their file name only, never by their folder. With ``json_errors`` the task reads
    JSON inputs, so ``KeyError``/``TypeError``/``ArithmeticError`` are reported as bad input; pass
    ``json_errors=False`` when the task also runs model code, where those errors are bugs and propagate.
    """
    reconfigure_stdout()  # a non-ASCII ``--out`` on a cp1252 console must not crash print()
    try:
        task()
    except OSError as exc:
        # a message-only OSError may quote a whole path: only a system reason is safe to show
        return _fail(one_line_error(exc) if exc.strerror else "a file could not be read or written")
    except ValueError as exc:
        return _fail(one_line_error(exc))
    except KeyError as exc:
        if not json_errors:
            raise
        return _fail(f"missing key {exc} in an input JSON file")
    except (TypeError, ArithmeticError):
        if not json_errors:
            raise
        return _fail("an input JSON file holds a value of the wrong type")
    return 0
