"""Shared plumbing for the reporting command lines: safe output files and one friendly stderr line.

Output files are only ever created (never replaced), the one-line ``error: ...`` goes to stderr with exit
code 1, and no message shows a folder name or lets a console that cannot print a character crash the script.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from punct_restore.prediction_io import check_new_output_paths


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
    with open(out, "x", encoding="utf-8") as handle:
        handle.write(text)


def name_only(text: str, path: Path) -> str:
    """``text`` with every spelling of ``path`` (as given, resolved, repr-escaped, slashes) cut to its file name."""
    for spelling in (str(path.resolve()), str(path)):  # the resolved form is longer: replace it first
        for form in (spelling, spelling.replace("\\", "\\\\"), Path(spelling).as_posix()):
            text = text.replace(form, path.name)
    return text


def _fail(message: str) -> int:
    print("error: " + " ".join(message.splitlines()), file=sys.stderr)
    return 1


def _show_unprintable_as_escapes() -> None:
    """Let print() survive a console that cannot encode a character (a non-ASCII ``--out`` on cp1252)."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="backslashreplace")


def run(task: Callable[[], None], *, json_errors: bool = True) -> int:
    """Run ``task``; a missing file or malformed input becomes ``error: <message>`` and exit code 1.

    Files are named by their file name only, never by their folder. With ``json_errors`` the task reads
    JSON inputs, so ``KeyError``/``TypeError``/``ArithmeticError`` are reported as bad input; pass
    ``json_errors=False`` when the task also runs model code, where those errors are bugs and propagate.
    """
    _show_unprintable_as_escapes()
    try:
        task()
    except OSError as exc:
        reason = exc.strerror or "a file could not be read or written"  # str(exc) may hold a whole path
        name = Path(str(exc.filename)).name if exc.filename else ""
        if name and Path(str(exc.filename)).is_dir():  # Windows reports opening a folder as permission denied
            reason = "is a folder, not a file"
        return _fail(f"{name}: {reason}" if name else reason)
    except ValueError as exc:
        return _fail(str(exc))
    except KeyError as exc:
        if not json_errors:
            raise
        return _fail(f"missing key {exc} in an input JSON file")
    except (TypeError, ArithmeticError):
        if not json_errors:
            raise
        return _fail("an input JSON file holds a value of the wrong type")
    return 0
