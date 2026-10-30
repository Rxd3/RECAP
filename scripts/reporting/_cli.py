"""Error handling shared by the reporting command lines: one friendly stderr line, exit code 1."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path


def _fail(message: str) -> int:
    print("error: " + " ".join(message.splitlines()), file=sys.stderr)
    return 1


def run(task: Callable[[], None], *, json_errors: bool = True) -> int:
    """Run ``task``; a missing file or malformed input becomes ``error: <message>`` and exit code 1.

    Files are named by their file name only, never by their folder. With ``json_errors`` the task reads
    JSON inputs, so ``KeyError``/``TypeError``/``ArithmeticError`` are reported as bad input; pass
    ``json_errors=False`` when the task also runs model code, where those errors are bugs and propagate.
    """
    try:
        task()
    except OSError as exc:
        name = Path(str(exc.filename)).name if exc.filename else ""
        return _fail(f"{name}: {exc.strerror}" if name else str(exc))
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
