"""Progress reporting for library functions: they take a `log` callable and stay silent by default."""

from __future__ import annotations

from collections.abc import Callable

Log = Callable[[str], None]


def discard_log(message: str) -> None:
    """The default `log` of the library functions: say nothing (the command line passes `print`)."""
