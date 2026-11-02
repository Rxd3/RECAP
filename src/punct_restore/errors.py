"""Error text for the commands: the friendly reason of an OS error, the one-line message a command prints after
``error:``, and the stdout setting that lets that line print whatever characters it holds.

Nothing here imports another module of the package, so every command can print through it at no cost to its imports.
`punct_restore.prediction_io` still exports these names (restructure spec section 2).
"""

from __future__ import annotations

import errno
import os
import sys
from collections.abc import Mapping
from pathlib import Path

# The friendly reason for the OS errors a user causes (evaluate-models words every one of them this way).
OS_REASONS: Mapping[int | None, str] = {errno.EEXIST: "already exists; refusing to overwrite",
                                         errno.ENOENT: "file or folder not found",
                                         errno.EISDIR: "is a folder, not a file",
                                         errno.EACCES: "permission denied",
                                         errno.EPERM: "permission denied"}


def _is_folder(name: str | bytes) -> bool:
    """Whether ``name`` is a folder; False when it cannot even be looked at (a folder above it may not be entered)."""
    try:
        return Path(os.fsdecode(name)).is_dir()
    except OSError:
        return False


def os_error_code(exc: OSError) -> int | None:
    """The errno of ``exc``; EISDIR when Windows reports opening a single folder as permission denied."""
    single_file = bool(exc.filename) and exc.filename2 is None  # a two-path call (a rename) keeps its own reason
    if exc.errno in (errno.EACCES, errno.EPERM) and single_file and _is_folder(exc.filename):
        return errno.EISDIR
    return exc.errno


def os_reason(exc: OSError) -> str | None:
    """Why an OS error is shown: the folder wording for a folder opened as a file (Windows says "permission
    denied" for it), else the system text; None when the error has none."""
    return OS_REASONS[errno.EISDIR] if os_error_code(exc) == errno.EISDIR else exc.strerror


def one_line_error_with_short_paths(exc: BaseException, *paths: str | os.PathLike[str] | None) -> str:
    """The message on one line, without the ``error: `` prefix.

    An OSError reads ``<file name>: <reason>`` (its own text carries the full path, which is never shown);
    anything else keeps its text. Each of ``paths`` that the text quotes is cut to its file name.
    """
    text = str(exc)
    if isinstance(exc, OSError):
        reason = os_reason(exc)
        if reason:
            name = Path(os.fsdecode(exc.filename)).name if exc.filename else ""
            text = f"{name}: {reason}" if name else reason
    for path in (given for given in paths if given):
        text = text.replace(str(Path(path)), Path(path).name)
    return " ".join(text.splitlines())


def one_line_error(exc: BaseException) -> str:
    """The message on one line, in main's call form ``one_line_error(exc)`` (kept, spec section 2).

    A path quoted in the text of a non-OS error stays; ``one_line_error_with_short_paths`` also cuts the paths it is
    given. One wording differs from main: opening a folder as a file, which Windows reports as "permission denied",
    now reads ``<name>: is a folder, not a file``.
    """
    return one_line_error_with_short_paths(exc)


def reconfigure_stdout() -> None:
    """Print unencodable characters as backslash escapes instead of failing; a stdout that cannot is left alone."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="backslashreplace")
