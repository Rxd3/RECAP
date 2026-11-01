"""File-level helpers shared by the commands: the create-new writers for a file and a folder, the JSONL, JSON and
YAML readers, the shape check of a parsed file, and the checks that a path is a file or a folder or does not exist yet.

Every error names a file by its name only (never the folder it sits in), so a command can print it as one line.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, get_args, get_origin


def require_file(path: str | os.PathLike[str], what: str) -> None:
    """Raise unless ``path`` is a file, in one line that names it by name only (the folder is never shown).

    Raises:
        IsADirectoryError: ``<name>: is a folder, not a file`` for a folder.
        FileNotFoundError: ``<name>: <what> not found`` for anything else that is not a file.
    """
    path = Path(path)
    name = Path(os.path.abspath(path)).name or str(path)
    if path.is_dir():
        raise IsADirectoryError(f"{name}: is a folder, not a file")
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {what} not found")


def require_folder(path: str | os.PathLike[str], what: str) -> None:
    """Raise unless ``path`` is a folder, in one line that names it by name only (the parent is never shown).

    Raises:
        NotADirectoryError: ``<name>: is a file, not a folder`` for a file.
        FileNotFoundError: ``<name>: <what> not found`` for anything else that is not a folder.
    """
    path = Path(path)
    name = Path(os.path.abspath(path)).name or str(path)
    if path.is_file():
        raise NotADirectoryError(f"{name}: is a file, not a folder")
    if not path.is_dir():
        raise FileNotFoundError(f"{name}: {what} not found")


def refuse_existing(*paths: Path | None) -> None:
    """Raise FileExistsError for the first of ``paths`` that already exists; ``None`` entries are skipped.

    A dangling symlink counts as existing. Nothing is touched. The error holds the path as given, so each command
    words it its own way (``one_line_error`` shows the file name only).
    """
    for path in paths:
        if path is not None and os.path.lexists(path):
            raise FileExistsError(errno.EEXIST, "already exists; refusing to overwrite", str(path))


def write_new_file(path: Path, text: str, *, newline: str | None = "\n") -> None:
    """Create ``path`` holding ``text`` as UTF-8, with LF line ends unless ``newline`` says otherwise.

    A file that already exists is never replaced and never deleted (FileExistsError). If writing fails, the
    half-written file this call created is deleted (a delete that itself fails is ignored) and the write
    error is re-raised.
    """
    handle = open(path, "x", encoding="utf-8", newline=newline)  # FileExistsError here: not our file
    try:
        with handle:
            handle.write(text)
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        raise


def _fill(path: Path, content: str | Callable[[Path], None]) -> None:
    """Make ``path`` (and its folders): ``content`` as UTF-8 text with LF line ends, or let ``content`` write it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8", newline="\n")
    else:
        content(path)


def write_new_folder(out: Path, files: Mapping[str, str | Callable[[Path], None]]) -> None:
    """Create the folder ``out`` holding ``files``, all or nothing: there is never a partial ``out``.

    ``files`` maps a path inside the folder (``"sub/name.txt"``; the folders are made) to the file's text, written as
    UTF-8 with LF line ends, or to a function that writes the file at the path it is given (for a streamed file).
    The files are written in the order given, into a temporary sibling folder that is renamed to ``out`` at the end.
    The folder that holds ``out`` is created when missing. If anything fails, the temporary folder is removed and
    the error is re-raised. The caller checks beforehand that ``out`` does not exist (an existing ``out`` is not
    looked at here).
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(dir=out.parent, prefix=f".{out.name}."))
    try:
        for name, content in files.items():
            _fill(temp / name, content)
        os.replace(temp, out)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def iter_jsonl[T](path: str | os.PathLike[str], parse: Callable[[str, str], T]) -> Iterator[tuple[int, T]]:
    """``(line number, parse(line, where))`` for every nonblank line of a UTF-8 JSONL file, in file order.

    ``parse`` gets the line without surrounding whitespace and ``where``, ``<file name> line <number>``, which
    starts the ValueError it raises for a bad line (``prediction_io.parse_json_object`` and ``parse_sample_record``
    do). Blank lines are skipped but counted in the numbering. The file is read lazily, a line at a time.

    Raises:
        OSError: If the file cannot be read.
        ValueError: ``<file name>: not valid UTF-8``, or what ``parse`` raises.
    """
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    yield number, parse(line.strip(), f"{path.name} line {number}")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: not valid UTF-8") from exc


def read_json(path: Path) -> Any:
    """Parse a UTF-8 JSON file.

    Raises:
        OSError: If the file cannot be read.
        ValueError: One line, ``<file name>: not valid UTF-8`` or ``<file name>: not valid JSON`` (also for a
            document nested too deeply or a number beyond Python's integer digit limit).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: not valid UTF-8") from exc
    try:
        return json.loads(text)
    except (ValueError, RecursionError) as exc:  # JSONDecodeError and the digit limit are ValueErrors
        raise ValueError(f"{path.name}: not valid JSON") from exc


def read_yaml(path: str | os.PathLike[str]) -> Any:
    """The parsed YAML of `path`, whatever it holds (a mapping, a list, a number or nothing at all).

    Raises:
        OSError: If the file cannot be read.
        ValueError: One line that names the file only (a YAML error mark shows the whole path):
            `<file name>: not valid UTF-8` or `<file name>: not valid YAML` (also for a document nested too deeply).
    """
    import yaml  # PyYAML is a declared runtime dependency of RECAP; imported here so file_io loads without it
    try:
        with open(path, encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{os.path.basename(path)}: not valid UTF-8") from exc
    except (yaml.YAMLError, RecursionError) as exc:  # RecursionError: nested too deeply
        raise ValueError(f"{os.path.basename(path)}: not valid YAML") from exc


_KIND_NAMES = {str: "nonempty text", list: "a JSON list", float: "a number", int: "a whole number"}


def _has_kind(value: Any, kind: type) -> bool:
    """Whether a JSON value is of the kind a schema asks for (`object`: anything; numbers: never a bool)."""
    if kind is str:
        return isinstance(value, str) and bool(value.strip())
    if kind is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return kind is object or isinstance(value, kind)


def _check_kind(name: str, value: Any, want: Any, place: str) -> None:
    """Require `value`, found at `place` in the file `name`, to match one entry of a schema (see `require_object`)."""
    if isinstance(want, Mapping):
        require_object(name, value, want, place)
    elif get_origin(want) is list:
        _check_kind(name, value, list, place)
        (item_kind,) = get_args(want)
        for number, item in enumerate(value):
            _check_kind(name, item, item_kind, f"{place}.{number}")
    elif not _has_kind(value, want):
        raise ValueError(f"{name}: {place} is not {_KIND_NAMES[want]}")


def require_object(name: str, value: Any, schema: Mapping[str, Any], at: str = "") -> dict[str, Any]:
    """Require `value`, the content of the file `name`, to be a JSON object that holds every key of `schema`.

    A schema value says what its key must hold: `object` anything (the key only has to be there), `str` nonempty
    text, `float` a number, `int` a whole number, `list` a JSON list, `list[str]` (or `list[float]`, `list[int]`) a
    list whose every item is of that kind, or a schema of its own for a nested object (`{}`: any object). Other
    keys are not looked at. `at` is where `value` sits inside the file (a dotted path), for a nested call.

    Raises:
        ValueError: One line, `<name>: <what is wrong>`, that names the key (or the list item) by its dotted path.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{name}: {at} is not a JSON object" if at else f"{name}: not a JSON object")
    for key, want in schema.items():
        place = f"{at}.{key}" if at else key
        if key not in value:
            raise ValueError(f"{name}: missing {place}")
        _check_kind(name, value[key], want, place)
    return value


def read_object(path: Path, schema: Mapping[str, Any]) -> dict[str, Any]:
    """The JSON object a UTF-8 file holds, checked against `schema` (see `read_json` and `require_object`).

    Raises:
        OSError: If the file cannot be read.
        ValueError: One line that starts with the file name only: not valid UTF-8, not valid JSON (also when it
            is nested too deeply), not a JSON object, or a key that is missing or of the wrong kind.
    """
    return require_object(path.name, read_json(path), schema)


def read_json_object(path: Path) -> dict[str, Any]:
    """The JSON object a UTF-8 file holds: `read_object` with no key required."""
    return read_object(path, {})
