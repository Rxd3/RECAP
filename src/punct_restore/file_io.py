"""File-level helpers shared by the commands: the create-new writers for a file and a folder, the JSONL, JSON and
YAML readers, the shape check of a parsed file, the checks that a path is a file or a folder or does not exist yet, the
name a message shows for a path, the file identity recorded in manifests and the "inside this folder" check.

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

import yaml

from .hashing import sha256_file
from .paths import REPO_ROOT as ROOT


def short_name(path: str | os.PathLike[str]) -> str:
    """The last part of ``path`` for a message (never the folder it sits in); the path itself when it has no name.

    A relative path gets the name of what it points at (``"."`` is the current folder's name); a drive or root, which
    has none, is shown as it was given.
    """
    return Path(os.path.abspath(path)).name or str(path)


def require_file(path: str | os.PathLike[str], what: str) -> None:
    """Raise unless ``path`` is a file, in one line that names it by name only (the folder is never shown).

    Raises:
        IsADirectoryError: ``<name>: is a folder, not a file`` for a folder.
        FileNotFoundError: ``<name>: <what> not found`` for anything else that is not a file.
    """
    name = short_name(path)
    path = Path(path)
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
    name = short_name(path)
    path = Path(path)
    if path.is_file():
        raise NotADirectoryError(f"{name}: is a file, not a folder")
    if not path.is_dir():
        raise FileNotFoundError(f"{name}: {what} not found")


def require_release(data_root: str | os.PathLike[str], release: str) -> Path:
    """The folder ``<data_root>/processed/<release>``, after checking the data root and that folder exist.

    A release exists once ``recap prepare-data build`` has made its processed folder. Every message names a folder by
    name only (the parent is never shown).

    Raises:
        NotADirectoryError: ``<name>: is a file, not a folder`` for a data root that is a file.
        FileNotFoundError: ``<name>: data root not found`` for a data root that is not there, or
            ``<release>: release folder not found in <data root name>; run recap prepare-data build first``.
    """
    require_folder(data_root, "data root")
    processed = Path(data_root) / "processed" / release
    if not processed.is_dir():
        raise FileNotFoundError(
            f"{release}: release folder not found in {short_name(data_root)}; run recap prepare-data build first")
    return processed


def file_identity_with(path: str | Path, root: Path, outside_parts: int) -> dict[str, str]:
    """POSIX path relative to the repository ``root``, and SHA-256; no absolute paths.

    A file outside ``root`` shows its last ``outside_parts`` path parts (2: folder and file name; 1: file name).
    """
    resolved = Path(path).resolve()
    try:
        shown = resolved.relative_to(root).as_posix()
    except ValueError:
        shown = "/".join(resolved.parts[-outside_parts:])
    return {"path": shown, "sha256": sha256_file(resolved)}


def file_identity(path: str | Path) -> dict[str, str]:
    """Repo-relative POSIX path (folder/name outside the repo) and SHA-256; no absolute paths."""
    return file_identity_with(path, ROOT, 2)


def inside_folder(base: str, rel: str) -> str | None:
    """The real path of ``rel`` below ``base``, or None when it resolves outside (``..``, an absolute path, a link).

    ``base`` is itself a real path (``os.path.realpath``), so a caller checking many paths resolves it once.
    """
    path = os.path.realpath(os.path.join(base, rel))
    try:
        inside = os.path.commonpath([base, path]) == base
    except ValueError:  # on Windows, a path on another drive has no common path with base
        return None
    return path if inside else None


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
    the error is re-raised; a temporary folder that cannot be made is an OSError that names ``out``. The caller
    checks beforehand that ``out`` does not exist (an existing ``out`` is not looked at here).
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        temp = Path(tempfile.mkdtemp(dir=out.parent, prefix=f".{out.name}."))
    except OSError as exc:  # its filename is the temporary name, which the user never gave
        raise OSError(exc.errno, exc.strerror, str(out)) from exc
    try:
        for name, content in files.items():
            _fill(temp / name, content)
        os.replace(temp, out)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def iter_jsonl[T](
    path: str | os.PathLike[str],
    parse: Callable[[str, str], T],
    *,
    encoding: str = "utf-8",
    skip_blank: bool = True,
) -> Iterator[tuple[int, T]]:
    """``(line number, parse(line, where))`` for every nonblank line of a UTF-8 JSONL file, in file order.

    ``parse`` gets the line without surrounding whitespace and ``where``, ``<file name> line <number>``, which
    starts the ValueError it raises for a bad line (``prediction_io.parse_json_object`` and ``parse_sample_record``
    do). Blank lines are skipped but counted in the numbering, unless ``skip_blank`` is False: then one is refused.
    ``encoding`` is ``utf-8-sig`` for a file that may start with a byte order mark. The file is read lazily, a line
    at a time.

    Raises:
        OSError: If the file cannot be read.
        ValueError: ``<file name>: not valid UTF-8``, ``<file name> line <number>: blank JSONL record`` (with
            ``skip_blank`` False), or what ``parse`` raises.
    """
    path = Path(path)
    try:
        with open(path, encoding=encoding) as handle:
            for number, line in enumerate(handle, 1):
                text, where = line.strip(), f"{path.name} line {number}"
                if text:
                    yield number, parse(text, where)
                elif not skip_blank:
                    raise ValueError(f"{where}: blank JSONL record")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: not valid UTF-8") from exc


def read_json(path: Path, *, encoding: str = "utf-8") -> Any:
    """Parse a JSON file (UTF-8; ``encoding="utf-8-sig"`` also drops a leading byte order mark).

    Raises:
        OSError: If the file cannot be read.
        ValueError: One line, ``<file name>: not valid UTF-8`` or ``<file name>: not valid JSON`` (also for a
            document nested too deeply or a number beyond Python's integer digit limit).
    """
    try:
        text = path.read_text(encoding=encoding)
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
    try:
        with open(path, encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{os.path.basename(path)}: not valid UTF-8") from exc
    except (yaml.YAMLError, RecursionError) as exc:  # RecursionError: nested too deeply
        raise ValueError(f"{os.path.basename(path)}: not valid YAML") from exc


_KIND_NAMES = {str: "nonempty text", list: "a JSON list", float: "a number", int: "a whole number"}


def _kind_name(kind: type, is_yaml: bool) -> str:
    """How a message calls a kind: a YAML file holds a list, not a JSON list."""
    return "a list" if is_yaml and kind is list else _KIND_NAMES[kind]


def _has_kind(value: Any, kind: type) -> bool:
    """Whether a JSON value is of the kind a schema asks for (`object`: anything; numbers: never a bool)."""
    if kind is str:
        return isinstance(value, str) and bool(value.strip())
    if kind is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return kind is object or isinstance(value, kind)


def _check_kind(name: str, value: Any, want: Any, place: str, is_yaml: bool) -> None:
    """Require `value`, found at `place` in the file `name`, to match one entry of a schema (see `require_object`)."""
    if isinstance(want, Mapping):
        require_object(name, value, want, place, is_yaml=is_yaml)
    elif get_origin(want) is list:
        _check_kind(name, value, list, place, is_yaml)
        (item_kind,) = get_args(want)
        for number, item in enumerate(value):
            _check_kind(name, item, item_kind, f"{place}.{number}", is_yaml)
    elif not _has_kind(value, want):
        raise ValueError(f"{name}: {place} is not {_kind_name(want, is_yaml)}")


def require_object(
    name: str, value: Any, schema: Mapping[str, Any], at: str = "", *, is_yaml: bool = False
) -> dict[str, Any]:
    """Require `value`, the content of the file `name`, to be a JSON object that holds every key of `schema`.

    A schema value says what its key must hold: `object` anything (the key only has to be there), `str` nonempty
    text, `float` a number, `int` a whole number, `list` a JSON list, `list[str]` (or `list[float]`, `list[int]`) a
    list whose every item is of that kind, or a schema of its own for a nested object (`{}`: any object). Other
    keys are not looked at. `at` is where `value` sits inside the file (a dotted path), for a nested call. For a
    YAML file (`is_yaml`) the message says "a mapping" and "a list" where a JSON file's says "a JSON object" and
    "a JSON list".

    Raises:
        ValueError: One line, `<name>: <what is wrong>`, that names the key (or the list item) by its dotted path.
    """
    if not isinstance(value, dict):
        noun = "a mapping" if is_yaml else "a JSON object"
        raise ValueError(f"{name}: {at} is not {noun}" if at else f"{name}: not {noun}")
    for key, want in schema.items():
        place = f"{at}.{key}" if at else key
        if key not in value:
            raise ValueError(f"{name}: missing {place}")
        _check_kind(name, value[key], want, place, is_yaml)
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
