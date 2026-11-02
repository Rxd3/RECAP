"""Input/output shared by the commands: gold-free prediction files and manifests (train_crf, tune_threshold,
project_joint, factor_predictions), the one ``.ids`` reader, the JSON-line parsers, the one record-selection reader and
output path checks. The file-level helpers (create-new writers, JSONL and JSON readers, the file identity) are in
punct_restore.file_io; the one-line error text is in punct_restore.errors."""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .file_io import file_identity, iter_jsonl, require_file
from .hashing import sha256_file

# isort: split
# old import paths kept: these helpers moved to file_io and to errors (restructure spec section 2)
from .errors import OS_REASONS as OS_REASONS
from .errors import one_line_error as one_line_error
from .errors import one_line_error_with_short_paths as one_line_error_with_short_paths
from .errors import os_error_code as os_error_code
from .errors import os_reason as os_reason
from .errors import reconfigure_stdout as reconfigure_stdout
from .file_io import refuse_existing as refuse_existing
from .file_io import write_new_file as write_new_file


def read_ids(path: str | os.PathLike[str], *, strict: bool = False, allow_empty: bool = False) -> list[str]:
    """The sample IDs of an ``.ids`` list, in file order.

    One ID per line; a leading byte order mark is dropped. By default each ID is stripped and blank lines are
    skipped. With ``strict`` a blank line or an ID with surrounding whitespace is refused instead. A list
    without any ID is an error unless ``allow_empty`` (then ``[]``).

    Raises:
        OSError: If the file cannot be read (a missing file is a FileNotFoundError).
        ValueError: One line naming the file by name only: not valid UTF-8, a refused line (``strict``), a
            duplicate ID or no ID at all.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: not valid UTF-8") from exc
    lines = text.split("\n")
    if lines[-1] == "":  # the newline that ends the last line
        lines.pop()
    ids: list[str] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, 1):
        sample_id = line.strip()
        if strict and (not sample_id or sample_id != line):
            raise ValueError(f"{path.name} line {number}: IDs must be nonempty, with no whitespace or blank lines")
        if not sample_id:
            continue
        if sample_id in seen:
            raise ValueError(f"Duplicate ID {sample_id!r} found at line {number} in {path.name}")
        seen.add(sample_id)
        ids.append(sample_id)
    if not ids and not allow_empty:
        raise ValueError(f"{path.name}: ID list is empty")
    return ids


def read_frozen_ids(ids_path: Path) -> list[str]:
    """The IDs of a frozen ID list, as main's reader gave them (old import path kept, restructure spec §2).

    Blank lines are skipped and each ID is stripped, as `read_ids` does by default.

    Raises:
        FileNotFoundError: If the ID file does not exist (a folder counts as missing).
        ValueError: If the ID list is empty or contains duplicate IDs.
    """
    if not ids_path.is_file():
        raise FileNotFoundError(f"Frozen ID list not found: {ids_path.name}")  # main's wording (old library name)
    return read_ids(ids_path)


def parse_json_object(line: str, where: str) -> dict[str, Any]:
    """The JSON object on ``line``; ``where`` (file name and line number) starts the one-line ValueError."""
    try:
        record = json.loads(line)
    except (ValueError, RecursionError) as exc:  # RecursionError: a line nested too deeply
        raise ValueError(f"{where}: not valid JSON") from exc
    if not isinstance(record, dict):
        raise ValueError(f"{where}: not a JSON object")  # ValueError, not TypeError: the scripts report only that
    return record


def parse_sample_record(line: str, where: str) -> dict[str, Any]:
    """``parse_json_object`` for a record that also needs a nonempty string ``sample_id``."""
    record = parse_json_object(line, where)
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"{where}: sample_id must be a nonempty string")
    return record


def _gold_free_observation(rec: Mapping[str, Any], where: str) -> dict[str, Any]:
    """Keep only sample_id and validated observable tokens (strictly gold-free); ``where`` is ``<file> line N``."""
    s_id = rec["sample_id"]
    tokens = rec.get("tokens")
    if not tokens or not isinstance(tokens, list):
        raise ValueError(f"{where}: tokens of {s_id!r} must be a nonempty list")
    if any(not isinstance(t, str) or not t.strip() for t in tokens):
        raise ValueError(f"{where}: tokens of {s_id!r} must be nonempty, non-whitespace strings")
    return {"sample_id": s_id, "tokens": list(tokens)}


def read_observations(input_path: Path, ids_path: Path | None) -> list[dict[str, Any]]:
    """Read gold-free ``{"sample_id", "tokens"}`` records in INPUT FILE order.

    When ``ids_path`` is given, only the listed IDs are kept and every one must be present.

    Raises:
        FileNotFoundError: If the input JSONL or the ID list does not exist.
        IsADirectoryError: If the input JSONL is a folder.
        ValueError: On corrupt, duplicate or invalid records (a bad line reads ``<file> line N: <reason>``, a file
            that is not UTF-8 ``<file>: not valid UTF-8``), or IDs missing from the input.
    """
    require_file(input_path, "input JSONL")  # before the ID list is read: a missing input is named first
    return select_observations(input_path, read_ids(ids_path) if ids_path is not None else None)


def select_records[T](
    rows: Iterable[tuple[int, dict[str, Any]]],
    name: str,
    ids: Sequence[str] | None,
    keep: Callable[[dict[str, Any], str], T],
) -> Iterator[T]:
    """``keep(record, where)`` for each record of ``rows`` whose sample_id is in ``ids``, in file order.

    The one reader behind every command that keeps the listed records of a JSONL file. ``rows`` are the ``(line
    number, record)`` pairs of the file (``iter_jsonl``), each record holding a string ``sample_id``; ``name`` is how
    the messages call the file; ``ids`` is the list in list order, or ``None`` to keep every record. ``keep`` checks and
    reshapes one record (``where`` is ``<name> line <number>``, the start of the ValueError it raises). The records
    come one at a time; the list is checked after the last one.

    Raises:
        ValueError: ``<name> line N: duplicate sample_id 'x'`` for a sample_id seen before (listed or not), or, after
            the last record, ``<name>: N listed ID(s) are missing, first 'x'`` (first in list order); or what
            ``keep`` raises.
    """
    wanted = None if ids is None else set(ids)
    seen: set[str] = set()
    for number, record in rows:
        sample_id, where = record["sample_id"], f"{name} line {number}"
        if sample_id in seen:
            raise ValueError(f"{where}: duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
        if wanted is None or sample_id in wanted:
            yield keep(record, where)
    missing = [sample_id for sample_id in dict.fromkeys(ids or ()) if sample_id not in seen]
    if missing:
        raise ValueError(f"{name}: {len(missing)} listed ID(s) are missing, first {missing[0]!r}")


def select_observations(input_path: Path, ids: Sequence[str] | None) -> list[dict[str, Any]]:
    """``read_observations`` for an ID list that is already read (``None``: keep every record).

    Raises:
        FileNotFoundError: If the input JSONL does not exist.
        IsADirectoryError: If the input JSONL is a folder.
        ValueError: As ``read_observations``; the message starts with the input's file name.
    """
    require_file(input_path, "input JSONL")
    rows = iter_jsonl(input_path, parse_sample_record)
    return list(select_records(rows, input_path.name, ids, _gold_free_observation))


def write_predictions_jsonl(predictions: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write one JSON object per line with the platform newline (byte-stable across the team)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(pred, ensure_ascii=False) + "\n" for pred in predictions)


def write_prediction_manifest(
    manifest_path: Path,
    *,
    predictions_path: Path,
    input_path: Path,
    model_name: str,
    label_mode: str,
    ordered_labels: Sequence[str],
) -> None:
    """Write the evaluator-facing sidecar manifest for a predictions file."""
    manifest: dict[str, Any] = {
        "model_name": model_name,
        "label_mode": label_mode,
        "ordered_labels": list(ordered_labels),
        "predictions_sha256": sha256_file(predictions_path),
        "input": file_identity(input_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def write_prediction_outputs(
    predictions: Sequence[Mapping[str, Any]],
    output_path: Path,
    manifest_path: Path | None,
    *,
    input_path: Path,
    model_name: str,
    label_mode: str,
    ordered_labels: Sequence[str],
) -> None:
    """Write the predictions and, when given, their manifest; on any failure try to delete both.

    Cleanup is best effort: it deletes each file this call started writing (including a file it
    was overwriting), ignores a delete that itself fails, and leaves any folders it created.
    The original error is always the one re-raised.
    """
    started: list[Path] = []
    done = False
    try:
        started.append(output_path)
        write_predictions_jsonl(predictions, output_path)
        if manifest_path is not None:
            started.append(manifest_path)
            write_prediction_manifest(
                manifest_path,
                predictions_path=output_path,
                input_path=input_path,
                model_name=model_name,
                label_mode=label_mode,
                ordered_labels=ordered_labels,
            )
        done = True
    finally:
        if not done:
            for path in started:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)


def same_path(a: str | Path, b: str | Path) -> bool:
    """True when both paths name the same file once resolved (case-blind where the OS is)."""
    return _comparable(a) == _comparable(b)


def _comparable(path: str | Path) -> str:
    """Resolved path as the OS sees it: abspath drops Windows' trailing dots and spaces."""
    return os.path.normcase(os.path.abspath(Path(path).resolve()))


def _check_creatable(flag: str, path: Path) -> None:
    """Raise ValueError if ``path`` is a folder or its nearest existing ancestor is not a writable folder.

    An OSError while looking (e.g. a parent that cannot be read) also becomes a one-line ValueError.
    """
    try:
        if path.is_dir():
            raise ValueError(f"{flag}: {path.resolve().name or path.resolve()} is a folder, not a file")
        ancestor = path.resolve().parent
        while not ancestor.exists() and ancestor.parent != ancestor:
            ancestor = ancestor.parent
        shown = ancestor.name or ancestor.anchor
        if not ancestor.is_dir():
            raise ValueError(f"{flag}: cannot create {path.name}: {shown} is not a folder")
        if not os.access(ancestor, os.W_OK):
            raise ValueError(f"{flag}: cannot create {path.name}: {shown} is not writable")
    except OSError as e:
        raise ValueError(f"{flag}: cannot check {path.name or path}: {e.strerror or e}") from e


def check_new_output_paths(
    outputs: Mapping[str, Path | None], inputs: Mapping[str, Path | None]
) -> None:
    """Raise a one-line ValueError if an output is another output or an input, or cannot be created.

    Keys are the flag names (``"--output"``) used in the message; ``None`` entries are skipped.
    """
    targets = [(flag, Path(path)) for flag, path in outputs.items() if path is not None]
    sources = [(flag, Path(path)) for flag, path in inputs.items() if path is not None]
    for index, (flag, path) in enumerate(targets):
        for other_flag, other in [*targets[index + 1:], *sources]:
            if same_path(path, other):
                raise ValueError(f"{flag} and {other_flag} are the same file")
    for flag, path in targets:
        _check_creatable(flag, path)


def check_script_paths(outputs: Mapping[str, Path], inputs: Mapping[str, Path]) -> None:
    """Raise a one-line ValueError for a path collision, an unwritable or already existing output, or a missing input.

    Keys are the flag names used in the message. Nothing is read or written, so a script can call this first.
    """
    check_new_output_paths(outputs, inputs)
    for flag, path in outputs.items():
        if os.path.lexists(path):
            raise ValueError(f"{flag}: {path.name} already exists, refusing to overwrite it")
    for flag, path in inputs.items():
        if not path.is_file():
            raise ValueError(f"{flag}: {path.name} is not an existing file")


def read_prediction_lines(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line number, JSON object)`` for every non-blank line of a predictions JSONL file.

    Blank lines count in the numbering. The caller checks each record as it arrives.

    Raises:
        ValueError: One line naming the file (and line) only: invalid UTF-8, invalid JSON, a line that is not a
            JSON object, or a file without any record.
        OSError: If the file cannot be read.
    """
    found = False
    # utf-8-sig: a leading byte order mark is dropped, as the evaluator's reader does
    for number, record in iter_jsonl(path, parse_json_object, encoding="utf-8-sig"):
        found = True
        yield number, record
    if not found:
        raise ValueError(f"{path.name}: no predictions found")
