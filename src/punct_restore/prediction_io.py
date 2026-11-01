"""Input/output shared by the commands: gold-free prediction files and manifests (train_crf, tune_threshold,
project_joint, factor_predictions), the one ``.ids`` reader, the JSON-line parsers, the create-new-file writer,
output path checks and one-line errors."""

from __future__ import annotations

import contextlib
import errno
import json
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .evaluation.inputs import file_identity
from .hashing import sha256_file


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
        raise ValueError(f"Frozen ID list is empty: {path.name}")
    return ids


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


def _parse_observation_line(line_str: str, line_num: int, input_path: Path) -> dict[str, Any]:
    """Parse one JSONL line into a record with a valid sample_id."""
    try:
        rec = json.loads(line_str)
    except ValueError as e:
        raise ValueError(f"Corrupt JSON at line {line_num}: {e}") from e
    if not isinstance(rec, dict):
        raise ValueError(f"Line {line_num} in {input_path} is not a JSON object")
    s_id = rec.get("sample_id")
    if not isinstance(s_id, str) or not s_id.strip():
        raise ValueError(f"Line {line_num} has missing or empty sample_id")
    return rec


def _gold_free_observation(rec: Mapping[str, Any], line_num: int) -> dict[str, Any]:
    """Keep only sample_id and validated observable tokens (strictly gold-free)."""
    s_id = rec["sample_id"]
    tokens = rec.get("tokens")
    if not tokens or not isinstance(tokens, list):
        raise ValueError(f"Invalid tokens at line {line_num} for ID {s_id!r}")
    if any(not isinstance(t, str) or not t.strip() for t in tokens):
        raise ValueError(f"Tokens contain invalid elements at line {line_num} for ID {s_id!r}")
    return {"sample_id": s_id, "tokens": list(tokens)}


def read_observations(input_path: Path, ids_path: Path | None) -> list[dict[str, Any]]:
    """Read gold-free ``{"sample_id", "tokens"}`` records in INPUT FILE order.

    When ``ids_path`` is given, only the listed IDs are kept and every one must be present.

    Raises:
        FileNotFoundError: If the input JSONL or the ID list does not exist.
        ValueError: On corrupt, duplicate or invalid records, or IDs missing from the input.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL not found at {input_path}")
    target_ids = read_ids(ids_path) if ids_path is not None else None
    target_set = set(target_ids) if target_ids is not None else None

    observations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with open(input_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            rec = _parse_observation_line(line_str, line_num, input_path)
            s_id = rec["sample_id"]
            if s_id in seen_ids:
                raise ValueError(f"Duplicate sample_id {s_id!r} encountered at line {line_num}")
            seen_ids.add(s_id)
            if target_set is None or s_id in target_set:
                observations.append(_gold_free_observation(rec, line_num))

    if target_ids is not None:
        missing_ids = set(target_ids) - {o["sample_id"] for o in observations}
        if missing_ids:
            raise ValueError(
                f"Missing {len(missing_ids)} selected IDs from input file {input_path}. "
                f"Examples: {sorted(missing_ids)[:5]}"
            )
        if len(observations) != len(target_ids):
            raise ValueError(
                f"Found {len(observations)} matching records, expected {len(target_ids)} from ID list."
            )
    return observations


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
    try:
        text = path.read_text(encoding="utf-8-sig")  # a leading BOM is dropped, as the evaluator's reader does
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: not valid UTF-8") from exc
    found = False
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        record = parse_json_object(line, f"{path.name} line {number}")
        found = True
        yield number, record
    if not found:
        raise ValueError(f"{path.name}: no predictions found")


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


def one_line_error(exc: BaseException, *paths: str | os.PathLike[str] | None) -> str:
    """The message on one line, without the ``error: `` prefix.

    An OSError reads ``<file name>: <reason>`` (its own text carries the full path, which is never shown);
    anything else keeps its text. Each of ``paths`` that the text quotes is cut to its file name.
    """
    text = str(exc)
    if isinstance(exc, OSError):
        # the system text says "permission denied" for a folder on Windows: that one reason is rewritten
        reason = OS_REASONS[errno.EISDIR] if os_error_code(exc) == errno.EISDIR else exc.strerror
        if reason:
            name = Path(os.fsdecode(exc.filename)).name if exc.filename else ""
            text = f"{name}: {reason}" if name else reason
    for path in filter(None, paths):
        text = text.replace(str(Path(path)), Path(path).name)
    return " ".join(text.splitlines())


def reconfigure_stdout() -> None:
    """Print unencodable characters as backslash escapes instead of failing; a stdout that cannot is left alone."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="backslashreplace")
