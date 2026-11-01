"""The file-level helpers of punct_restore.file_io: the JSONL reader, the new-folder writer and the JSON readers.

The older helpers (write_new_file, refuse_existing, require_file) are tested through prediction_io, which still
exports them: the last test pins that those names are the very objects of file_io.
"""

from __future__ import annotations

import errno
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from punct_restore import file_io, prediction_io
from punct_restore.file_io import iter_jsonl, read_json, read_json_object, write_new_folder
from punct_restore.models import hmm_experiment_protocol, hmm_workflow
from punct_restore.prediction_io import parse_json_object, parse_sample_record


def _write(path: Path, content: bytes | str) -> Path:
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return path


# ---------------------------------------------------------------- iter_jsonl
def test_iter_jsonl_yields_the_line_number_and_the_record_and_counts_blank_lines(tmp_path: Path) -> None:
    path = _write(tmp_path / "f.jsonl", '{"sample_id": "a"}\n\n   \n{"sample_id": "b"}\n')
    assert [(number, record["sample_id"]) for number, record in iter_jsonl(path, parse_sample_record)] == [
        (1, "a"), (4, "b")]


def test_iter_jsonl_hands_the_parser_the_line_and_the_file_name_with_the_line_number(tmp_path: Path) -> None:
    path = _write(tmp_path / "f.jsonl", "first\n\nthird\n")
    seen: list[tuple[str, str]] = []

    def parse(line: str, where: str) -> str:
        seen.append((line.strip(), where))
        return line.strip()

    assert [record for _, record in iter_jsonl(path, parse)] == ["first", "third"]
    assert seen == [("first", "f.jsonl line 1"), ("third", "f.jsonl line 3")]


def test_iter_jsonl_reads_utf8(tmp_path: Path) -> None:
    path = _write(tmp_path / "f.jsonl", json.dumps({"sample_id": "é-ü-日本"}, ensure_ascii=False) + "\n")
    assert [record["sample_id"] for _, record in iter_jsonl(path, parse_sample_record)] == ["é-ü-日本"]


def test_iter_jsonl_names_the_file_when_the_bytes_are_not_utf8(tmp_path: Path) -> None:
    path = _write(tmp_path / "f.jsonl", b'{"sample_id": "a"}\n\xff\xfe\n')
    with pytest.raises(ValueError) as problem:
        list(iter_jsonl(path, parse_sample_record))
    assert str(problem.value) == "f.jsonl: not valid UTF-8"


GOOD_LINE = '{"sample_id": "a"}\n'


@pytest.mark.parametrize(("second_line", "message"), [
    ("[" * 100_000 + "\n", "f.jsonl line 2: not valid JSON"),
    ("[1]\n", "f.jsonl line 2: not a JSON object"),
    ('{"id": 1}\n', "f.jsonl line 2: sample_id must be a nonempty string"),
], ids=["nested", "not-an-object", "no-sample-id"])
def test_iter_jsonl_lets_the_parser_name_the_file_and_line(tmp_path: Path, second_line: str, message: str) -> None:
    path = _write(tmp_path / "f.jsonl", GOOD_LINE + second_line)
    with pytest.raises(ValueError) as problem:
        list(iter_jsonl(path, parse_sample_record))
    assert str(problem.value) == message


def test_iter_jsonl_of_a_missing_file_is_a_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_jsonl(tmp_path / "nope.jsonl", parse_json_object))


# ---------------------------------------------------------------- write_new_folder
def test_write_new_folder_writes_text_as_utf8_with_lf_and_makes_subfolders(tmp_path: Path) -> None:
    out = tmp_path / "made" / "out"  # the parent folder does not exist yet
    write_new_folder(out, {"a.ids": "x\ny\n", "sub/deeper/b.txt": "é\r\n"})
    assert (out / "a.ids").read_bytes() == b"x\ny\n"
    assert (out / "sub" / "deeper" / "b.txt").read_bytes() == "é\r\n".encode()
    assert sorted(path.name for path in (tmp_path / "made").iterdir()) == ["out"]  # no temp folder left


def test_write_new_folder_gives_a_writer_function_a_path_inside_a_temporary_folder_not_out(tmp_path: Path) -> None:
    out = tmp_path / "out"
    given: list[Path] = []

    def writer(path: Path) -> None:
        given.append(path)
        assert not out.exists()  # out appears only when everything is written
        path.write_bytes(b"streamed")

    write_new_folder(out, {"data/rows.jsonl": writer})
    assert given[0].parent.parent != out and given[0].name == "rows.jsonl"
    assert (out / "data" / "rows.jsonl").read_bytes() == b"streamed"


def test_write_new_folder_writes_the_files_in_the_order_given(tmp_path: Path) -> None:
    order: list[str] = []

    def writer(name: str) -> Callable[[Path], None]:
        return lambda path: (order.append(name), path.write_text(name))[0]

    write_new_folder(tmp_path / "out", {"b": writer("b"), "a": writer("a"), "c": writer("c")})
    assert order == ["b", "a", "c"]


@pytest.mark.parametrize("failure", [OSError(errno.ENOSPC, "No space left on device"), KeyboardInterrupt()],
                         ids=["os-error", "keyboard-interrupt"])
def test_write_new_folder_leaves_neither_out_nor_a_temporary_folder_when_a_file_fails(
        tmp_path: Path, failure: BaseException) -> None:
    def fail(path: Path) -> None:
        raise failure

    with pytest.raises(type(failure)) as problem:
        write_new_folder(tmp_path / "out", {"first.txt": "written", "second.txt": fail})
    assert problem.value is failure  # the original error, not a wrapper
    assert list(tmp_path.iterdir()) == []


def test_write_new_folder_leaves_nothing_when_the_final_rename_fails(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def deny(src: object, dst: object) -> None:
        raise PermissionError(errno.EACCES, "Permission denied", str(dst))

    monkeypatch.setattr(os, "replace", deny)
    with pytest.raises(PermissionError):
        write_new_folder(tmp_path / "out", {"a.txt": "x"})
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- read_json / read_json_object
@pytest.mark.parametrize(("content", "message"), [
    (b"\xff\xfe", "doc.json: not valid UTF-8"),
    (b"{oops", "doc.json: not valid JSON"),
    (b"[" * 100_000, "doc.json: not valid JSON"),
    (b'{"n": ' + b"9" * 5000 + b"}", "doc.json: not valid JSON"),  # beyond Python's integer digit limit
], ids=["not-utf8", "syntax", "nested", "huge-integer"])
@pytest.mark.parametrize("reader", [read_json, read_json_object, hmm_experiment_protocol.read_json,
                                    hmm_workflow.read_json_object])
def test_a_json_reader_gives_one_line_naming_the_file(
        reader: Callable[[Path], Any], tmp_path: Path, content: bytes, message: str) -> None:
    with pytest.raises(ValueError) as problem:
        reader(_write(tmp_path / "doc.json", content))
    assert str(problem.value) == message


def test_read_json_returns_any_json_value_and_read_json_object_only_an_object(tmp_path: Path) -> None:
    array = _write(tmp_path / "array.json", "[1, 2]")
    assert read_json(array) == [1, 2]
    with pytest.raises(ValueError) as problem:
        read_json_object(array)
    assert str(problem.value) == "array.json: not a JSON object"
    assert read_json_object(_write(tmp_path / "o.json", '{"é": 1}')) == {"é": 1}


def test_a_missing_json_file_is_an_os_error_for_both_readers(tmp_path: Path) -> None:
    for reader in (read_json, read_json_object):
        with pytest.raises(FileNotFoundError):
            reader(tmp_path / "nope.json")


# ---------------------------------------------------------------- old names
@pytest.mark.parametrize("name", ["write_new_file", "refuse_existing", "require_file"])
def test_prediction_io_still_exports_the_helpers_that_moved_to_file_io(name: str) -> None:
    assert getattr(prediction_io, name) is getattr(file_io, name)


def test_the_json_readers_are_importable_where_they_were_and_are_the_one_reader() -> None:
    assert hmm_experiment_protocol.read_json is file_io.read_json
    assert hmm_workflow.read_json_object is file_io.read_json_object
