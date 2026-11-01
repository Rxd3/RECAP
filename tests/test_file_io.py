"""The file-level helpers of punct_restore.file_io: the JSONL reader, the new-folder writer, the JSON readers and the
shape check every study file goes through.

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
from punct_restore.data import labeling
from punct_restore.file_io import (
    iter_jsonl,
    read_json,
    read_json_object,
    read_object,
    read_yaml,
    require_folder,
    require_object,
    write_new_folder,
)
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


# ---------------------------------------------------------------- require_object / read_object
def test_require_object_returns_the_value_it_checked() -> None:
    value = {"a": {"b": [1]}, "extra": 1}
    assert require_object("f.json", value, {"a": {"b": list}}) is value


@pytest.mark.parametrize("value", [[], None, "text", 3], ids=["list", "null", "text", "number"])
def test_require_object_refuses_anything_but_an_object(value: Any) -> None:
    with pytest.raises(ValueError, match="^f.json: not a JSON object$"):
        require_object("f.json", value, {})


def test_require_object_names_a_missing_key_by_its_dotted_path() -> None:
    with pytest.raises(ValueError, match="^f.json: missing a.b.c$"):
        require_object("f.json", {"a": {"b": {}}}, {"a": {"b": {"c": object}}})


def test_require_object_names_a_nested_value_of_the_wrong_kind() -> None:
    with pytest.raises(ValueError, match="^f.json: a.b is not a JSON object$"):
        require_object("f.json", {"a": {"b": []}}, {"a": {"b": {}}})
    with pytest.raises(ValueError, match="^f.json: a.xs is not a JSON list$"):
        require_object("f.json", {"a": {"xs": {}}}, {"a": {"xs": list}})


def test_a_key_that_may_hold_any_value_only_has_to_be_present() -> None:
    require_object("f.json", {"a": None}, {"a": object})
    with pytest.raises(ValueError, match="^f.json: missing a$"):
        require_object("f.json", {}, {"a": object})


def test_the_prefix_places_an_element_of_a_list_in_the_message() -> None:
    with pytest.raises(ValueError, match="^f.json: candidates.0 is not a JSON object$"):
        require_object("f.json", 5, {}, "candidates.0")
    with pytest.raises(ValueError, match="^f.json: candidates.0.model is not nonempty text$"):
        require_object("f.json", {"model": ""}, {"model": str}, "candidates.0")


NOT_TEXT = [{}, ["O"], 1, None, True, "", "   "]


@pytest.mark.parametrize("item", NOT_TEXT, ids=["object", "list", "number", "null", "bool", "empty", "blank"])
def test_a_list_of_text_names_the_first_item_that_is_not_nonempty_text(item: Any) -> None:
    """A check that only asks for a list lets `[{}]` reach code that uses the items as dict keys."""
    with pytest.raises(ValueError, match=r"^f.json: a.xs.1 is not nonempty text$"):
        require_object("f.json", {"a": {"xs": ["O", item, "x"]}}, {"a": {"xs": list[str]}})


@pytest.mark.parametrize("value", [{}, "O", None, 3], ids=["object", "text", "null", "number"])
def test_a_list_of_text_that_is_no_list_is_refused_as_a_list(value: Any) -> None:
    with pytest.raises(ValueError, match="^f.json: xs is not a JSON list$"):
        require_object("f.json", {"xs": value}, {"xs": list[str]})


def test_a_list_of_text_may_be_empty_and_is_returned_as_it_was() -> None:
    value = {"xs": [], "ys": ["O", "COMMA"]}
    assert require_object("f.json", value, {"xs": list[str], "ys": list[str]}) is value


@pytest.mark.parametrize("item", [True, "1", None, {}, [1]], ids=["bool", "text", "null", "object", "list"])
def test_a_list_of_numbers_takes_numbers_only_and_no_bool(item: Any) -> None:
    require_object("f.json", {"xs": [1, 2.5]}, {"xs": list[float]})
    with pytest.raises(ValueError, match="^f.json: xs.0 is not a number$"):
        require_object("f.json", {"xs": [item]}, {"xs": list[float]})


@pytest.mark.parametrize("value", [2.5, True, "3", None, [3]], ids=["float", "bool", "text", "null", "list"])
def test_a_whole_number_is_an_int_but_no_bool_or_float(value: Any) -> None:
    require_object("f.json", {"n": 3}, {"n": int})
    with pytest.raises(ValueError, match="^f.json: n is not a whole number$"):
        require_object("f.json", {"n": value}, {"n": int})


def test_read_object_names_the_file_without_its_folder(tmp_path: Path) -> None:
    folder = tmp_path / "deep"
    folder.mkdir()
    _write(folder / "r.json", '{"a": []}')
    with pytest.raises(ValueError) as caught:
        read_object(folder / "r.json", {"b": object})
    assert str(caught.value) == "r.json: missing b"
    assert read_object(folder / "r.json", {"a": list}) == {"a": []}


@pytest.mark.parametrize(
    ("content", "message"), [(b"{", "r.json: not valid JSON"), (b"\xff", "r.json: not valid UTF-8")]
)
def test_read_object_keeps_the_json_reader_messages(tmp_path: Path, content: bytes, message: str) -> None:
    with pytest.raises(ValueError) as caught:
        read_object(_write(tmp_path / "r.json", content), {})
    assert str(caught.value) == message


def test_read_json_object_is_read_object_with_no_required_key(tmp_path: Path) -> None:
    """One checker, one wording: the object reader refuses what the shape check refuses, in the same words."""
    path = _write(tmp_path / "a.json", "null")
    with pytest.raises(ValueError) as one:
        read_json_object(path)
    with pytest.raises(ValueError) as other:
        read_object(path, {})
    assert str(one.value) == str(other.value) == "a.json: not a JSON object"


# ---------------------------------------------------------------- read_yaml
@pytest.mark.parametrize(("text", "value"), [("", None), ("[1, 2]\n", [1, 2]), ("a: 1\n", {"a": 1}), ("null\n", None)])
def test_read_yaml_gives_whatever_the_file_holds(tmp_path: Path, text: str, value: object) -> None:
    """The reader does not look at the shape: that is for the caller's schema (the label config, an experiment)."""
    assert read_yaml(_write(tmp_path / "any.yaml", text)) == value


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (b"a: [unclosed", "not valid YAML"),
        (b"a: " + b"[" * 100_000, "not valid YAML"),
        (b"a: \xe9\xff", "not valid UTF-8"),
    ],
    ids=["syntax", "nested", "not-utf8"],
)
def test_read_yaml_names_the_file_only_and_gives_no_parser_text(tmp_path: Path, raw: bytes, reason: str) -> None:
    folder = tmp_path / "secretfolder"
    folder.mkdir()
    with pytest.raises(ValueError) as problem:
        read_yaml(_write(folder / "broken.yaml", raw))
    assert str(problem.value) == f"broken.yaml: {reason}"


def test_read_yaml_of_a_missing_file_is_an_oserror(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_yaml(tmp_path / "nope.yaml")


# ---------------------------------------------------------------- require_folder
def test_require_folder_accepts_a_folder(tmp_path: Path) -> None:
    assert require_folder(tmp_path, "data root") is None


def test_require_folder_names_a_missing_folder_by_its_name_only(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        require_folder(tmp_path / "deep" / "nowhere", "folder")
    assert str(caught.value) == "nowhere: folder not found"


def test_require_folder_says_what_the_missing_folder_was_for(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        require_folder(tmp_path / "nowhere", "data root")
    assert str(caught.value) == "nowhere: data root not found"


def test_require_folder_says_so_for_a_file_and_is_not_a_not_found_error(tmp_path: Path) -> None:
    path = _write(tmp_path / "notes.txt", "x")
    with pytest.raises(NotADirectoryError) as caught:
        require_folder(path, "folder")
    assert str(caught.value) == "notes.txt: is a file, not a folder"


def test_require_folder_gives_a_relative_path_its_own_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as caught:
        require_folder(Path("missing"), "folder")
    assert str(caught.value) == "missing: folder not found"


# ---------------------------------------------------------------- old names
@pytest.mark.parametrize("name", ["write_new_file", "refuse_existing", "require_file"])
def test_prediction_io_still_exports_the_helpers_that_moved_to_file_io(name: str) -> None:
    assert getattr(prediction_io, name) is getattr(file_io, name)


def test_labeling_still_exports_read_yaml_and_it_is_the_one_yaml_reader() -> None:
    assert labeling.read_yaml is file_io.read_yaml


def test_the_json_readers_are_importable_where_they_were_and_are_the_one_reader() -> None:
    assert hmm_experiment_protocol.read_json is file_io.read_json
    assert hmm_workflow.read_json_object is file_io.read_json_object
