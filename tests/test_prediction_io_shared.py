"""The shared readers and writer of src/punct_restore/prediction_io.py: read_ids, the JSON-line parsers and
write_new_file. Each test pins one point of the contract the old per-module copies each held a part of."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from punct_restore import data, prediction_io
from punct_restore.paths import REPO_ROOT

# --- read_ids ------------------------------------------------------------------------------------


def _ids(tmp_path: Path, content: bytes, name: str = "x.ids") -> Path:
    path = tmp_path / "deep" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(content)
    return path


def test_read_ids_keeps_the_order_of_the_file_not_a_sorted_order(tmp_path: Path) -> None:
    assert prediction_io.read_ids(_ids(tmp_path, b"c\na\nb\n")) == ["c", "a", "b"]


def test_read_ids_skips_blank_lines_and_strips_each_id_including_crlf(tmp_path: Path) -> None:
    assert prediction_io.read_ids(_ids(tmp_path, b"a\r\n\r\n  b \n\t\nc")) == ["a", "b", "c"]


def test_read_ids_drops_a_leading_byte_order_mark(tmp_path: Path) -> None:
    assert prediction_io.read_ids(_ids(tmp_path, b"\xef\xbb\xbfa\nb\n")) == ["a", "b"]


def test_read_ids_rejects_a_duplicate_naming_the_id_the_line_and_the_file_by_name_only(tmp_path: Path) -> None:
    path = _ids(tmp_path, b"a\n\nb\na\n")
    with pytest.raises(ValueError, match=r"^Duplicate ID 'a' found at line 4 in x\.ids$"):
        prediction_io.read_ids(path)


@pytest.mark.parametrize("content", [b"", b"\n", b" \r\n\t\n"])
def test_read_ids_rejects_a_file_without_ids(tmp_path: Path, content: bytes) -> None:
    with pytest.raises(ValueError, match=r"^x\.ids: ID list is empty$"):
        prediction_io.read_ids(_ids(tmp_path, content))


@pytest.mark.parametrize("content", [b"", b"\n \r\n"])
def test_read_ids_allow_empty_returns_an_empty_list_instead_of_the_error(tmp_path: Path, content: bytes) -> None:
    assert prediction_io.read_ids(_ids(tmp_path, content), allow_empty=True) == []


def test_read_ids_allow_empty_still_refuses_a_duplicate_and_bad_bytes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate ID 'a'"):
        prediction_io.read_ids(_ids(tmp_path, b"a\na\n"), allow_empty=True)
    with pytest.raises(ValueError, match=r"^x\.ids: not valid UTF-8$"):
        prediction_io.read_ids(_ids(tmp_path, b"\xff\n"), allow_empty=True)


def test_data_read_ids_returns_an_empty_list_for_an_empty_split(tmp_path: Path) -> None:
    """A release whose default_train list is empty (every train video flagged) still reads as []."""
    assert data.read_ids(str(_ids(tmp_path, b""))) == []


def test_read_ids_names_a_file_that_is_not_utf8(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"^x\.ids: not valid UTF-8$"):
        prediction_io.read_ids(_ids(tmp_path, b"a\n\xff\xfe\n"))


def test_read_ids_reports_a_missing_file_as_an_os_error_that_one_line_error_names_by_name(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        prediction_io.read_ids(tmp_path / "gone" / "ghost.ids")
    assert prediction_io.one_line_error(caught.value).startswith("ghost.ids: ")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"a\n\nb\n", r"^x\.ids line 2: IDs must be nonempty, with no whitespace or blank lines$"),
        (b"a\n b\n", r"^x\.ids line 2: IDs must be nonempty, with no whitespace or blank lines$"),
        (b"a\r\nb \r\n", r"^x\.ids line 2: IDs must be nonempty, with no whitespace or blank lines$"),
        (b"a\nb\n\n", r"^x\.ids line 3: IDs must be nonempty, with no whitespace or blank lines$"),
        (b"a\nb\na\n", r"^Duplicate ID 'a' found at line 3 in x\.ids$"),
        (b"", r"^x\.ids: ID list is empty$"),
    ],
)
def test_read_ids_strict_refuses_a_blank_line_a_padded_id_a_duplicate_and_an_empty_file(
        tmp_path: Path, content: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        prediction_io.read_ids(_ids(tmp_path, content), strict=True)


@pytest.mark.parametrize("content", [b"c\na\nb\n", b"c\r\na\r\nb\r\n", b"c\na\nb", b"\xef\xbb\xbfc\na\nb\n"])
def test_read_ids_strict_reads_a_clean_list_exactly_as_the_lenient_reader_does(
        tmp_path: Path, content: bytes) -> None:
    path = _ids(tmp_path, content)
    assert prediction_io.read_ids(path, strict=True) == prediction_io.read_ids(path) == ["c", "a", "b"]


def test_read_ids_accepts_a_str_path_as_the_documented_data_api_does(tmp_path: Path) -> None:
    assert data.read_ids(str(_ids(tmp_path, b"a\nb\n"))) == ["a", "b"]


# --- parse_json_object / parse_sample_record -----------------------------------------------------


@pytest.mark.parametrize("line", ["{not json", "[" * 100_000 + "]" * 100_000, ""])
def test_parse_json_object_says_not_valid_json_for_text_json_cannot_read(line: str) -> None:
    with pytest.raises(ValueError, match=r"^p\.jsonl line 7: not valid JSON$"):
        prediction_io.parse_json_object(line, "p.jsonl line 7")


@pytest.mark.parametrize("line", ["[1, 2]", "null", '"s"', "3"])
def test_parse_json_object_says_not_a_json_object_for_another_json_value(line: str) -> None:
    with pytest.raises(ValueError, match=r"^p\.jsonl line 7: not a JSON object$"):
        prediction_io.parse_json_object(line, "p.jsonl line 7")


def test_parse_json_object_returns_every_key_of_the_object() -> None:
    assert prediction_io.parse_json_object('{"a": 1, "b": [2]}\n', "w") == {"a": 1, "b": [2]}


@pytest.mark.parametrize("line", ['{"tokens": []}', '{"sample_id": 5}', '{"sample_id": ""}', '{"sample_id": "  "}'])
def test_parse_sample_record_needs_a_nonempty_string_sample_id(line: str) -> None:
    with pytest.raises(ValueError, match=r"^p\.jsonl line 7: sample_id must be a nonempty string$"):
        prediction_io.parse_sample_record(line, "p.jsonl line 7")


def test_parse_sample_record_checks_the_object_before_the_sample_id() -> None:
    with pytest.raises(ValueError, match=r"^w: not a JSON object$"):
        prediction_io.parse_sample_record("[]", "w")


def test_parse_sample_record_returns_the_whole_record() -> None:
    line = '{"sample_id": "s1", "split": "train", "tokens": ["a"]}'
    assert prediction_io.parse_sample_record(line, "w") == {"sample_id": "s1", "split": "train", "tokens": ["a"]}


# --- write_new_file ------------------------------------------------------------------------------


def test_write_new_file_writes_utf8_with_lf_newlines_by_default(tmp_path: Path) -> None:
    out = tmp_path / "o.txt"
    prediction_io.write_new_file(out, "é\nb\n")
    assert out.read_bytes() == b"\xc3\xa9\nb\n"


def test_write_new_file_can_write_the_platform_newline(tmp_path: Path) -> None:
    out = tmp_path / "o.txt"
    prediction_io.write_new_file(out, "a\nb\n", newline=None)
    assert out.read_bytes() == b"a" + os.linesep.encode() + b"b" + os.linesep.encode()


def test_write_new_file_never_replaces_a_file_and_does_not_delete_it_either(tmp_path: Path) -> None:
    out = tmp_path / "o.txt"
    out.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        prediction_io.write_new_file(out, "new")
    assert out.read_bytes() == b"keep me"


def test_write_new_file_deletes_its_own_half_written_file_when_writing_fails(tmp_path: Path) -> None:
    out = tmp_path / "o.txt"
    with pytest.raises(UnicodeEncodeError):
        prediction_io.write_new_file(out, "ok\ud800")  # a lone surrogate cannot be written as UTF-8
    assert not out.exists()


def test_write_new_file_creates_no_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prediction_io.write_new_file(tmp_path / "missing" / "o.txt", "x")
    assert not (tmp_path / "missing").exists()


# --- the file-level helpers are imported from their home -------------------------------------------

_SOURCE = REPO_ROOT / "src" / "punct_restore"
_FILE_HELPERS = {"require_file", "refuse_existing", "write_new_file"}  # prediction_io keeps them for old imports only
_IMPORT_HOME_MODULES = [
    "cli/train_hidden_event_lm.py", "data/extraction.py", "models/crf_io.py", "models/crf_prediction.py",
    "models/hidden_event_lm.py", "pipeline.py", "reporting/asr_noise_table.py",
]


@pytest.mark.parametrize("module", _IMPORT_HOME_MODULES)
def test_a_module_imports_the_file_helpers_from_file_io_not_from_the_prediction_io_re_export(module: str) -> None:
    tree = ast.parse((_SOURCE / module).read_text(encoding="utf-8"))
    through_prediction_io = sorted(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("prediction_io")
        for alias in node.names
        if alias.name in _FILE_HELPERS
    )
    assert through_prediction_io == []
