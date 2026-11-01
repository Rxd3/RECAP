"""The old library name `punct_restore.data.read_ids` keeps the behaviour of main 72ebfd1 (restructure spec section 2).

Main read an `.ids` list as `[line.strip() for line in f if line.strip()]`. The commands refuse a duplicate or empty
list through `prediction_io.read_ids`; the library name does not, and the tests say so line by line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from punct_restore import data
from punct_restore.data import release


def _ids(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "list.ids"
    path.write_bytes(content)
    return path


def test_the_package_and_the_release_module_share_one_read_ids() -> None:
    assert data.read_ids is release.read_ids


def test_read_ids_keeps_a_repeated_id_each_time_it_occurs_in_file_order(tmp_path: Path) -> None:
    assert data.read_ids(_ids(tmp_path, b"a\nb\na\n")) == ["a", "b", "a"]


def test_read_ids_skips_blank_and_whitespace_only_lines_and_strips_each_id_including_crlf(tmp_path: Path) -> None:
    assert data.read_ids(_ids(tmp_path, b"a\r\n\r\n  b \n\t\nc")) == ["a", "b", "c"]


@pytest.mark.parametrize("content", [b"", b"\n", b" \r\n\t\n"], ids=["empty", "one-newline", "whitespace-only"])
def test_read_ids_reads_a_list_without_ids_as_an_empty_list(tmp_path: Path, content: bytes) -> None:
    assert data.read_ids(_ids(tmp_path, content)) == []


def test_read_ids_keeps_a_leading_byte_order_mark_in_the_first_id_as_main_did(tmp_path: Path) -> None:
    assert data.read_ids(_ids(tmp_path, b"\xef\xbb\xbfa\nb\n")) == ["﻿a", "b"]


def test_read_ids_raises_the_unicode_error_of_the_open_call_for_bytes_that_are_not_utf8(tmp_path: Path) -> None:
    with pytest.raises(UnicodeDecodeError):
        data.read_ids(_ids(tmp_path, b"a\n\xff\n"))


def test_read_ids_accepts_a_str_path_and_raises_file_not_found_for_a_missing_file(tmp_path: Path) -> None:
    assert data.read_ids(str(_ids(tmp_path, b"a\n"))) == ["a"]
    with pytest.raises(FileNotFoundError):
        data.read_ids(str(tmp_path / "gone.ids"))
