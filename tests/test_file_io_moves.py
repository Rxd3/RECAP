"""Round 3, item 9: what moved into punct_restore.file_io (the file identity, the "inside this folder" check) and the
options its readers gained (a blank line refused, another encoding). The old import paths stay and name the same
objects.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from punct_restore import file_io, prediction_io
from punct_restore.evaluation import inputs
from punct_restore.file_io import inside_folder, iter_jsonl, read_json
from punct_restore.hashing import sha256_file
from punct_restore.paths import REPO_ROOT
from punct_restore.prediction_io import parse_json_object

BOM = "\ufeff"


# ---------------------------------------------------------------- the file identity
@pytest.mark.parametrize("name", ["file_identity", "file_identity_with"])
def test_the_file_identity_lives_in_file_io_and_the_old_homes_name_the_same_function(name: str) -> None:
    assert getattr(inputs, name) is getattr(file_io, name)
    if name == "file_identity":
        assert prediction_io.file_identity is file_io.file_identity


def test_file_io_gives_a_file_outside_the_repository_its_folder_and_name(tmp_path: Path) -> None:
    path = tmp_path / "folder" / "f.txt"
    path.parent.mkdir()
    path.write_bytes(b"x")
    assert file_io.file_identity(path) == {"path": "folder/f.txt", "sha256": sha256_file(path)}
    assert file_io.file_identity_with(path, tmp_path, 1) == {"path": "folder/f.txt", "sha256": sha256_file(path)}
    assert file_io.file_identity_with(path, tmp_path / "elsewhere", 1) == {
        "path": "f.txt", "sha256": sha256_file(path)}


# ---------------------------------------------------------------- inside_folder
def test_inside_folder_gives_the_real_path_of_a_file_below_the_folder(tmp_path: Path) -> None:
    base = os.path.realpath(tmp_path)
    assert inside_folder(base, "sub/a.txt") == os.path.realpath(os.path.join(base, "sub", "a.txt"))


@pytest.mark.parametrize("rel", ["../outside.txt", "sub/../../outside.txt"])
def test_inside_folder_refuses_a_path_that_climbs_out(tmp_path: Path, rel: str) -> None:
    assert inside_folder(os.path.realpath(tmp_path), rel) is None


def test_inside_folder_refuses_an_absolute_path_elsewhere(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    other = tmp_path / "other.txt"
    assert inside_folder(os.path.realpath(base), str(other)) is None


def test_inside_folder_follows_a_link_out_of_the_folder(tmp_path: Path) -> None:
    base, elsewhere = tmp_path / "base", tmp_path / "elsewhere"
    base.mkdir()
    elsewhere.mkdir()
    try:
        (base / "link").symlink_to(elsewhere, target_is_directory=True)
    except OSError:
        pytest.skip("this system does not let the test make a link")
    assert inside_folder(os.path.realpath(base), "link/a.txt") is None


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows has paths with no common path (other drives)")
def test_inside_folder_refuses_a_path_on_another_drive(tmp_path: Path) -> None:
    base = os.path.realpath(tmp_path)
    other_drive = "Z:\\" if not base.upper().startswith("Z") else "Y:\\"
    assert inside_folder(base, other_drive + "x.txt") is None


def test_the_inside_this_folder_check_is_written_once() -> None:
    """The archive writer and the raw manifest check both go through `inside_folder`."""
    package = REPO_ROOT / "src" / "punct_restore"
    writers = sorted(path.relative_to(package).as_posix() for path in package.rglob("*.py")
                     if "commonpath" in path.read_text(encoding="utf-8"))
    assert writers == ["file_io.py"]


# ---------------------------------------------------------------- iter_jsonl and read_json options
def test_iter_jsonl_can_refuse_a_blank_line_and_names_it(tmp_path: Path) -> None:
    path = tmp_path / "f.jsonl"
    path.write_bytes(b'{"a": 1}\n\n{"a": 2}\n')
    assert [number for number, _ in iter_jsonl(path, parse_json_object)] == [1, 3]  # the default skips it
    with pytest.raises(ValueError) as caught:
        list(iter_jsonl(path, parse_json_object, skip_blank=False))
    assert str(caught.value) == "f.jsonl line 2: blank JSONL record"


def test_iter_jsonl_that_refuses_blank_lines_also_refuses_one_at_the_end(tmp_path: Path) -> None:
    path = tmp_path / "f.jsonl"
    path.write_bytes(b'{"a": 1}\n\n')
    with pytest.raises(ValueError, match=r"^f\.jsonl line 2: blank JSONL record$"):
        list(iter_jsonl(path, parse_json_object, skip_blank=False))


def test_iter_jsonl_that_refuses_blank_lines_takes_a_final_line_without_a_newline(tmp_path: Path) -> None:
    path = tmp_path / "f.jsonl"
    path.write_bytes(b'{"a": 1}\n{"a": 2}')
    assert [record for _, record in iter_jsonl(path, parse_json_object, skip_blank=False)] == [{"a": 1}, {"a": 2}]


def test_iter_jsonl_reads_utf8_with_a_byte_order_mark_only_when_asked(tmp_path: Path) -> None:
    path = tmp_path / "f.jsonl"
    path.write_bytes(BOM.encode("utf-8") + b'{"a": 1}\n{"a": 2}\n')
    with pytest.raises(ValueError, match=r"^f\.jsonl line 1: not valid JSON$"):
        list(iter_jsonl(path, parse_json_object))
    assert [record for _, record in iter_jsonl(path, parse_json_object, encoding="utf-8-sig")] == [{"a": 1}, {"a": 2}]


def test_iter_jsonl_names_the_file_for_bytes_that_are_not_text_in_the_asked_encoding(tmp_path: Path) -> None:
    path = tmp_path / "f.jsonl"
    path.write_bytes(b'{"a": "\xff"}\n')
    with pytest.raises(ValueError, match=r"^f\.jsonl: not valid UTF-8$"):
        list(iter_jsonl(path, parse_json_object, encoding="utf-8-sig"))


def test_read_json_can_drop_a_byte_order_mark_and_does_not_by_default(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_bytes(BOM.encode("utf-8") + b'{"a": 1}')
    with pytest.raises(ValueError, match=r"^m\.json: not valid JSON$"):
        read_json(path)
    assert read_json(path, encoding="utf-8-sig") == {"a": 1}
