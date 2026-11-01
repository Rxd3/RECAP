"""release.read_titles: the dated metadata snapshot is read line by line; a bad line is one ValueError naming it."""

from __future__ import annotations

from pathlib import Path

import pytest

from punct_restore.data import release

OK = '{"video_id": "aaaaaaaaaaa", "status": 200, "title": "A talk"}\n'
FOLDER = "secretfolder"


def _titles(tmp_path: Path, raw: bytes | str) -> Path:
    folder = tmp_path / FOLDER
    folder.mkdir()
    path = folder / "titles.jsonl"
    path.write_bytes(raw if isinstance(raw, bytes) else raw.encode("utf-8"))
    return path


def test_status_200_rows_give_their_titles_and_other_rows_are_skipped(tmp_path: Path) -> None:
    rows = (OK + '{"video_id": "bbbbbbbbbbb", "status": 200}\n'
            '{"video_id": "ccccccccccc", "status": 200, "title": null}\n'
            '{"video_id": "ddddddddddd", "status": 404, "title": "gone"}\n{"status": 404}\n')
    assert release.read_titles(_titles(tmp_path, rows)) == {"aaaaaaaaaaa": "A talk", "bbbbbbbbbbb": "",
                                                            "ccccccccccc": ""}


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (OK + "[1]\n", "titles.jsonl line 2: not a JSON object"),
        (OK + "{broken\n", "titles.jsonl line 2: not valid JSON"),
        ("[" * 100_000 + "\n", "titles.jsonl line 1: not valid JSON"),
        ('{"status": 200}\n', "titles.jsonl line 1: video_id must be a string"),
        (OK + '{"status": 200, "video_id": 5}\n', "titles.jsonl line 2: video_id must be a string"),
        (b"\xe9\xff not utf-8\n", "titles.jsonl: not valid UTF-8"),
    ],
    ids=["array", "broken", "nested", "no-video-id", "number-video-id", "not-utf8"],
)
def test_a_bad_line_is_one_value_error_naming_the_file_and_line_only(
        tmp_path: Path, raw: bytes | str, message: str) -> None:
    path = _titles(tmp_path, raw)
    with pytest.raises(ValueError) as problem:
        release.read_titles(path)
    assert str(problem.value) == message
