"""The record selection reports what it read in its result, and its checks never change what they are given."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_hmm_cli import jsonl

from punct_restore.models.hmm_records import SelectedRecords, _check_record_basics


def _row(sample_id: str, tokens: list[str] | None = None) -> dict[str, Any]:
    return {"schema_version": 1, "sample_id": sample_id, "tokens": tokens or ["hi"]}


def test_counts_cover_every_row_read_including_unselected_rows_after_the_last_selected_one(tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    jsonl(source, [_row("first", ["a", "b"]), _row("skip"), _row("last"), _row("trailing")])
    selection = SelectedRecords(source, ("first", "last"))
    assert [r["sample_id"] for r in selection] == ["first", "last"]
    assert selection.counts == {"input_records": 4, "selected_records": 2, "selected_tokens": 3}
    assert list(selection.counts) == ["input_records", "selected_records", "selected_tokens"]


def test_iterating_again_gives_the_same_counts_not_doubled_ones(tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    jsonl(source, [_row("a"), _row("b")])
    selection = SelectedRecords(source, None)
    list(selection)
    list(selection)
    assert selection.counts == {"input_records": 2, "selected_records": 2, "selected_tokens": 2}


def test_a_selected_id_that_is_missing_is_still_refused_after_the_last_row(tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    jsonl(source, [_row("a")])
    with pytest.raises(ValueError, match="1 selection IDs missing"):
        list(SelectedRecords(source, ("a", "gone")))


def test_the_basics_check_returns_the_id_and_tokens_and_leaves_the_seen_set_alone(tmp_path: Path) -> None:
    seen = {"earlier"}
    assert _check_record_basics(_row("later", ["x"]), tmp_path / "in.jsonl", seen) == ("later", ["x"])
    assert seen == {"earlier"}


def test_the_basics_check_refuses_an_id_seen_before_and_names_the_file_only(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"^in\.jsonl: invalid or duplicate sample_id 'earlier'$"):
        _check_record_basics(_row("earlier"), tmp_path / "deep" / "in.jsonl", {"earlier"})
