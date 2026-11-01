"""Old library names whose function was deleted or changed keep main's (72ebfd1) behaviour (spec section 2).

Each wrapper is a thin layer over the new code; these tests pin what the old callers relied on: the return type, what
is skipped, what is refused and, for `selected_records`, how a caller-supplied `counts` dict is filled.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from test_hmm_cli import jsonl

from punct_restore.hashing import sha256_file
from punct_restore.models.hmm_records import SelectedRecords
from punct_restore.models.hmm_workflow import read_id_selection, selected_records, sha256
from punct_restore.prediction_io import read_frozen_ids

COUNT_KEYS = ["input_records", "selected_records", "selected_tokens"]


def _ids(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "list.ids"
    path.write_text(text, encoding="utf-8")
    return path


def test_read_frozen_ids_skips_blank_lines_strips_each_id_and_returns_a_list(tmp_path: Path) -> None:
    result = read_frozen_ids(_ids(tmp_path, "a\n\n  b  \n   \nc\n"))
    assert result == ["a", "b", "c"]
    assert type(result) is list


def test_read_frozen_ids_raises_file_not_found_for_a_missing_file_and_for_a_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="^Frozen ID list not found: gone.ids$"):
        read_frozen_ids(tmp_path / "gone.ids")
    with pytest.raises(FileNotFoundError, match=f"^Frozen ID list not found: {tmp_path.name}$"):
        read_frozen_ids(tmp_path)


def test_read_frozen_ids_refuses_a_duplicate_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate ID 'a'"):
        read_frozen_ids(_ids(tmp_path, "a\nb\na\n"))


@pytest.mark.parametrize("text", ["", "\n  \n"], ids=["empty", "blank-lines-only"])
def test_read_frozen_ids_refuses_a_list_without_any_id(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        read_frozen_ids(_ids(tmp_path, text))


def test_read_id_selection_returns_a_tuple_in_file_order(tmp_path: Path) -> None:
    result = read_id_selection(_ids(tmp_path, "b\na\n"))
    assert result == ("b", "a")
    assert type(result) is tuple


@pytest.mark.parametrize(
    "text",
    ["a\n\nb\n", " a\nb\n", "a\nb \n", "a\na\n", "", "\n"],
    ids=["blank-line", "leading-space", "trailing-space", "duplicate", "empty", "one-blank-line"],
)
def test_read_id_selection_refuses_what_main_refused(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValueError):
        read_id_selection(_ids(tmp_path, text))


def test_sha256_is_the_hex_sha256_of_the_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc" * 1000)
    assert sha256(path) == hashlib.sha256(b"abc" * 1000).hexdigest() == sha256_file(path)


def _row(sample_id: str, tokens: list[str]) -> dict[str, Any]:
    return {"schema_version": 1, "sample_id": sample_id, "tokens": tokens, "punct_labels": ["O"] * len(tokens)}


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "in.jsonl"
    jsonl(path, [_row("a", ["x", "y"]), _row("b", ["z"]), _row("c", ["u", "v", "w"]), _row("d", ["q"])])
    return path


def test_selected_records_yields_the_records_of_selected_records_class_in_file_order(tmp_path: Path) -> None:
    source = _source(tmp_path)
    ids = ("c", "a")
    assert list(selected_records(source, ids)) == list(SelectedRecords(source, ids))
    assert [r["sample_id"] for r in selected_records(source, ids)] == ["a", "c"]
    assert list(selected_records(source, ids))[0] == {"schema_version": 1, "sample_id": "a", "tokens": ["x", "y"]}
    assert len(list(selected_records(source, None))) == 4


def test_selected_records_fills_counts_after_the_iteration_like_main(tmp_path: Path) -> None:
    counts: dict[str, int] = {}
    list(selected_records(_source(tmp_path), ("a", "c"), counts=counts))
    assert counts == {"input_records": 4, "selected_records": 2, "selected_tokens": 5}
    assert list(counts) == COUNT_KEYS


def test_selected_records_counts_stay_untouched_until_the_first_row_is_asked_for(tmp_path: Path) -> None:
    counts = {"input_records": 99, "selected_records": 99, "selected_tokens": 99}
    rows = selected_records(_source(tmp_path), None, counts=counts)
    assert counts == {"input_records": 99, "selected_records": 99, "selected_tokens": 99}
    next(rows)
    assert counts == dict.fromkeys(COUNT_KEYS, 0)  # reset at the first next()
    assert list(counts) == COUNT_KEYS
    list(rows)
    assert counts == {"input_records": 4, "selected_records": 4, "selected_tokens": 7}


def test_selected_records_need_no_counts_argument(tmp_path: Path) -> None:
    assert len(list(selected_records(_source(tmp_path), ("b",)))) == 1


def test_selected_records_still_refuse_a_missing_selected_id_after_the_last_row(tmp_path: Path) -> None:
    counts: dict[str, int] = {}
    with pytest.raises(ValueError, match="1 selection IDs missing"):
        list(selected_records(_source(tmp_path), ("a", "gone"), counts=counts))
    assert counts == dict.fromkeys(COUNT_KEYS, 0)  # documented: no partial totals after an error


def test_selected_records_counts_stay_zero_after_an_early_stop(tmp_path: Path) -> None:
    counts: dict[str, int] = {}
    for _ in selected_records(_source(tmp_path), None, counts=counts):
        break
    assert counts == dict.fromkeys(COUNT_KEYS, 0)  # documented: no partial totals when the caller stops early


def _training_row(sample_id: str, source_revision: str) -> dict[str, Any]:
    meta = {"policy_version": "p1", "source_revision": source_revision, "annotation_view": "canonical"}
    return {"schema_version": 1, "sample_id": sample_id, "source": "unit", "split": "train", "label_mode": "A",
            "tokens": ["hello", "nasa"], "punct_labels": ["O", "PERIOD"], "case_labels": ["FIRST_UPPER", "ALL_UPPER"],
            "metadata": meta}


def test_selected_records_pass_the_training_arguments_on(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    jsonl(source, [_training_row("t1", "r1")])
    wanted = list(selected_records(source, None, training_mode="A", policy_version="p1", source_revision="r1"))
    assert wanted == [_training_row("t1", "r1")]  # training keeps the whole record
    with pytest.raises(ValueError, match="train/B"):
        list(selected_records(source, None, training_mode="B"))
    with pytest.raises(ValueError, match="canonical release policy"):
        list(selected_records(source, None, training_mode="A", policy_version="p1", source_revision="other"))
