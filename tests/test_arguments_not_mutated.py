"""Record-checking helpers return their result and leave the collections they are given alone."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from punct_restore.cli import train_hidden_event_lm
from punct_restore.models import crf_records
from punct_restore.prediction_io import select_records

ROW = {"sample_id": "s1", "split": "train", "label_mode": "A", "tokens": ["a", "b"], "punct_labels": ["O", "PERIOD"],
       "metadata": {"unused": True}}


def test_a_checked_crf_record_is_not_added_to_the_set_of_seen_ids() -> None:
    seen = {"earlier"}

    sample_id, tokens = crf_records._checked_training_record(ROW, "Record at index 0", seen)

    assert (sample_id, tokens) == ("s1", ["a", "b"])
    assert seen == {"earlier"}


def test_a_crf_record_whose_id_was_seen_is_refused() -> None:
    with pytest.raises(ValueError, match="duplicate sample_id 's1'"):
        crf_records._checked_training_record(ROW, "Record at index 1", {"s1"})


def test_a_slimmed_hidden_event_record_is_a_new_object_and_the_given_one_is_left_alone() -> None:
    before = json.dumps(ROW)

    record = train_hidden_event_lm._slim_training_record(ROW, "f.jsonl line 1", "A")

    assert record == {key: ROW[key] for key in ("sample_id", "split", "label_mode", "tokens", "punct_labels")}
    assert record is not ROW and record["tokens"] is not ROW["tokens"] and json.dumps(ROW) == before


def test_the_record_selection_reader_leaves_the_id_list_and_the_records_alone() -> None:
    ids = ["s1", "gone"]
    rows = [(1, dict(ROW)), (2, {"sample_id": "s2"})]
    snapshot = json.dumps(rows)

    with pytest.raises(ValueError, match="1 listed ID"):
        list(select_records(iter(rows), "f.jsonl", ids, lambda record, where: record))

    assert ids == ["s1", "gone"] and json.dumps(rows) == snapshot


def test_an_unlisted_hidden_event_line_is_skipped_and_a_listed_repeat_is_refused(tmp_path: Path) -> None:
    path, ids = tmp_path / "f.jsonl", tmp_path / "f.ids"
    ids.write_text("s1\n", encoding="utf-8")
    path.write_text(json.dumps({**ROW, "sample_id": "other"}) + "\n" + json.dumps(ROW) + "\n", encoding="utf-8")
    assert [r["sample_id"] for r in train_hidden_event_lm.read_training_records(path, ids, "A")] == ["s1"]
    path.write_text(json.dumps(ROW) + "\n" + json.dumps(ROW) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="f.jsonl line 2: duplicate sample_id 's1'"):
        train_hidden_event_lm.read_training_records(path, ids, "A")
