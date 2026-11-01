"""Record-checking helpers return their result and leave the collections they are given alone."""

from __future__ import annotations

import json

import pytest

from punct_restore.cli import train_hidden_event_lm
from punct_restore.models import crf_records

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


def test_a_kept_hidden_event_line_is_returned_slim_and_not_added_to_the_kept_mapping() -> None:
    kept = {"earlier": {"sample_id": "earlier"}}

    record = train_hidden_event_lm._keep_line(json.dumps(ROW), "f.jsonl line 1", {"s1"}, kept, "A")

    assert record == {key: ROW[key] for key in ("sample_id", "split", "label_mode", "tokens", "punct_labels")}
    assert list(kept) == ["earlier"]


def test_an_unlisted_hidden_event_line_is_skipped() -> None:
    assert train_hidden_event_lm._keep_line(json.dumps(ROW), "f.jsonl line 1", {"other"}, {}, "A") is None


def test_a_listed_id_that_was_kept_before_is_refused() -> None:
    with pytest.raises(ValueError, match="f.jsonl line 2: duplicate sample_id 's1'"):
        train_hidden_event_lm._keep_line(json.dumps(ROW), "f.jsonl line 2", {"s1"}, {"s1": {}}, "A")
