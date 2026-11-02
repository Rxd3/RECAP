"""Round 3, item 7: one record-selection reader (`prediction_io.select_records`) for every command that keeps the
records of an ID list from a JSONL file: the listed records in file order, a repeated sample_id refused, every listed
ID found, and one wording for each problem."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from punct_restore.cli import build_word_clusters, run_ablations
from punct_restore.file_io import iter_jsonl
from punct_restore.models.crf_training_data import load_and_filter_training_records
from punct_restore.models.threshold import stream_train_labels
from punct_restore.prediction_io import parse_sample_record, select_observations, select_records


def _rows(*ids: str) -> Iterator[tuple[int, dict[str, Any]]]:
    """(line number, record) for each id, numbered from 1, as `iter_jsonl` gives them."""
    return iter(enumerate(({"sample_id": sample_id} for sample_id in ids), 1))


def _keep(record: dict[str, Any], where: str) -> tuple[str, str]:
    return record["sample_id"], where


# ---------------------------------------------------------------- the reader itself
def test_it_keeps_the_listed_records_in_file_order_not_in_list_order() -> None:
    kept = list(select_records(_rows("a", "b", "c", "d"), "f.jsonl", ["d", "b"], _keep))
    assert kept == [("b", "f.jsonl line 2"), ("d", "f.jsonl line 4")]


def test_it_keeps_every_record_without_a_list() -> None:
    assert [sample_id for sample_id, _ in select_records(_rows("b", "a"), "f.jsonl", None, _keep)] == ["b", "a"]


def test_what_keep_returns_is_what_comes_out() -> None:
    kept = select_records(_rows("a", "b"), "f.jsonl", ["b"], lambda record, where: len(where))
    assert list(kept) == [len("f.jsonl line 2")]


def test_the_name_in_every_message_is_the_one_given_not_the_file_name() -> None:
    with pytest.raises(ValueError, match=r"^mode_A/val\.jsonl line 2: duplicate sample_id 'a'$"):
        list(select_records(_rows("a", "a"), "mode_A/val.jsonl", None, _keep))
    with pytest.raises(ValueError, match=r"^mode_A/val\.jsonl: 1 listed ID\(s\) are missing, first 'z'$"):
        list(select_records(_rows("a"), "mode_A/val.jsonl", ["z"], _keep))


@pytest.mark.parametrize("ids", [None, ["a"], ["b"], ["a", "b"]])
def test_a_repeated_sample_id_is_refused_whether_or_not_it_is_listed(ids: list[str] | None) -> None:
    with pytest.raises(ValueError) as caught:
        list(select_records(_rows("a", "b", "a"), "f.jsonl", ids, _keep))
    assert str(caught.value) == "f.jsonl line 3: duplicate sample_id 'a'"


def test_it_says_how_many_listed_ids_are_missing_and_names_the_first_in_list_order() -> None:
    with pytest.raises(ValueError) as caught:
        list(select_records(_rows("a"), "f.jsonl", ["z", "a", "y", "b"], _keep))
    assert str(caught.value) == "f.jsonl: 3 listed ID(s) are missing, first 'z'"


def test_it_reads_one_record_at_a_time_and_checks_the_list_after_the_last_one() -> None:
    reader = select_records(_rows("a", "b"), "f.jsonl", ["a", "z"], _keep)
    assert next(reader) == ("a", "f.jsonl line 1")  # the first record is out before the file is finished
    with pytest.raises(ValueError, match="1 listed ID"):
        next(reader)


def test_a_problem_found_by_keep_stops_the_reading_at_that_record() -> None:
    read: list[str] = []

    def rows() -> Iterator[tuple[int, dict[str, Any]]]:
        for number, record in _rows("a", "b", "c"):
            read.append(record["sample_id"])
            yield number, record

    def keep(record: dict[str, Any], where: str) -> None:
        if record["sample_id"] == "b":
            raise ValueError(f"{where}: bad")

    with pytest.raises(ValueError, match=r"^f\.jsonl line 2: bad$"):
        list(select_records(rows(), "f.jsonl", None, keep))
    assert read == ["a", "b"]


def test_an_empty_list_keeps_nothing_and_finds_nothing_missing() -> None:
    assert list(select_records(_rows("a"), "f.jsonl", [], _keep)) == []


# ---------------------------------------------------------------- the commands that use it
def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_select_observations_goes_through_the_reader_and_refuses_an_unlisted_repeat(tmp_path: Path) -> None:
    rows = [{"sample_id": s, "tokens": ["w"]} for s in ("a", "b", "a")]
    with pytest.raises(ValueError, match=r"^in\.jsonl line 3: duplicate sample_id 'a'$"):
        select_observations(_write(tmp_path / "in.jsonl", rows), ["b"])


def _train_row(sample_id: str) -> dict[str, Any]:
    return {"sample_id": sample_id, "split": "train", "label_mode": "A", "tokens": ["hello", "world"],
            "punct_labels": ["O", "PERIOD"]}


def test_training_records_refuse_an_unlisted_repeat_and_name_the_missing_id_alike(tmp_path: Path) -> None:
    data = _write(tmp_path / "data.jsonl", [_train_row("a"), _train_row("b"), _train_row("b")])
    ids = tmp_path / "ids.txt"
    ids.write_text("a\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"^data\.jsonl line 3: duplicate sample_id 'b'$"):
        load_and_filter_training_records(data, ids, "A")
    ids.write_text("a\nz\n", encoding="utf-8")
    _write(data, [_train_row("a")])
    with pytest.raises(ValueError, match=r"^data\.jsonl: 1 listed ID\(s\) are missing, first 'z'$"):
        load_and_filter_training_records(data, ids, "A")


def test_threshold_priors_refuse_an_unlisted_repeat(tmp_path: Path) -> None:
    train = _write(tmp_path / "train.jsonl", [_train_row("a"), _train_row("b"), _train_row("b")])
    with pytest.raises(ValueError, match=r"^train\.jsonl line 3: duplicate sample_id 'b'$"):
        list(stream_train_labels(train, ["a"], "A"))


def test_the_ablation_reader_names_a_missing_id_in_the_shared_wording(tmp_path: Path) -> None:
    path = _write(tmp_path / "train.jsonl", [_train_row("a")])
    with pytest.raises(ValueError, match=r"^train\.jsonl: 2 listed ID\(s\) are missing, first 'y'$"):
        run_ablations.read_records(path, ["y", "a", "z"])
    assert list(run_ablations.read_records(path, ["a"])) == ["a"]


def test_the_cluster_reader_names_a_missing_id_in_the_shared_wording(tmp_path: Path) -> None:
    path = _write(tmp_path / "train.jsonl", [_train_row("a")])
    assert list(build_word_clusters.train_tokens(path, ["a"])) == [["hello", "world"]]
    with pytest.raises(ValueError, match=r"^train\.jsonl: 1 listed ID\(s\) are missing, first 'z'$"):
        list(build_word_clusters.train_tokens(path, ["a", "z"]))


def test_iter_jsonl_and_parse_sample_record_feed_the_reader_as_the_commands_use_it(tmp_path: Path) -> None:
    path = _write(tmp_path / "f.jsonl", [{"sample_id": "a"}, {"sample_id": "b"}])
    kept = select_records(iter_jsonl(path, parse_sample_record), path.name, ["b"], lambda record, where: where)
    assert list(kept) == ["f.jsonl line 2"]
