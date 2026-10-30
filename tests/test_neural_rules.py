"""Alignment, protocol and optional-dependency rules; no downloads or GPU required."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from punct_restore.models.neural_tagger import BertTagger, first_subword_labels
from punct_restore.models.neural_utils import observations
from punct_restore.models.neural_workflow import choose, frozen_selection, training_records
from punct_restore.models.seq2seq import T5Seq2Seq, align_generated, output_words, target_text, word_case

ROOT = Path(__file__).resolve().parents[1]


def test_first_piece_only_and_partial_word_mask():
    ids = [None, 0, 1, 1, 2, None]
    assert first_subword_labels(ids, [3, 2, 1]) == [-100, 3, 2, -100, 1, -100]
    assert first_subword_labels(ids, [3, 2, 1], {0, 1}) == [-100, 3, 2, -100, -100, -100]


@pytest.mark.parametrize("tokens,text,mode,expected,missing", [
    (["hello", "lost", "world"], "hello world.", "A", ["O", "O", "PERIOD"], 1),
    (["hello", "world"], "hello inserted world?", "A", ["O", "QUESTION"], 0),
    (["hello", "world"], "changed world!", "B", ["O", "EXCLAMATION"], 1),
    (["hello", "world"], "HELLO, World?", "C", ["COMMA|ALL_UPPER", "QUESTION|FIRST_UPPER"], 0),
    (["hello", "world"], "hello - world.", "B", ["DASH", "PERIOD"], 0),
    (["a", "b", "c", "d"], "a! b: c; d -", "A", ["PERIOD", "COMMA", "PERIOD", "COMMA"], 0),
    (["well-known", "word"], "well-known word-", "B", ["O", "O"], 1),
    (["a"], "a?!", "B", ["EXCLAMATION"], 0),
    (["hello"], "", "C", ["O|LOWER"], 1),
    (["a", "b"], "a new b", "C", ["O|LOWER", "O|LOWER"], 0),
    (["a", "b", "a"], "a a!", "A", ["O", "O", "PERIOD"], 1),
])
def test_exact_seq2seq_alignment(tokens, text, mode, expected, missing):
    assert align_generated(tokens, text, mode) == (expected, missing)


def test_dash_and_last_mark_parsing():
    assert output_words("- word, - next?! last-word") == [
        ("word", "DASH"), ("next", "EXCLAMATION"), ("last-word", "O")]
    assert output_words("hello , world") == [("hello", "COMMA"), ("world", "O")]


@pytest.mark.parametrize("word,expected", [("NASA", "ALL_UPPER"), ("Hello", "FIRST_UPPER"),
    ("iPhone", "LOWER"), ("I", "FIRST_UPPER"), ("A1", "ALL_UPPER"), ("42", "LOWER")])
def test_generated_case(word, expected):
    assert word_case(word) == expected


def test_target_uses_shared_reconstruction():
    record = {"tokens": ["hello", "nasa"], "punct_labels": ["COMMA", "PERIOD"],
              "case_labels": ["FIRST_UPPER", "ALL_UPPER"]}
    assert target_text(record, "A") == "hello, nasa."
    assert target_text(record, "B") == "hello, nasa."
    assert target_text(record, "C") == "Hello, NASA."


def test_observations_strip_gold_and_check_ids():
    record = {"sample_id": "one", "tokens": ["hello"], "punct_labels": ["QUESTION"], "text": "secret"}
    assert observations([record]) == [{"sample_id": "one", "tokens": ["hello"]}]
    with pytest.raises(ValueError, match="unique"):
        observations([record, record])
    with pytest.raises(ValueError, match="lowercase"):
        observations([{**record, "tokens": ["Hello"]}])


def test_training_filters_exact_ids_and_rejects_wrong_split(tmp_path):
    source, ids = tmp_path / "train.jsonl", tmp_path / "train.ids"
    good = {"sample_id": "one", "tokens": ["hello"], "punct_labels": ["PERIOD"],
            "split": "train", "label_mode": "A"}
    source.write_text(json.dumps(good) + "\n" + json.dumps({**good, "sample_id": "excluded"}), encoding="utf-8")
    ids.write_text("one\n", encoding="utf-8")
    assert [r["sample_id"] for r in training_records(source, ids, "A")] == ["one"]
    source.write_text(json.dumps({**good, "split": "test"}), encoding="utf-8")
    with pytest.raises(ValueError, match="split train"):
        training_records(source, ids, "A")
    ids.write_text("missing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing frozen IDs"):
        training_records(source, ids, "A")


def test_selection_uses_official_primary_and_stable_ties():
    candidates = [{"primary": {"value": score}, "epoch": epoch, "learning_rate": rate}
                  for score, epoch, rate in [(0.4, 1, 2e-5), (0.6, 3, 2e-5),
                                             (0.6, 2, 5e-5), (0.6, 2, 2e-5)]]
    assert choose(candidates) is candidates[-1]


def test_unfrozen_selection_rejected_before_input_read(tmp_path):
    from punct_restore.models.neural_tagger import BertTagger
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"frozen": False, "model_name": "bert_tagger"}), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen selection"):
        frozen_selection(path, tmp_path / "nonexistent-model", BertTagger)


@pytest.mark.parametrize("script", ["train_neural_tagger.py", "train_seq2seq.py"])
def test_cli_help_without_downloading(script):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "train", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--manifest-out" in result.stdout
    assert "--selection-out" in result.stdout


def test_neural_extra_is_optional_and_exactly_pinned():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert all("==" in item for item in project["optional-dependencies"]["neural"])
    assert not any(item.startswith(("torch", "transformers")) for item in project["dependencies"])


@pytest.mark.parametrize("model_class", [BertTagger, T5Seq2Seq])
def test_uninitialized_artifact_refuses_save(model_class, tmp_path):
    with pytest.raises(RuntimeError, match="not initialized"):
        model_class("A").save_model(tmp_path / "model")
    assert not (tmp_path / "model").exists()
