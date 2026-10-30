"""Hardening tests for punct_restore.evaluation.inputs: strict UTF-8, BOM, group_id and read-only identities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from eval_fixtures import (BAD_UTF8_LINE, BOM, FILE_NAMES, SCORED, Case, corrupt_line_two, gold_rows, perfect_preds,
                           valid_case, write_case)

from punct_restore.evaluation import evaluate
from punct_restore.evaluation.inputs import (LABEL_SETS_PATH, Bundle, EvaluationInputError, Vocab, load_bundle,
                                              load_vocab)


def _folder(parent: Path, name: str) -> Path:
    folder = parent / name
    folder.mkdir()
    return folder


def _load(case: Case, vocab: Vocab) -> Bundle:
    return load_bundle(case.gold, case.ids, case.predictions, case.masks, vocab)


def _load_error(case: Case, vocab: Vocab) -> str:
    with pytest.raises(EvaluationInputError) as info:
        _load(case, vocab)
    return str(info.value)


def _gold_with_metadata(index: int, metadata: Any) -> list[dict[str, Any]]:
    """Valid mode A gold rows where the row at `index` carries the given metadata."""
    return [{**row, "metadata": metadata} if position == index else row
            for position, row in enumerate(gold_rows("A"))]


def _case_with_gold(folder: Path, gold: list[dict[str, Any]]) -> Case:
    return write_case(folder, gold, SCORED, perfect_preds("A"))


# ---------------------------------------------------------------- strict UTF-8
@pytest.mark.parametrize("target", FILE_NAMES)
def test_non_utf8_input_names_file_and_line(tmp_path: Path, vocab: Vocab, target: str) -> None:
    case = valid_case(tmp_path, "C")
    corrupt_line_two(getattr(case, target))
    assert _load_error(case, vocab) == f"{FILE_NAMES[target]} line 2: not UTF-8 text"


def test_non_utf8_first_line_is_line_1(tmp_path: Path, vocab: Vocab) -> None:
    case = valid_case(tmp_path, "C")
    case.gold.write_bytes(BAD_UTF8_LINE + case.gold.read_bytes())
    assert _load_error(case, vocab) == "gold.jsonl line 1: not UTF-8 text"


def test_latin1_gold_record_is_rejected_not_decoded(tmp_path: Path, vocab: Vocab) -> None:
    case = valid_case(tmp_path, "C")
    first, *rest = gold_rows("C")
    latin1_first = {**first, "tokens": ["what", "is", "caf\xe9"]}
    lines = [json.dumps(latin1_first, ensure_ascii=False).encode("latin-1")]
    lines += [json.dumps(row).encode("utf-8") for row in rest]
    case.gold.write_bytes(b"\n".join(lines) + b"\n")
    assert _load_error(case, vocab) == "gold.jsonl line 1: not UTF-8 text"


@pytest.mark.parametrize("target", FILE_NAMES)
def test_bom_on_line_1_still_loads(tmp_path: Path, vocab: Vocab, target: str) -> None:
    clean = _load(valid_case(_folder(tmp_path, "clean"), "C"), vocab)
    bom_case = valid_case(_folder(tmp_path, "bom"), "C")
    path = getattr(bom_case, target)
    path.write_bytes(BOM + path.read_bytes())
    bom = _load(bom_case, vocab)
    assert bom.label_mode == clean.label_mode
    assert bom.model_name == clean.model_name
    assert bom.records == clean.records
    assert bom.ignored_prediction_ids == clean.ignored_prediction_ids


@pytest.mark.parametrize(("keep_real_config", "tail"),
                         [(False, b"punctuation:\n  standard: [O, \xe9\xff]\n"),
                          (True, b"# caf\xe9\n")],
                         ids=["bad_bytes_in_list", "latin1_comment_in_real_config"])
def test_load_vocab_rejects_non_utf8_yaml(tmp_path: Path, keep_real_config: bool, tail: bytes) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_bytes((LABEL_SETS_PATH.read_bytes() if keep_real_config else b"") + tail)
    with pytest.raises(EvaluationInputError) as info:
        load_vocab(bad)
    assert str(info.value) == "bad.yaml: not a valid label configuration"


# ---------------------------------------------------------------- metadata.group_id
@pytest.mark.parametrize("group_id", [7, ["g1"], "", True], ids=["int", "list", "empty_string", "bool"])
def test_invalid_group_id_names_the_line(tmp_path: Path, vocab: Vocab, group_id: Any) -> None:
    case = _case_with_gold(tmp_path, _gold_with_metadata(1, {"group_id": group_id}))
    with pytest.raises(EvaluationInputError, match=r"gold\.jsonl line 2: .*group_id"):
        _load(case, vocab)


def test_group_id_rule_applies_to_unscored_records(tmp_path: Path, vocab: Vocab) -> None:
    case = _case_with_gold(tmp_path, _gold_with_metadata(2, {"group_id": 7}))  # s3 is not in list.ids
    with pytest.raises(EvaluationInputError, match=r"gold\.jsonl line 3: .*group_id"):
        _load(case, vocab)


@pytest.mark.parametrize(("metadata", "expected"),
                         [({"group_id": "talk-7"}, "talk-7"), ({"group_id": None}, None), ({"other": 1}, None)],
                         ids=["string_kept", "null_is_none", "no_group_id_key"])
def test_group_id_string_null_or_absent(tmp_path: Path, vocab: Vocab, metadata: dict[str, Any],
                                        expected: str | None) -> None:
    bundle = _load(_case_with_gold(tmp_path, _gold_with_metadata(0, metadata)), vocab)
    assert bundle.records[0].sample_id == "s1"
    assert bundle.records[0].group_id == expected


# ---------------------------------------------------------------- read-only file identities
@pytest.mark.parametrize("mode", ["A", "C"])
def test_bundle_inputs_are_read_only(tmp_path: Path, vocab: Vocab, mode: str) -> None:
    bundle = _load(valid_case(tmp_path, mode), vocab)
    original_path = bundle.inputs["gold"]["path"]
    with pytest.raises(TypeError):
        bundle.inputs["gold"] = {}
    with pytest.raises(TypeError):
        bundle.inputs["gold"]["path"] = "x"
    with pytest.raises(TypeError):
        del bundle.inputs["ids"]
    if mode == "C":
        with pytest.raises(TypeError):
            bundle.inputs["masks"]["sha256"] = "x"
    assert bundle.inputs["gold"]["path"] == original_path


@pytest.mark.parametrize("mode", ["A", "C"])
def test_report_inputs_are_plain_serialisable_dicts(tmp_path: Path, vocab: Vocab, mode: str) -> None:
    bundle = _load(valid_case(tmp_path, mode), vocab)
    inputs = evaluate(bundle, vocab)["inputs"]
    assert type(inputs["gold"]) is dict
    assert json.loads(json.dumps(inputs)) == inputs
