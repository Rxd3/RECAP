"""Tests for punct_restore.evaluation.inputs. Rule tests map a case id to keyword arguments for `_bundle`."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from eval_fixtures import (
    BOM,
    SCORED,
    Case,
    all_masks,
    gold_row,
    gold_rows,
    mask_row,
    perfect_preds,
    pred_row,
    valid_case,
    write_case,
    write_ids,
    write_jsonl,
)

from punct_restore.evaluation.inputs import (
    LABEL_SETS_PATH,
    Bundle,
    EvaluationInputError,
    Vocab,
    file_identity,
    file_identity_with,
    load_bundle,
    load_scored_gold,
    load_vocab,
    sha256_file,
)

_DEL = object()    # in a field change: remove the field
_UNSET = object()  # _bundle default for masks: "what a valid case of this mode has"
FILES = {"gold": "gold.jsonl", "ids": "list.ids", "predictions": "preds.jsonl", "masks": "masks.jsonl"}
A = ("O", "COMMA", "PERIOD", "QUESTION")
B = ("O", "COMMA", "PERIOD", "QUESTION", "EXCLAMATION", "COLON", "SEMICOLON", "DASH")
CASING = ("LOWER", "FIRST_UPPER", "ALL_UPPER")
C = tuple(f"{p}|{c}" for p in A for c in CASING)


def _changed(row: Any, change: Any) -> Any:
    """A str change replaces the row verbatim; a dict sets fields on a copy (value _DEL removes the field)."""
    if isinstance(change, str):
        return change
    return {key: value for key, value in {**row, **change}.items() if value is not _DEL}


def _apply(rows: list[Any], edits: dict[int | str, Any]) -> list[Any]:
    """Copy of rows; edits maps a row index (or "*" for every row) to a verbatim str or a field-change dict."""
    out = list(rows)
    for key, change in edits.items():
        for index in (range(len(out)) if key == "*" else (key,)):
            out[index] = _changed(out[index], change)
    return out


def _case_with(target: str, edits: dict[int | str, Any], mode: str = "A") -> dict[str, Any]:
    """Keyword arguments for `_bundle`: the valid rows of `target` in `mode` with `edits` applied."""
    rows = {"gold": gold_rows, "predictions": perfect_preds, "masks": lambda _: all_masks()}[target](mode)
    return {"mode": mode, target: _apply(rows, edits)}


gold_with = functools.partial(_case_with, "gold")
preds_with = functools.partial(_case_with, "predictions")
masks_with = functools.partial(_case_with, "masks", mode="C")


def _load(case: Case, vocab: Vocab) -> Bundle:
    return load_bundle(case.gold, case.ids, case.predictions, case.masks, vocab)


def _bundle(folder: Path, vocab: Vocab, mode: str = "A", *, gold: Any = None, ids: Sequence[str] = SCORED,
            predictions: Any = None, masks: Any = _UNSET) -> Bundle:
    """Write a case (valid pieces unless overridden; masks=None means no masks file) and load it."""
    if masks is _UNSET:
        masks = all_masks() if mode == "C" else None
    case = write_case(folder, gold_rows(mode) if gold is None else gold, ids,
                      perfect_preds(mode) if predictions is None else predictions, masks)
    return _load(case, vocab)


def _error(call: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    with pytest.raises(EvaluationInputError) as info:
        call(*args, **kwargs)
    return str(info.value)


def _ids(records: Sequence[Any]) -> tuple[str, ...]:
    return tuple(record.sample_id for record in records)


def test_load_vocab_ok(vocab: Vocab) -> None:
    assert (vocab.A, vocab.B, vocab.casing, vocab.C) == (A, B, CASING, C)
    assert len(vocab.C) == 12
    assert (vocab.labels("A"), vocab.labels("B"), vocab.labels("C")) == (A, B, C)


def test_load_vocab_independent_of_cwd(vocab: Vocab, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_vocab() == vocab


def _edited(*path: str, value: Any) -> Callable[[dict[str, Any]], str]:
    """YAML text of the real config with the value at `path` replaced."""
    def build(cfg: dict[str, Any]) -> str:
        functools.reduce(dict.__getitem__, path[:-1], cfg)[path[-1]] = value
        return yaml.safe_dump(cfg)
    return build


BAD_CONFIGS: dict[str, Callable[[dict[str, Any]], str]] = {
    "standard_has_3_labels": _edited("punctuation", "standard", value=list(A[:3])),
    "extended_has_7_labels": _edited("punctuation", "extended", value=list(B[:7])),
    "casing_has_2_labels": _edited("casing", value=list(CASING[:2])),
    "mode_c_has_11_labels": _edited("modes", "C", "labels", value=list(C[:11])),
    "mode_c_casing_major_order": _edited("modes", "C", "labels", value=[f"{p}|{c}" for c in CASING for p in A]),
    "mode_c_slash_separator": _edited("modes", "C", "labels", value=[f"{p}/{c}" for p in A for c in CASING]),
    "casing_key_missing": lambda cfg: yaml.safe_dump({k: v for k, v in cfg.items() if k != "casing"}),
    "whole_file_is_a_list": lambda cfg: yaml.safe_dump([1, 2]),
    "empty_file": lambda cfg: "",
    "invalid_yaml": lambda cfg: "punctuation: [O, COMMA",
}


@pytest.mark.parametrize("build", BAD_CONFIGS.values(), ids=BAD_CONFIGS.keys())
def test_load_vocab_bad_structure_rejected(tmp_path: Path, build: Callable[[dict[str, Any]], str]) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(build(yaml.safe_load(LABEL_SETS_PATH.read_text(encoding="utf-8"))), encoding="utf-8")
    assert "bad.yaml" in _error(load_vocab, bad)


def test_load_vocab_missing_file_propagates_oserror(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_vocab(tmp_path / "nope.yaml")


def test_load_vocab_without_pyyaml_is_a_friendly_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "yaml", None)
    assert "PyYAML" in _error(load_vocab)


def _bom_crlf_copy(case: Case, folder: Path) -> Case:
    """The same inputs rewritten as UTF-8 with a BOM and CRLF line endings, into a new folder."""
    folder.mkdir()

    def convert(source: Path) -> Path:
        target = folder / source.name
        target.write_bytes(BOM + source.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        return target

    return Case(**{name: convert(path) if path else None for name, path in dataclasses.asdict(case).items()})


@pytest.mark.parametrize("mode", ["A", "C"])
def test_gold_with_bom_loads(tmp_path: Path, vocab: Vocab, mode: str) -> None:
    clean_case = valid_case(tmp_path, mode)
    clean = _load(clean_case, vocab)
    bom = _load(_bom_crlf_copy(clean_case, tmp_path / "bom"), vocab)
    assert len(clean.records) == 2
    assert dataclasses.replace(bom, inputs={}) == dataclasses.replace(clean, inputs={})  # all but the file identities


@pytest.mark.parametrize("prefix", [b"", BOM], ids=["plain", "bom"])
def test_ids_with_crlf_loads(tmp_path: Path, vocab: Vocab, prefix: bytes) -> None:
    case = valid_case(tmp_path, "A")
    case.ids.write_bytes(prefix + b"s2\r\n\r\ns1\r\n")
    assert _ids(_load(case, vocab).records) == ("s2", "s1")


def test_ids_order_kept(tmp_path: Path, vocab: Vocab) -> None:
    case = valid_case(tmp_path, "A")
    write_ids(case.ids, ["s2", "s1"])
    assert _ids(_load(case, vocab).records) == ("s2", "s1")
    mode, records = load_scored_gold(case.gold, case.ids, vocab)
    assert (mode, _ids(records)) == ("A", ("s2", "s1"))


EMPTY_FILES = {
    "gold_zero_bytes": ("gold", b"", "A"),
    "gold_blank_lines": ("gold", b"\n  \n\r\n", "A"),
    "gold_bom_only": ("gold", BOM, "A"),
    "predictions_zero_bytes": ("predictions", b"", "A"),
    "masks_zero_bytes": ("masks", b"", "C"),
    "ids_zero_bytes": ("ids", b"", "A"),
    "ids_blank_lines": ("ids", b"\n \n", "A"),
}


@pytest.mark.parametrize(("target", "content", "mode"), EMPTY_FILES.values(), ids=EMPTY_FILES.keys())
def test_empty_files_rejected(tmp_path: Path, vocab: Vocab, target: str, content: bytes, mode: str) -> None:
    case = valid_case(tmp_path, mode)
    getattr(case, target).write_bytes(content)
    message = _error(_load, case, vocab)
    assert FILES[target] in message
    assert "empty" in message


WRONG_SHAPES = {
    **{f"gold_json_{name}": gold_with({1: line}) for name, line in
       {"array": "[1, 2]", "string": '"text"', "null": "null", "number": "42"}.items()},
    "gold_invalid_json": gold_with({1: '{"sample_id": '}),
    "gold_tokens_string": gold_with({0: {"tokens": "what is it"}}),
    "gold_tokens_non_string_item": gold_with({0: {"tokens": ["what", 1, "it"]}}),
    "gold_punct_labels_string": gold_with({0: {"punct_labels": "O"}}),
    "gold_case_labels_dict": gold_with({0: {"case_labels": {"x": 1}}}),
    "gold_punct_labels_null_item": gold_with({0: {"punct_labels": ["O", None, "QUESTION"]}}),
    "gold_sample_id_int": gold_with({0: {"sample_id": 5}}),
    "gold_sample_id_empty": gold_with({0: {"sample_id": ""}}),
    "pred_json_array": preds_with({0: "[1]"}),
    "pred_invalid_json": preds_with({0: '{"sample_id": '}),
    "pred_labels_string": preds_with({0: {"predicted_labels": "O O QUESTION"}}),
    "pred_labels_ints": preds_with({0: {"predicted_labels": [0, 0, 1]}}),
    "pred_model_name_int": preds_with({0: {"model_name": 7}}),
    "pred_model_name_empty": preds_with({0: {"model_name": ""}}),
    "pred_sample_id_null": preds_with({0: {"sample_id": None}}),
    **{f"mask_positions_{name}": masks_with({0: {"excluded_positions": value}}) for name, value in
       {"string": "1", "float": [1.0], "bool": [True], "null": None}.items()},
    "mask_line_empty_array": masks_with({0: "[]"}),
}


@pytest.mark.parametrize("kwargs", WRONG_SHAPES.values(), ids=WRONG_SHAPES.keys())
def test_wrong_json_shapes_rejected(tmp_path: Path, vocab: Vocab, kwargs: dict[str, Any]) -> None:
    target = next(name for name in ("gold", "predictions", "masks") if name in kwargs)
    assert FILES[target] in _error(_bundle, tmp_path, vocab, **kwargs)


def test_metadata_list_gives_no_group(tmp_path: Path, vocab: Vocab) -> None:
    bundle = _bundle(tmp_path, vocab, gold=gold_with({0: {"metadata": ["g1"]}})["gold"])
    assert bundle.records[0].group_id is None


BARE_OR_BAD_JOINT = {
    "mode_c_bare_labels": preds_with({0: {"predicted_labels": ["O", "O", "QUESTION"]}}, "C"),
    "mode_c_lowercase": preds_with({0: {"predicted_labels": ["O|FIRST_UPPER", "O|LOWER", "QUESTION|lower"]}}, "C"),
    "mode_c_space": preds_with({0: {"predicted_labels": [" O|FIRST_UPPER", "O|LOWER", "QUESTION|LOWER"]}}, "C"),
    "mode_a_joint_label": preds_with({0: {"predicted_labels": ["O|FIRST_UPPER", "O", "QUESTION"]}}),
}


@pytest.mark.parametrize("kwargs", BARE_OR_BAD_JOINT.values(), ids=BARE_OR_BAD_JOINT.keys())
def test_mode_c_bare_labels_rejected(tmp_path: Path, vocab: Vocab, kwargs: dict[str, Any]) -> None:
    assert "predicted_labels" in _error(_bundle, tmp_path, vocab, **kwargs)


@pytest.mark.parametrize(("mode", "expected"), [("C", ("O|FIRST_UPPER", "PERIOD|LOWER")), ("A", ("O", "PERIOD"))],
                         ids=["mode_c_joined", "mode_a_plain"])
def test_mode_c_gold_joined(tmp_path: Path, vocab: Vocab, mode: str, expected: tuple[str, ...]) -> None:
    gold = write_jsonl(tmp_path / "gold.jsonl",
                       [gold_row("s4", mode, ["hi", "there"], ["O", "PERIOD"], ["FIRST_UPPER", "LOWER"])])
    loaded_mode, records = load_scored_gold(gold, write_ids(tmp_path / "list.ids", ["s4"]), vocab)
    assert (loaded_mode, records[0].tokens, records[0].labels) == (mode, ("hi", "there"), expected)


GOLD_RULES = {
    **{f"missing_{field}": gold_with({1: {field: _DEL}}) for field in
       ("sample_id", "label_mode", "tokens", "punct_labels", "case_labels")},
    "mixed_label_modes": gold_with({1: {"label_mode": "B"}}),
    **{f"label_mode_{name}": gold_with({"*": {"label_mode": value}}) for name, value in
       {"d": "D", "lowercase_a": "a", "empty": "", "null": None}.items()},
    "punct_labels_one_short": gold_with({1: {"punct_labels": ["COMMA", "O"]}}),
    "case_labels_one_long": gold_with({1: {"case_labels": ["FIRST_UPPER", "LOWER", "LOWER", "LOWER"]}}),
    "tokens_empty": gold_with({1: {"tokens": [], "punct_labels": [], "case_labels": []}}),
    "mode_a_extended_punct": gold_with({1: {"punct_labels": ["COMMA", "O", "EXCLAMATION"]}}),
    "mode_c_extended_punct": gold_with({1: {"punct_labels": ["COMMA", "O", "EXCLAMATION"]}}, "C"),
    "mode_c_joint_punct": gold_with({1: {"punct_labels": ["COMMA", "O", "O|LOWER"]}}, "C"),
    "mode_a_unknown_case": gold_with({1: {"case_labels": ["FIRST_UPPER", "UPPER", "LOWER"]}}),
    "mode_b_unknown_case": gold_with({1: {"case_labels": ["Title", "LOWER", "LOWER"]}}, "B"),
    "duplicate_sample_id": gold_with({1: {"sample_id": "s1"}}),
    # s3 (row 2) is outside the ID list: the release version is checked over the whole gold file
    "mixed_policy_versions": gold_with({"*": {"metadata": {"policy_version": "m1-v1"}},
                                        2: {"metadata": {"policy_version": "m1-v2"}}}),
    "policy_version_on_first_line_only": gold_with({0: {"metadata": {"policy_version": "m1-v2"}}}),
    # an unversioned line counts as a version (None), not as "nothing seen yet"
    "policy_version_after_unversioned_lines": gold_with({2: {"metadata": {"policy_version": "m1-v2"}}}),
    "policy_version_not_a_string": gold_with({"*": {"metadata": {"policy_version": 2}}}),
    "policy_version_empty": gold_with({"*": {"metadata": {"policy_version": ""}}}),
}


@pytest.mark.parametrize("kwargs", GOLD_RULES.values(), ids=GOLD_RULES.keys())
def test_gold_rules(tmp_path: Path, vocab: Vocab, kwargs: dict[str, Any]) -> None:
    assert "gold.jsonl" in _error(_bundle, tmp_path, vocab, **kwargs)


def test_policy_version_read_from_gold(tmp_path: Path, vocab: Vocab) -> None:
    versioned = gold_with({"*": {"metadata": {"group_id": "g1", "policy_version": "m1-v2"}}})
    assert _bundle(tmp_path, vocab, **versioned).policy_version == "m1-v2"
    assert _bundle(tmp_path, vocab).policy_version is None  # fixture gold has no policy_version


def test_mixed_policy_versions_name_the_line_and_both_versions(tmp_path: Path, vocab: Vocab) -> None:
    mixed = gold_with({"*": {"metadata": {"policy_version": "m1-v1"}}, 2: {"metadata": {"policy_version": "m1-v2"}}})
    message = _error(_bundle, tmp_path, vocab, **mixed)
    assert all(part in message for part in ("gold.jsonl line 3", "policy_version", "'m1-v1'", "'m1-v2'"))
    case = write_case(tmp_path, mixed["gold"], SCORED, perfect_preds("A"))
    assert "policy_version" in _error(load_scored_gold, case.gold, case.ids, vocab)


def test_mode_b_gold_uses_extended_vocab(tmp_path: Path, vocab: Vocab) -> None:
    bundle = _load(valid_case(tmp_path, "B"), vocab)
    assert bundle.label_mode == "B"
    assert bundle.records[1].gold == ("COLON", "O", "EXCLAMATION")


def test_group_id_kept_or_none(tmp_path: Path, vocab: Vocab) -> None:
    bundle = _bundle(tmp_path, vocab, ids=("s1", "s3"), predictions=perfect_preds("A", ("s1", "s3")))
    assert [record.group_id for record in bundle.records] == ["g1", None]
    empty_metadata = _bundle(tmp_path, vocab, gold=gold_with({0: {"metadata": {}}})["gold"])
    assert empty_metadata.records[0].group_id is None


@pytest.mark.parametrize(("ids", "offender"), [(["s1", "s1"], "s1"), (["s1", "s9"], "s9")],
                         ids=["duplicate_id", "id_not_in_gold"])
def test_ids_rules(tmp_path: Path, vocab: Vocab, ids: list[str], offender: str) -> None:
    message = _error(_bundle, tmp_path, vocab, ids=ids)
    assert "list.ids" in message
    assert offender in message


PRED_RULES = {
    "model_name_changes": preds_with({1: {"model_name": "other"}}),
    "label_mode_b_everywhere": preds_with({"*": {"label_mode": "B"}}),
    "label_mode_b_on_s2_only": preds_with({1: {"label_mode": "B"}}),
    "duplicate_sample_id": preds_with({1: {"sample_id": "s1"}}),
    "sample_id_not_in_gold": {"predictions": perfect_preds("A") + [pred_row("s9", "A", ["O"])]},
    "scored_s2_missing": {"predictions": perfect_preds("A", ("s1",))},
    "s1_two_labels": preds_with({0: {"predicted_labels": ["O", "O"]}}),
    "s1_four_labels": preds_with({0: {"predicted_labels": ["O", "O", "O", "QUESTION"]}}),
    "s1_extended_label_in_mode_a": preds_with({0: {"predicted_labels": ["O", "O", "EXCLAMATION"]}}),
    "s1_lowercase_label": preds_with({0: {"predicted_labels": ["o", "O", "QUESTION"]}}),
    **{f"missing_{field}": preds_with({0: {field: _DEL}}) for field in
       ("sample_id", "model_name", "label_mode", "predicted_labels")},
}


@pytest.mark.parametrize("kwargs", PRED_RULES.values(), ids=PRED_RULES.keys())
def test_prediction_rules(tmp_path: Path, vocab: Vocab, kwargs: dict[str, Any]) -> None:
    assert "preds.jsonl" in _error(_bundle, tmp_path, vocab, **kwargs)


def test_mode_b_predictions_accept_extended_labels(tmp_path: Path, vocab: Vocab) -> None:
    predictions = [pred_row("s1", "B", ["DASH", "SEMICOLON", "QUESTION"]),
                   pred_row("s2", "B", ["COLON", "O", "EXCLAMATION"])]
    bundle = _bundle(tmp_path, vocab, "B", predictions=predictions)
    assert bundle.records[0].pred == ("DASH", "SEMICOLON", "QUESTION")


@pytest.mark.parametrize(("ids", "ignored"), [(("s1", "s2", "s3"), 1), (("s1", "s2"), 0)],
                         ids=["with_s3_prediction", "without_s3_prediction"])
def test_extra_prediction_ids_counted(tmp_path: Path, vocab: Vocab, ids: tuple[str, ...], ignored: int) -> None:
    bundle = _bundle(tmp_path, vocab, predictions=perfect_preds("A", ids))
    assert bundle.ignored_prediction_ids == ignored
    assert _ids(bundle.records) == ("s1", "s2")


def test_extra_prediction_still_validated(tmp_path: Path, vocab: Vocab) -> None:
    predictions = perfect_preds("A") + [pred_row("s3", "A", ["O", "O"])]
    assert "preds.jsonl" in _error(_bundle, tmp_path, vocab, predictions=predictions)


MASK_RULES = {
    **{f"masks_given_for_mode_{mode.lower()}": {"mode": mode, "masks": all_masks()} for mode in "AB"},
    "mode_c_without_masks": {"mode": "C", "masks": None},
    "scored_s2_missing": {"mode": "C", "masks": [mask_row("s1", []), mask_row("s3", [])]},
    **{f"position_{name}": masks_with({0: {"excluded_positions": positions}}) for name, positions in
       {"negative": [-1], "equals_length": [3], "duplicated": [1, 1], "float": [1.5], "string": ["1"],
        "bool": [True]}.items()},
    "two_entries_for_s1": {"mode": "C", "masks": all_masks() + [mask_row("s1", [])]},
    "missing_excluded_positions": masks_with({0: {"excluded_positions": _DEL}}),
    "missing_reason": masks_with({0: {"reason": _DEL}}),  # decisions.md: spec 3 record includes "reason"
}


@pytest.mark.parametrize("kwargs", MASK_RULES.values(), ids=MASK_RULES.keys())
def test_mask_rules(tmp_path: Path, vocab: Vocab, kwargs: dict[str, Any]) -> None:
    message = _error(_bundle, tmp_path, vocab, **kwargs)
    if kwargs["mode"] == "C" and kwargs["masks"] is not None:  # errors inside the masks file name it and the place
        assert "masks.jsonl" in message and ("line " in message or "s2" in message)


def test_masks_for_unscored_ids_accepted(tmp_path: Path, vocab: Vocab) -> None:
    bundle = _bundle(tmp_path, vocab, "C", masks=all_masks({"s1": [0, 2]}))
    assert bundle.records[0].masked == frozenset({0, 2})
    assert bundle.records[1].masked == frozenset()


def test_aligned_records_are_frozen(tmp_path: Path, vocab: Vocab) -> None:
    bundle = _bundle(tmp_path, vocab, "C", masks=all_masks({"s1": [1]}))
    record = bundle.records[0]
    assert (type(record.gold), type(record.pred), type(record.masked)) == (tuple, tuple, frozenset)
    assert record.masked == frozenset({1})
    assert record.gold == ("O|FIRST_UPPER", "O|LOWER", "QUESTION|LOWER")
    for target, field in ((record, "pred"), (bundle, "records")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(target, field, ())


def test_bundle_inputs_identity(tmp_path: Path, vocab: Vocab) -> None:
    case = valid_case(tmp_path, "A")
    bundle = _load(case, vocab)
    expected = {name: {"path": f"{tmp_path.name}/{FILES[name]}",
                       "sha256": hashlib.sha256(getattr(case, name).read_bytes()).hexdigest()}
                for name in ("gold", "ids", "predictions")}
    assert bundle.label_mode == "A"
    assert bundle.model_name == "m"
    assert bundle.inputs == {**expected, "masks": None}
    assert sha256_file(case.gold) == expected["gold"]["sha256"]
    assert file_identity(LABEL_SETS_PATH) == {"path": "configs/label_sets.yaml", "sha256": sha256_file(LABEL_SETS_PATH)}


def test_file_identity_with_shows_a_path_relative_to_its_root_or_the_last_parts_outside_it(tmp_path: Path) -> None:
    path = tmp_path / "folder" / "f.txt"
    path.parent.mkdir()
    path.write_text("x", encoding="utf-8")
    digest = sha256_file(path)
    elsewhere = tmp_path / "elsewhere"
    assert file_identity_with(path, tmp_path, 1) == {"path": "folder/f.txt", "sha256": digest}  # inside the root
    assert file_identity_with(path, elsewhere, 2) == {"path": "folder/f.txt", "sha256": digest}  # outside: 2 parts
    assert file_identity_with(path, elsewhere, 1) == {"path": "f.txt", "sha256": digest}  # outside: the name only
    assert file_identity(path) == file_identity_with(path, elsewhere, 2)  # the default root is the repository


def test_bundle_inputs_identity_includes_masks_in_mode_c(tmp_path: Path, vocab: Vocab) -> None:
    case = valid_case(tmp_path, "C")
    assert _load(case, vocab).inputs["masks"] == {
        "path": f"{tmp_path.name}/masks.jsonl", "sha256": hashlib.sha256(case.masks.read_bytes()).hexdigest()}


LOCATED = {
    "duplicate_gold_id_names_line_2": (gold_with({1: {"sample_id": "s1"}}), ("gold.jsonl", "line 2")),
    "blank_lines_still_counted": ({"gold": [gold_rows("A")[0], "", "   ", '{"bad": ']}, ("gold.jsonl", "line 4")),
    "missing_prediction_names_id": ({"predictions": perfect_preds("A", ("s1",))}, ("preds.jsonl", "s2")),
    "id_not_in_gold_names_id": ({"ids": ("s1", "s9")}, ("list.ids", "s9")),
    "missing_mask_names_id": ({"mode": "C", "masks": [mask_row("s1", []), mask_row("s3", [])]},
                              ("masks.jsonl", "s2")),
}


@pytest.mark.parametrize(("kwargs", "fragments"), LOCATED.values(), ids=LOCATED.keys())
def test_errors_name_file_and_location(tmp_path: Path, vocab: Vocab, kwargs: dict[str, Any],
                                       fragments: tuple[str, ...]) -> None:
    message = _error(_bundle, tmp_path, vocab, **kwargs)
    for fragment in fragments:
        assert fragment in message


def test_bundle_identity_returns_the_recorded_file_identity() -> None:
    identity = {"path": "gold.jsonl", "sha256": "abc"}
    bundle = Bundle("A", "m", (), 0, {"gold": identity, "masks": None})
    assert bundle.identity("gold") == identity


def test_bundle_identity_of_an_input_that_was_not_given_is_an_input_error() -> None:
    bundle = Bundle("A", "m", (), 0, {"gold": None})
    with pytest.raises(EvaluationInputError, match="no gold file identity"):
        bundle.identity("gold")
