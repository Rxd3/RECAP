"""What validate_config requires of an experiment config: its shape first, then the frozen protocol."""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from test_hmm_cli import ROOT

from punct_restore.models.hmm_experiment_protocol import validate_config

TEXT_KEYS = ("processed_dir", "splits_dir", "label_config", "dataset_release_id", "policy_version", "experiment_id")


def config() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "configs/hmm_sponspeech_v2.yaml").read_text(encoding="utf-8"))


def without(content: dict[str, Any], key: str) -> dict[str, Any]:
    return {k: v for k, v in content.items() if k != key}


# ---- validate_config: the keys every step reads ----


@pytest.mark.parametrize("key", TEXT_KEYS)
def test_a_config_without_a_text_key_is_refused_naming_the_source_and_the_key(key: str) -> None:
    """A validator that stops checking the keys the later steps index lets a KeyError reach the user."""
    with pytest.raises(ValueError, match=f"^plan.json: invalid experiment config: missing {key}$"):
        validate_config(without(config(), key), "plan.json")


def test_a_config_without_the_evaluator_commit_is_refused_naming_the_nested_key() -> None:
    broken = {**config(), "evaluation": without(config()["evaluation"], "repository_commit")}
    with pytest.raises(ValueError, match="^x.yaml: invalid experiment config: missing evaluation.repository_commit$"):
        validate_config(broken, "x.yaml")


@pytest.mark.parametrize("value", ["", "  ", None, 5, ["x"]], ids=["empty", "blank", "null", "number", "list"])
@pytest.mark.parametrize("key", TEXT_KEYS)
def test_a_text_key_that_is_not_nonempty_text_is_refused(key: str, value: Any) -> None:
    """A validator that only checks the key is present accepts an empty YAML value (null) or a number."""
    with pytest.raises(ValueError, match=f"^x.yaml: invalid experiment config: {key} is not nonempty text$"):
        validate_config({**config(), key: value}, "x.yaml")


def test_modes_must_be_a_list_not_a_string_of_letters() -> None:
    """The string 'ABC' passes every other check, because iterating it gives the three letters."""
    with pytest.raises(ValueError, match="^x.yaml: invalid experiment config: modes is not a JSON list$"):
        validate_config({**config(), "modes": "ABC"}, "x.yaml")


def test_the_source_name_is_optional_and_a_valid_config_is_returned_unchanged() -> None:
    valid = config()
    assert validate_config(valid) is valid
    with pytest.raises(ValueError, match="^config: invalid experiment config: missing experiment_id$"):
        validate_config(without(valid, "experiment_id"))


# ---- the shape check comes first and never lets Python's own error text out ----

PREFIX = "x.yaml: invalid experiment config: "
PYTHON_TEXT = ("subscriptable", "unhashable", "indices", "not supported", "NoneType", "KeyError", "TypeError",
               "not iterable", "object has no", "int()", "float()")
MISSING = object()  # "the key is not there"
# every key of the config that the later steps index, with the kind of value it must hold
LISTS = ("modes", "search.smoothing", "search.min_token_count", "search.tie_break", "evaluation.test_sets")
SECTIONS = ("search", "evaluation", "runtime", "search.selection_blocks")
LEAVES = ("schema_version", *TEXT_KEYS, "evaluation.repository_commit", "evaluation.zero_division",
          "runtime.comparison_ids", "runtime.warmup_sequences", "runtime.repetitions", "runtime.model_threads",
          "search.selection_split", "search.metric", "search.score_precision", "search.retrain_on_validation")
ALL_KEYS = (*LEAVES, *LISTS, *SECTIONS)


def damaged(path: str, value: Any) -> dict[str, Any]:
    """The real config with the value at the dotted `path` replaced (or, for MISSING, taken out)."""
    content = config()
    *parents, name = path.split(".")
    holder = content
    for parent in parents:
        holder = holder[parent]
    if value is MISSING:
        del holder[name]
    else:
        holder[name] = value
    return content


DAMAGE = {"missing": MISSING, "null": None, "list": [], "object": {}, "number": 5, "text": "text",
          "items-list": [["A"]], "items-object": [{}], "items-null": [None], "items-number": [1]}


@pytest.mark.parametrize("how", DAMAGE)
@pytest.mark.parametrize("path", ALL_KEYS)
def test_a_damaged_config_is_a_value_error_with_no_python_text_whatever_the_key_and_the_damage(
        path: str, how: str) -> None:
    """The shape check runs before any step indexes the config; a TypeError or KeyError escaping fails this test."""
    try:
        validate_config(damaged(path, DAMAGE[how]), "x.yaml")
    except ValueError as problem:
        text = str(problem)
        assert "\n" not in text and not [bit for bit in PYTHON_TEXT if bit in text], text


@pytest.mark.parametrize("path", ALL_KEYS)
def test_a_config_that_lacks_a_key_names_it(path: str) -> None:
    with pytest.raises(ValueError) as problem:
        validate_config(damaged(path, MISSING), "x.yaml")
    assert str(problem.value) == f"{PREFIX}missing {path}"


@pytest.mark.parametrize("path", [*SECTIONS])
@pytest.mark.parametrize("value", [None, [], 5, "text"], ids=["null", "list", "number", "text"])
def test_a_section_that_is_no_object_is_refused_by_name(path: str, value: Any) -> None:
    with pytest.raises(ValueError) as problem:
        validate_config(damaged(path, value), "x.yaml")
    assert str(problem.value) == f"{PREFIX}{path} is not a JSON object"


@pytest.mark.parametrize("path", LISTS)
@pytest.mark.parametrize("value", [None, {}, "text", 5], ids=["null", "object", "text", "number"])
def test_a_list_that_is_no_list_is_refused_by_name(path: str, value: Any) -> None:
    with pytest.raises(ValueError) as problem:
        validate_config(damaged(path, value), "x.yaml")
    assert str(problem.value) == f"{PREFIX}{path} is not a JSON list"


@pytest.mark.parametrize(
    "item", [["A"], {}, None, 1, "", "  "], ids=["list", "object", "null", "number", "empty", "blank"]
)
def test_a_mode_that_is_not_text_is_refused_naming_its_place(item: Any) -> None:
    """`[["A"]]` reached `set(modes)` and died with 'unhashable type: list'."""
    with pytest.raises(ValueError) as problem:
        validate_config(damaged("modes", ["A", item]), "x.yaml")
    assert str(problem.value) == f"{PREFIX}modes.1 is not nonempty text"


@pytest.mark.parametrize("path", ["search.smoothing", "search.min_token_count"])
@pytest.mark.parametrize("item", [[1], {}, None, True, "1"], ids=["list", "object", "null", "bool", "text"])
def test_a_search_value_that_is_not_a_number_is_refused_naming_its_place(path: str, item: Any) -> None:
    with pytest.raises(ValueError) as problem:
        validate_config(damaged(path, [1, item]), "x.yaml")
    assert str(problem.value) == f"{PREFIX}{path}.1 is not a number"


@pytest.mark.parametrize("path", ["runtime.repetitions", "runtime.warmup_sequences"])
@pytest.mark.parametrize("value", [2.5, True, "3", None], ids=["float", "bool", "text", "null"])
def test_a_runtime_count_that_is_not_a_whole_number_is_refused(path: str, value: Any) -> None:
    """A text count died in `repetitions < 2`; a fractional one would die later in `range(repetitions)`."""
    with pytest.raises(ValueError) as problem:
        validate_config(damaged(path, value), "x.yaml")
    assert str(problem.value) == f"{PREFIX}{path} is not a whole number"


@pytest.mark.parametrize("content", [None, [1], "text", 3], ids=["null", "list", "text", "number"])
def test_a_config_that_is_no_object_is_refused(content: Any) -> None:
    with pytest.raises(ValueError) as problem:
        validate_config(content, "x.yaml")
    assert str(problem.value) == f"{PREFIX}not a JSON object"


def test_a_config_object_with_no_key_is_refused_naming_the_first_key() -> None:
    with pytest.raises(ValueError) as problem:
        validate_config({}, "x.yaml")
    assert str(problem.value) == f"{PREFIX}missing schema_version"


def test_a_mode_without_a_selection_block_is_a_protocol_error_not_a_key_error() -> None:
    content = config()
    del content["search"]["selection_blocks"]["B"]
    with pytest.raises(ValueError) as problem:
        validate_config(content, "x.yaml")
    assert str(problem.value) == f"{PREFIX}mode B: wrong primary selection block"


def test_the_shape_is_checked_before_the_protocol() -> None:
    """A config that breaks a rule of the protocol AND lacks a key is reported for the missing key."""
    content = config()
    content["search"]["selection_split"] = "test"
    del content["runtime"]
    with pytest.raises(ValueError) as problem:
        validate_config(content, "x.yaml")
    assert str(problem.value) == f"{PREFIX}missing runtime"


# ---- the protocol rules: the same one-line form as the shape errors, starting with the file name ----

RULE_CASES = [
    ("schema_version", 2, "schema_version must be 1 and modes must not be empty"),
    ("modes", [], "schema_version must be 1 and modes must not be empty"),
    ("modes", ["A", "A"], "modes must be unique members of A/B/C"),
    ("modes", ["A", "D"], "modes must be unique members of A/B/C"),
    ("search.selection_split", "test", "settings must be selected on validation; training stays train-only"),
    ("search.retrain_on_validation", True, "settings must be selected on validation; training stays train-only"),
    ("search.metric", "accuracy", "selection must use the shared scorer's six-decimal primary macro F1"),
    ("search.score_precision", 4, "selection must use the shared scorer's six-decimal primary macro F1"),
    ("search.tie_break", ["test_score"], "unsupported tie-break policy"),
    ("evaluation.test_sets", ["test"], "report test and test_amb separately"),
    ("search.selection_blocks", {"A": "punctuation/all", "B": "punctuation/all", "C": "punctuation/all"},
     "mode C: wrong primary selection block"),
    ("search.smoothing", [], "search values must be nonempty and unique"),
    ("search.smoothing", [0.1, 0.1], "search values must be nonempty and unique"),
    ("search.smoothing", [-0.1], "smoothing/cutoffs must be positive; cutoffs must be integers"),
    ("search.smoothing", [0.0], "smoothing/cutoffs must be positive; cutoffs must be integers"),
    ("search.smoothing", [float("inf")], "smoothing/cutoffs must be positive; cutoffs must be integers"),
    ("search.min_token_count", [1.5], "smoothing/cutoffs must be positive; cutoffs must be integers"),
    ("search.min_token_count", [0], "smoothing/cutoffs must be positive; cutoffs must be integers"),
    ("runtime.repetitions", 1, "runtime measurement requires warmup and repetitions"),
    ("runtime.warmup_sequences", 0, "runtime measurement requires warmup and repetitions"),
    ("runtime.model_threads", 2, "this implementation uses one model thread and zero_division=0"),
    ("evaluation.zero_division", 1, "this implementation uses one model thread and zero_division=0"),
]


@pytest.mark.parametrize(("path", "value", "reason"), RULE_CASES, ids=[f"{p}={v}" for p, v, _ in RULE_CASES])
def test_a_protocol_rule_error_starts_with_the_file_name_like_a_shape_error(path: str, value: Any, reason: str) -> None:
    with pytest.raises(ValueError) as problem:
        validate_config(damaged(path, value), "x.yaml")
    assert str(problem.value) == f"{PREFIX}{reason}"


def test_a_protocol_rule_error_names_the_default_source_when_none_is_given() -> None:
    with pytest.raises(ValueError) as problem:
        validate_config(damaged("modes", ["A", "A"]))
    assert str(problem.value) == "config: invalid experiment config: modes must be unique members of A/B/C"
