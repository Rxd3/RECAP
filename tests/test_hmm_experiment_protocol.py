"""What validate_config requires of an experiment config, and the shape check every study JSON file goes through."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from test_hmm_cli import ROOT

from punct_restore.models.hmm_experiment_protocol import read_object, require_object, validate_config

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


@pytest.mark.parametrize("value", ["", None, 5, ["x"]], ids=["empty", "null", "number", "list"])
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


# ---- require_object and read_object ----


def test_require_object_returns_the_value_it_checked() -> None:
    value = {"a": {"b": [1]}, "extra": 1}
    assert require_object("f.json", value, {"a": {"b": list}}) is value


@pytest.mark.parametrize("value", [[], None, "text", 3], ids=["list", "null", "text", "number"])
def test_require_object_refuses_anything_but_an_object(value: Any) -> None:
    with pytest.raises(ValueError, match="^f.json: not a JSON object$"):
        require_object("f.json", value, {})


def test_require_object_names_a_missing_key_by_its_dotted_path() -> None:
    with pytest.raises(ValueError, match="^f.json: missing a.b.c$"):
        require_object("f.json", {"a": {"b": {}}}, {"a": {"b": {"c": object}}})


def test_require_object_names_a_nested_value_of_the_wrong_kind() -> None:
    with pytest.raises(ValueError, match="^f.json: a.b is not a JSON object$"):
        require_object("f.json", {"a": {"b": []}}, {"a": {"b": {}}})
    with pytest.raises(ValueError, match="^f.json: a.xs is not a JSON list$"):
        require_object("f.json", {"a": {"xs": {}}}, {"a": {"xs": list}})


def test_a_key_that_may_hold_any_value_only_has_to_be_present() -> None:
    require_object("f.json", {"a": None}, {"a": object})
    with pytest.raises(ValueError, match="^f.json: missing a$"):
        require_object("f.json", {}, {"a": object})


def test_the_prefix_places_an_element_of_a_list_in_the_message() -> None:
    with pytest.raises(ValueError, match="^f.json: candidates.0 is not a JSON object$"):
        require_object("f.json", 5, {}, "candidates.0")
    with pytest.raises(ValueError, match="^f.json: candidates.0.model is not nonempty text$"):
        require_object("f.json", {"model": ""}, {"model": str}, "candidates.0")


def test_read_object_names_the_file_without_its_folder(tmp_path: Path) -> None:
    folder = tmp_path / "deep"
    folder.mkdir()
    (folder / "r.json").write_text('{"a": []}', encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        read_object(folder / "r.json", {"b": object})
    assert str(caught.value) == "r.json: missing b"
    assert read_object(folder / "r.json", {"a": list}) == {"a": []}


@pytest.mark.parametrize(
    ("content", "message"), [(b"{", "r.json: not valid JSON"), (b"\xff", "r.json: not valid UTF-8")]
)
def test_read_object_keeps_the_json_reader_messages(tmp_path: Path, content: bytes, message: str) -> None:
    (tmp_path / "r.json").write_bytes(content)
    with pytest.raises(ValueError) as caught:
        read_object(tmp_path / "r.json", {})
    assert str(caught.value) == message
