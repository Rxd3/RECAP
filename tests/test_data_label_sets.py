"""labeling.load_label_sets: a YAML file that parses but is not shaped like label_sets.yaml is one ValueError."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from punct_restore.data import labeling

REAL = Path(labeling.LABEL_SETS_PATH)


def _config() -> dict:
    return yaml.safe_load(REAL.read_text(encoding="utf-8"))


def _written(tmp_path: Path, config: object) -> Path:
    path = tmp_path / "custom_label_sets.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _without(config: dict, *keys: str) -> dict:
    """A copy of `config` with the nested entry at `keys` removed."""
    copy = yaml.safe_load(yaml.safe_dump(config))
    node = copy
    for key in keys[:-1]:
        node = node[key]
    del node[keys[-1]]
    return copy


def _at(config: dict, *keys: str) -> object:
    """The entry of `config` at `keys`."""
    node: Any = config
    for key in keys:
        node = node[key]
    return node


def _with(config: dict, value: object, *keys: str) -> dict:
    """A copy of `config` with the nested entry at `keys` replaced by `value`."""
    copy = yaml.safe_load(yaml.safe_dump(config))
    node = copy
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    return copy


def test_the_real_file_loads_with_its_four_eight_and_twelve_labels() -> None:
    sets = labeling.load_label_sets()
    assert sets["A"] == ["O", "COMMA", "PERIOD", "QUESTION"]
    assert len(sets["B"]) == 8 and len(sets["C"]) == 12
    assert sets["casing"] == ["LOWER", "FIRST_UPPER", "ALL_UPPER"]
    assert sets["separator"] == "|" and sets["projection"]["approved"] is True


def test_a_copy_of_the_real_file_loads_to_the_same_label_sets(tmp_path: Path) -> None:
    copy = tmp_path / "copy.yaml"
    copy.write_bytes(REAL.read_bytes())
    assert labeling.load_label_sets(copy) == labeling.load_label_sets()


@pytest.mark.parametrize(
    ("keys", "missing"),
    [
        (("punctuation",), "punctuation"),
        (("punctuation", "standard"), "punctuation.standard"),
        (("punctuation", "extended"), "punctuation.extended"),
        (("casing",), "casing"),
        (("modes",), "modes"),
        (("modes", "C"), "modes.C"),
        (("modes", "C", "separator"), "modes.C.separator"),
        (("modes", "C", "labels"), "modes.C.labels"),
        (("extended_to_standard",), "extended_to_standard"),
    ],
)
def test_a_missing_key_is_a_value_error_naming_the_file_and_the_key(
        tmp_path: Path, keys: tuple[str, ...], missing: str) -> None:
    path = _written(tmp_path, _without(_config(), *keys))
    with pytest.raises(ValueError) as problem:
        labeling.load_label_sets(path)
    assert str(problem.value) == f"{path.name}: missing {missing}"


@pytest.mark.parametrize(
    ("keys", "value", "expected"),
    [
        (("punctuation",), ["O"], "punctuation is not a JSON object"),
        (("punctuation", "standard"), "O,COMMA,PERIOD,QUESTION", "punctuation.standard is not a JSON list"),
        (("casing",), "LOWER", "casing is not a JSON list"),
        (("modes", "C", "separator"), 7, "modes.C.separator is not nonempty text"),
        (("modes", "C", "separator"), "", "modes.C.separator is not nonempty text"),
        (("modes", "C", "labels"), {"O|LOWER": 1}, "modes.C.labels is not a JSON list"),
        (("extended_to_standard",), ["approved"], "extended_to_standard is not a JSON object"),
    ],
)
def test_a_value_of_the_wrong_type_is_a_value_error_naming_the_file_and_the_key(
        tmp_path: Path, keys: tuple[str, ...], value: object, expected: str) -> None:
    path = _written(tmp_path, _with(_config(), value, *keys))
    with pytest.raises(ValueError) as problem:
        labeling.load_label_sets(path)
    assert str(problem.value) == f"{path.name}: {expected}"


@pytest.mark.parametrize("item", [{}, ["O"], 1, None], ids=["object", "list", "number", "null"])
@pytest.mark.parametrize("keys", [("punctuation", "standard"), ("punctuation", "extended"), ("casing",),
                                  ("modes", "C", "labels")])
def test_a_label_list_with_an_item_that_is_not_text_is_a_value_error_naming_the_item(
        tmp_path: Path, keys: tuple[str, ...], item: object) -> None:
    """A list check that only asks for a list lets `[1, 2, 3, 4]` build the label sets of a mode."""
    labels = list(_at(_config(), *keys))
    labels[1] = item
    path = _written(tmp_path, _with(_config(), labels, *keys))
    with pytest.raises(ValueError) as problem:
        labeling.load_label_sets(path)
    assert str(problem.value) == f"{path.name}: {'.'.join(keys)}.1 is not nonempty text"


@pytest.mark.parametrize("content", [[], "just text", None, 3])
def test_a_file_that_is_not_a_mapping_at_the_top_is_a_value_error(tmp_path: Path, content: object) -> None:
    path = _written(tmp_path, content)
    with pytest.raises(ValueError) as problem:
        labeling.load_label_sets(path)
    assert str(problem.value) == f"{path.name}: not a JSON object"


def test_the_message_names_the_file_only_not_its_folder(tmp_path: Path) -> None:
    folder = tmp_path / "some" / "deep" / "folder"
    folder.mkdir(parents=True)
    path = folder / "broken.yaml"
    path.write_text(yaml.safe_dump(_without(_config(), "casing")), encoding="utf-8")
    with pytest.raises(ValueError) as problem:
        labeling.load_label_sets(path)
    assert str(problem.value) == "broken.yaml: missing casing"
    assert str(tmp_path) not in str(problem.value)


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (b"punctuation: [unclosed", "not valid YAML"),
        (b"a: " + b"[" * 100_000, "not valid YAML"),  # nested so deep that the parser hits the recursion limit
        (b"a: \xe9\xff", "not valid UTF-8"),
    ],
    ids=["syntax", "nested", "not-utf8"],
)
def test_a_file_that_cannot_be_read_as_yaml_is_one_value_error_naming_the_file_only(
    tmp_path: Path, raw: bytes, reason: str
) -> None:
    folder = tmp_path / "secretfolder"
    folder.mkdir()
    path = folder / "broken.yaml"
    path.write_bytes(raw)
    with pytest.raises(ValueError) as problem:
        labeling.load_label_sets(path)
    assert str(problem.value) == f"broken.yaml: {reason}"  # the YAML mark would show the whole path


def test_a_wrong_structure_names_the_given_file_not_the_default_one(tmp_path: Path) -> None:
    folder = tmp_path / "secretfolder"
    folder.mkdir()
    path = folder / "c1.yaml"
    path.write_text(yaml.safe_dump(_with(_config(), ["LOWER"], "casing")), encoding="utf-8")
    with pytest.raises(ValueError) as problem:
        labeling.load_label_sets(path)
    assert str(problem.value) == "c1.yaml: does not have the agreed 4/8/12 structure"


@pytest.mark.parametrize(("text", "value"), [("", None), ("[1, 2]\n", [1, 2]), ("a: 1\n", {"a": 1}), ("null\n", None)])
def test_read_yaml_gives_whatever_the_file_holds(tmp_path: Path, text: str, value: object) -> None:
    """The reader does not look at the shape: that is for the caller's schema (the label config, an experiment)."""
    path = tmp_path / "any.yaml"
    path.write_text(text, encoding="utf-8")
    assert labeling.read_yaml(path) == value


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (b"a: [unclosed", "not valid YAML"),
        (b"a: " + b"[" * 100_000, "not valid YAML"),
        (b"a: \xe9\xff", "not valid UTF-8"),
    ],
    ids=["syntax", "nested", "not-utf8"],
)
def test_read_yaml_names_the_file_only_and_gives_no_parser_text(tmp_path: Path, raw: bytes, reason: str) -> None:
    folder = tmp_path / "secretfolder"
    folder.mkdir()
    path = folder / "broken.yaml"
    path.write_bytes(raw)
    with pytest.raises(ValueError) as problem:
        labeling.read_yaml(path)
    assert str(problem.value) == f"broken.yaml: {reason}"
