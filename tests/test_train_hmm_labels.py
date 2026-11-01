"""train-hmm keeps main's extra label-configuration checks on top of the shared loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from punct_restore.cli.train_hmm import load_labels
from punct_restore.data import labeling
from punct_restore.data.labeling import LABEL_SETS_PATH


def _config(tmp_path: Path, edit: dict[str, object] | None = None, mode_edit: tuple[str, str, object] | None = None
            ) -> Path:
    config = yaml.safe_load(Path(LABEL_SETS_PATH).read_text(encoding="utf-8"))
    config.update(edit or {})
    if mode_edit:
        mode, key, value = mode_edit
        config["modes"][mode][key] = value
    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize("mode", ["A", "B", "C"])
def test_the_committed_config_gives_the_mode_labels(mode: str, tmp_path: Path) -> None:
    labels = load_labels(_config(tmp_path), mode)
    assert len(labels) == {"A": 4, "B": 8, "C": 12}[mode] and all(isinstance(label, str) for label in labels)


@pytest.mark.parametrize("version", [2, "1", None])
def test_another_schema_version_is_refused(version: object, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"^labels\.yaml: unsupported label configuration version$"):
        load_labels(_config(tmp_path, {"schema_version": version}), "A")


def test_a_wrong_num_classes_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"^labels\.yaml: num_classes does not match label vocabulary$"):
        load_labels(_config(tmp_path, mode_edit=("A", "num_classes", 5)), "A")


def test_joint_labels_must_use_the_bar_separator(tmp_path: Path) -> None:
    config = yaml.safe_load(Path(LABEL_SETS_PATH).read_text(encoding="utf-8"))
    config["modes"]["C"]["separator"] = "/"
    config["modes"]["C"]["labels"] = [label.replace("|", "/") for label in config["modes"]["C"]["labels"]]
    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=r"^labels\.yaml: joint labels must use the \| separator$"):
        load_labels(path, "C")


def test_the_label_config_is_parsed_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The version, mode, separator and num_classes checks use the mapping the labels came from."""
    reads: list[object] = []
    real = labeling.read_yaml
    monkeypatch.setattr(labeling, "read_yaml", lambda path: reads.append(path) or real(path))
    load_labels(_config(tmp_path), "C")
    assert len(reads) == 1


def test_the_labels_are_the_ones_load_label_sets_gives(tmp_path: Path) -> None:
    path = _config(tmp_path)
    assert {mode: load_labels(path, mode) for mode in "ABC"} == {
        mode: labeling.load_label_sets(path)[mode] for mode in "ABC"}


def _without_key(tmp_path: Path, *keys: str) -> Path:
    config = yaml.safe_load(Path(LABEL_SETS_PATH).read_text(encoding="utf-8"))
    node = config
    for key in keys[:-1]:
        node = node[key]
    del node[keys[-1]]
    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("keys", "mode", "message"),
    [
        (("schema_version",), "A", "labels.yaml: missing schema_version"),
        (("modes", "A"), "A", "labels.yaml: missing modes.A"),
        (("modes", "B", "num_classes"), "B", "labels.yaml: missing modes.B.num_classes"),
    ],
    ids=["version", "mode", "num-classes"],
)
def test_a_missing_key_is_named_by_its_dotted_path(tmp_path: Path, keys: tuple[str, ...], mode: str, message: str
                                                  ) -> None:
    with pytest.raises(ValueError) as problem:
        load_labels(_without_key(tmp_path, *keys), mode)
    assert str(problem.value) == message


@pytest.mark.parametrize("value", ["4", 4.0, True, None], ids=["text", "float", "bool", "null"])
def test_num_classes_that_is_no_whole_number_is_named_not_called_a_mismatch(tmp_path: Path, value: object) -> None:
    path = _config(tmp_path, mode_edit=("A", "num_classes", value))
    with pytest.raises(ValueError) as problem:
        load_labels(path, "A")
    assert str(problem.value) == "labels.yaml: modes.A.num_classes is not a whole number"


def test_a_mode_that_is_no_mapping_is_named(tmp_path: Path) -> None:
    config = yaml.safe_load(Path(LABEL_SETS_PATH).read_text(encoding="utf-8"))
    config["modes"]["A"] = ["not", "a", "mapping"]
    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError) as problem:
        load_labels(path, "A")
    assert str(problem.value) == "labels.yaml: modes.A is not a JSON object"
