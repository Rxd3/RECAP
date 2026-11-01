"""train-hmm keeps main's extra label-configuration checks on top of the shared loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from punct_restore.cli.train_hmm import load_labels
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
