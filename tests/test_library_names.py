"""Spec section 2: every public library name of main 72ebfd1 is still importable from its old module.

The fixture is the snapshot of 446 names in 34 library modules taken on main. A function may move between modules
only if the old name stays importable from its old module; shrinking `__all__` is fine, removing the name is not.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "library_names_72ebfd1.json"
SNAPSHOT_NAMES = 446


def _snapshot() -> dict[str, list[str]]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_the_snapshot_holds_every_name_of_main() -> None:
    """A truncated or empty fixture would make the import test pass vacuously."""
    assert sum(len(names) for names in _snapshot().values()) == SNAPSHOT_NAMES


def test_every_name_of_main_is_importable_from_its_old_module() -> None:
    missing = [
        f"{module}.{name}"
        for module, names in _snapshot().items()
        for name in names
        if not hasattr(importlib.import_module(module), name)
    ]
    assert missing == []


# Names that only moved: the old module re-exports the object of the new home (an alias, not a copy).
_CONSTANTS = "punct_restore.data.release_constants"
_STATS = "punct_restore.data.release_stats"
_LEXICONS = "punct_restore.features.crf_lexicons"
_LABELING = "punct_restore.data.labeling"
_VALIDATION = "punct_restore.data.validation"
_MOVED = [
    *(("punct_restore.data.release", name, _CONSTANTS, name)
      for name in ("JSON", "RELEASE_FILES", "SOURCE_REFERENCE", "UNRESOLVED_CASE")),
    ("punct_restore.data.release", "describe_files", "punct_restore.data.release_manifest", "describe_files"),
    *(("punct_restore.data.release", name, _STATS, name)
      for name in ("label_distribution", "length_stats", "mode_labels")),
    ("punct_restore.data.release", "sha256_file", "punct_restore.hashing", "sha256_file"),
    ("punct_restore.data.release", "write_lines", "punct_restore.data.release_outputs", "write_lines"),
    *(("punct_restore.features.crf_features", name, _LEXICONS, name)
      for name in ("FILLERS", "FILLER_BIGRAMS", "LOOKBACK_WINDOW", "TAG_CUE_PRONOUNS", "TAG_CUE_WORDS",
                   "UNKNOWN_CLUSTER_LABEL")),
    ("punct_restore.models.crf", "CASING_VOCABULARY", "punct_restore.models.crf_modes", "CASING_VOCABULARY"),
    ("punct_restore.models.crf", "MODEL_SCHEMA_VERSION", "punct_restore.models.crf_io", "MODEL_SCHEMA_VERSION"),
    ("punct_restore.models.crf", "join_labels", _LABELING, "join_labels"),
    ("punct_restore.models.crf", "load_label_sets", _LABELING, "load_label_sets"),
    ("punct_restore.models.hmm_workflow", "check_model_input", _VALIDATION, "check_model_input"),
    ("punct_restore.models.hmm_workflow", "validate_record", _VALIDATION, "validate_record"),
    ("punct_restore.models.hmm_workflow", "load_label_sets", _LABELING, "load_label_sets"),
    ("punct_restore.models.hmm_workflow", "read_jsonl", "punct_restore.models.hmm_records", "read_jsonl"),
]


@pytest.mark.parametrize(("old_module", "name", "new_module", "new_name"), _MOVED,
                         ids=[f"{old}.{name}" for old, name, _, _ in _MOVED])
def test_a_name_that_only_moved_is_the_same_object_as_in_its_new_home(
        old_module: str, name: str, new_module: str, new_name: str) -> None:
    assert getattr(importlib.import_module(old_module), name) is getattr(importlib.import_module(new_module), new_name)
