"""Dataset preparation and shared-contract validation, owned by Member 1.

Modules: preprocessing (source loading, cleaning, tokenization), labeling (punctuation and casing labels),
splits (sample IDs, split mapping, leakage audits), validation (DATA_CONTRACT.md checks), release (building and
loading versioned releases). Policies D1-D6 are recorded in docs/DECISIONS.md; usage in data/README.md.

The names below load their module on first use (`punct_restore.lazy_exports`): importing `punct_restore.data.labeling`
does not load the release builder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..lazy_exports import lazy_exports

if TYPE_CHECKING:  # what the lazy names are, for type checkers and editors
    from .labeling import join_labels, load_label_sets
    from .release import ReleaseSources, build_release, build_release_from, load_records, read_ids
    from .validation import validate_file, validate_record

__all__ = ["ReleaseSources", "build_release", "build_release_from", "join_labels", "load_label_sets",
           "load_records", "read_ids", "validate_file", "validate_record"]

__getattr__, __dir__ = lazy_exports(__name__, {
    "ReleaseSources": "release", "build_release": "release", "build_release_from": "release",
    "join_labels": "labeling", "load_label_sets": "labeling", "load_records": "release", "read_ids": "release",
    "validate_file": "validation", "validate_record": "validation"})
