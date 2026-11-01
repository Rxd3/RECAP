"""Dataset preparation and shared-contract validation, owned by Member 1.

Modules: preprocessing (source loading, cleaning, tokenization), labeling (punctuation and casing labels),
splits (sample IDs, split mapping, leakage audits), validation (DATA_CONTRACT.md checks), release (building and
loading versioned releases). Policies D1-D6 are recorded in docs/DECISIONS.md; usage in data/README.md.
"""

from .labeling import join_labels, load_label_sets
from .release import ReleaseSources, build_release, load_records, read_ids
from .validation import validate_file, validate_record

__all__ = ["ReleaseSources", "build_release", "join_labels", "load_label_sets", "load_records", "read_ids",
           "validate_file", "validate_record"]
