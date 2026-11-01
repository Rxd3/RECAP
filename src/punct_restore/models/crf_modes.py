"""CRF label modes: the mode names, the casing vocabulary and the per-mode label vocabularies."""

from __future__ import annotations

from pathlib import Path

from punct_restore.data.labeling import load_label_sets

CASING_VOCABULARY: tuple[str, ...] = ("LOWER", "FIRST_UPPER", "ALL_UPPER")

SCORING_MODES: tuple[str, ...] = ("A", "B", "C")
LABEL_MODES: tuple[str, ...] = (*SCORING_MODES, "CASE")


def require_scoring_mode(label_mode: str) -> None:
    """Raise ValueError unless label_mode is a scoring mode; the casing helper 'CASE' is not one."""
    if label_mode not in SCORING_MODES:
        raise ValueError(f"Unsupported label_mode: {label_mode!r}. Must be 'A', 'B' or 'C' (a scoring mode).")


def get_mode_vocabulary(label_mode: str, *, config_path: str | Path | None = None) -> list[str]:
    """Retrieve the authoritative ordered label vocabulary for a given mode from configs/label_sets.yaml.

    Args:
        label_mode: 'A', 'B', 'C', or the casing helper mode 'CASE'.
        config_path: Optional path to label_sets.yaml (defaults to repository configuration).

    Returns:
        List of valid label strings in configuration order.

    Raises:
        ValueError: If label_mode is unsupported or configuration structure is invalid.
        FileNotFoundError: If configuration file does not exist.
    """
    if label_mode not in LABEL_MODES:
        raise ValueError(f"Unsupported label_mode: {label_mode!r}. Must be 'A', 'B', 'C', or 'CASE'.")

    label_sets = load_label_sets(str(config_path)) if config_path is not None else load_label_sets()
    return list(label_sets["casing" if label_mode == "CASE" else label_mode])
