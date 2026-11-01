"""Repository-level paths shared by the `recap` commands."""

from __future__ import annotations

from pathlib import Path

# src/punct_restore/paths.py -> parents[0] = punct_restore, parents[1] = src, parents[2] = the repository root.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
