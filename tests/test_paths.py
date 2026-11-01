"""Tests for punct_restore.paths: the repository root every command resolves its defaults against."""

from __future__ import annotations

from pathlib import Path

from punct_restore.paths import REPO_ROOT


def test_repo_root_holds_the_project():
    assert (REPO_ROOT / "pyproject.toml").is_file() and (REPO_ROOT / "src" / "punct_restore").is_dir()


def test_repo_root_is_an_absolute_path():
    assert isinstance(REPO_ROOT, Path) and REPO_ROOT.is_absolute()
