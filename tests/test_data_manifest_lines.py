"""MANIFEST.sha256: one parser for its lines; a line that is not '<sha256>  <path>' is one ValueError naming it."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from punct_restore.data import extraction, preprocessing

FOLDER = "secretfolder"


def _raw_root(tmp_path: Path, manifest_text: str, name: str = "MANIFEST.sha256") -> tuple[Path, Path]:
    """A raw root with one listed file `sponspeech_text/a.txt` and a manifest holding `manifest_text`."""
    root = tmp_path / FOLDER
    (root / "sponspeech_text").mkdir(parents=True)
    (root / "sponspeech_text" / "a.txt").write_text("hello", encoding="utf-8")
    manifest = root / name
    manifest.write_text(manifest_text, encoding="utf-8", newline="\n")
    return root, manifest


GOOD = f"{hashlib.sha256(b'hello').hexdigest()}  sponspeech_text/a.txt\n"
BAD_LINES = {
    "no-separator": "just-one-field\n",
    "single-space": f"{hashlib.sha256(b'hello').hexdigest()} sponspeech_text/a.txt\n",
    "no-path": "abc123  \n",
    "empty-line": "\n",
}


def test_a_good_manifest_still_verifies_clean(tmp_path: Path) -> None:
    root, manifest = _raw_root(tmp_path, GOOD)
    assert preprocessing.verify_raw_manifest(manifest) == []
    assert extraction.verify_extraction(root, manifest) == ([], [])


@pytest.mark.parametrize("bad", BAD_LINES.values(), ids=BAD_LINES.keys())
def test_verify_raw_manifest_names_the_file_and_line_of_a_malformed_line(tmp_path: Path, bad: str) -> None:
    _, manifest = _raw_root(tmp_path, GOOD + bad)
    with pytest.raises(ValueError) as problem:
        preprocessing.verify_raw_manifest(manifest)
    assert str(problem.value) == "MANIFEST.sha256 line 2: expected '<sha256>  <path>'"


@pytest.mark.parametrize("bad", BAD_LINES.values(), ids=BAD_LINES.keys())
def test_verify_extraction_names_the_file_and_line_of_a_malformed_line(tmp_path: Path, bad: str) -> None:
    root, manifest = _raw_root(tmp_path, bad + GOOD)
    with pytest.raises(ValueError) as problem:
        extraction.verify_extraction(root, manifest)
    assert str(problem.value) == "MANIFEST.sha256 line 1: expected '<sha256>  <path>'"


def test_the_message_names_the_manifest_that_was_given_not_a_fixed_name(tmp_path: Path) -> None:
    _, manifest = _raw_root(tmp_path, "oops\n", name="OTHER.sha256")
    with pytest.raises(ValueError) as problem:
        preprocessing.verify_raw_manifest(manifest)
    assert str(problem.value) == "OTHER.sha256 line 1: expected '<sha256>  <path>'"
