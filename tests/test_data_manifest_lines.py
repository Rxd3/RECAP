"""MANIFEST.sha256: one parser for its lines; a line that is not '<sha256>  <path>' is one ValueError naming it."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from punct_restore.cli import prepare_data
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


# --- a listed file that is missing, and a listed path that leaves the raw root --------------------------------------


def test_a_listed_file_that_is_missing_names_the_path_and_says_what_to_run(tmp_path: Path) -> None:
    gone = f"{hashlib.sha256(b'x').hexdigest()}  sponspeech_text/sub/gone.txt\n"
    _, manifest = _raw_root(tmp_path, GOOD + gone)
    with pytest.raises(ValueError) as problem:
        preprocessing.verify_raw_manifest(manifest)
    assert str(problem.value) == (
        "sponspeech_text/sub/gone.txt: listed in MANIFEST.sha256 but missing; run recap prepare-data extract again")
    assert FOLDER not in str(problem.value)


def test_the_missing_file_message_names_the_manifest_that_was_given(tmp_path: Path) -> None:
    _, manifest = _raw_root(tmp_path, f"{hashlib.sha256(b'x').hexdigest()}  gone.txt\n", name="OTHER.sha256")
    with pytest.raises(ValueError, match=r"^gone\.txt: listed in OTHER\.sha256 but missing"):
        preprocessing.verify_raw_manifest(manifest)


def test_a_listed_file_that_differs_is_still_returned_not_raised(tmp_path: Path) -> None:
    _, manifest = _raw_root(tmp_path, f"{hashlib.sha256(b'other').hexdigest()}  sponspeech_text/a.txt\n")
    assert preprocessing.verify_raw_manifest(manifest) == ["sponspeech_text/a.txt"]


def _outside_file(tmp_path: Path) -> tuple[Path, str]:
    """A file next to the raw root, and its SHA-256: a manifest that lists it would verify clean without the check."""
    outside = tmp_path / "outside.txt"
    outside.write_text("hello", encoding="utf-8")
    return outside, hashlib.sha256(b"hello").hexdigest()


def test_a_listed_path_that_climbs_out_of_the_raw_root_is_refused_although_its_hash_matches(tmp_path: Path) -> None:
    _, digest = _outside_file(tmp_path)
    _, manifest = _raw_root(tmp_path, f"{digest}  ../outside.txt\n")
    with pytest.raises(ValueError) as problem:
        preprocessing.verify_raw_manifest(manifest)
    assert str(problem.value) == "../outside.txt: listed in MANIFEST.sha256 but outside the raw root"


def test_a_listed_path_that_climbs_out_after_going_in_is_refused(tmp_path: Path) -> None:
    _, digest = _outside_file(tmp_path)
    _, manifest = _raw_root(tmp_path, f"{digest}  sponspeech_text/../../outside.txt\n")
    with pytest.raises(ValueError, match="outside the raw root"):
        preprocessing.verify_raw_manifest(manifest)


def test_an_absolute_listed_path_is_refused_even_when_the_file_is_there(tmp_path: Path) -> None:
    outside, digest = _outside_file(tmp_path)
    _, manifest = _raw_root(tmp_path, f"{digest}  {outside}\n")
    with pytest.raises(ValueError) as problem:
        preprocessing.verify_raw_manifest(manifest)
    assert str(problem.value) == f"{outside}: listed in MANIFEST.sha256 but outside the raw root"


def test_a_listed_path_that_stays_inside_by_a_detour_is_still_checked(tmp_path: Path) -> None:
    root, _ = _raw_root(tmp_path, GOOD)
    detour = f"{hashlib.sha256(b'hello').hexdigest()}  sponspeech_text/../sponspeech_text/a.txt\n"
    (root / "MANIFEST.sha256").write_text(detour, encoding="utf-8", newline="\n")
    assert preprocessing.verify_raw_manifest(root / "MANIFEST.sha256") == []


def test_prepare_data_build_reports_a_partly_extracted_raw_root_in_one_line(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, _ = _raw_root(tmp_path, GOOD + f"{hashlib.sha256(b'x').hexdigest()}  sponspeech_text/gone.txt\n")
    (root / "video_metadata_oembed.jsonl").write_text("", encoding="utf-8")
    argv = ["build", "--raw-root", str(root), "--release-id", "rel", "--processed-dir", str(tmp_path / "p"),
            "--splits-dir", str(tmp_path / "s")]
    assert prepare_data.main(argv) == 1
    assert capsys.readouterr().err == (
        "error: sponspeech_text/gone.txt: listed in MANIFEST.sha256 but missing; "
        "run recap prepare-data extract again\n")
