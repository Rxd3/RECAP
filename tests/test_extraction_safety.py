"""Archive extraction never writes outside raw_root/sponspeech_text, whatever names the tar entries carry."""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import pytest

from punct_restore.data import extraction


def _tar_with(tmp_path: Path, entries: dict[str, bytes]) -> Path:
    archive = tmp_path / "crafted.tar.gz"
    with tarfile.open(archive, mode="w:gz") as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return archive


def _extract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entries: dict[str, bytes]) -> Path:
    archive = _tar_with(tmp_path, entries)
    monkeypatch.setattr(extraction, "ARCHIVE_BYTES", archive.stat().st_size)
    raw_root = tmp_path / "raw"
    extraction.extract_archive(archive, raw_root)
    return raw_root


def test_a_normal_entry_lands_under_the_output_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_root = _extract(tmp_path, monkeypatch, {"sponspeech/train/text": b"utt1 hello\n"})
    assert (raw_root / extraction.OUTPUT_DIR / "train" / "text").read_bytes() == b"utt1 hello\n"


@pytest.mark.parametrize("name", ["sponspeech/../../escaped.txt", "sponspeech/train/../../../escaped.txt"])
def test_an_entry_climbing_out_of_the_output_folder_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    with pytest.raises(ValueError, match="outside"):
        _extract(tmp_path, monkeypatch, {name: b"x"})
    assert not (tmp_path / "escaped.txt").exists()
    assert not (tmp_path / "raw" / "escaped.txt").exists()


def test_an_absolute_entry_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "abs" / "escaped.txt"
    with pytest.raises(ValueError, match="outside"):
        extraction._write(tmp_path / "raw", "sponspeech/" + target.as_posix(), b"x")
    assert not target.exists()


def test_an_entry_on_another_drive_is_refused_with_the_entry_named(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_common_path(paths: list[str]) -> str:
        raise ValueError("Paths don't have the same drive")

    monkeypatch.setattr(extraction.os.path, "commonpath", no_common_path)
    with pytest.raises(ValueError, match=r"^sponspeech/Z:/escaped\.txt: archive entry would be written outside"):
        extraction._write(tmp_path / "raw", "sponspeech/Z:/escaped.txt", b"x")


def test_the_refusal_names_the_entry_not_a_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError) as caught:
        _extract(tmp_path, monkeypatch, {"sponspeech/../../escaped.txt": b"x"})
    assert str(tmp_path) not in str(caught.value) and "escaped.txt" in str(caught.value)


def test_a_sibling_folder_that_starts_with_the_output_folder_name_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """raw/sponspeech_text_evil/x starts with the string raw/sponspeech_text but is not inside that folder."""
    with pytest.raises(ValueError, match="outside"):
        _extract(tmp_path, monkeypatch, {"sponspeech/../sponspeech_text_evil/x": b"x"})
    assert not (tmp_path / "raw" / "sponspeech_text_evil").exists()


def _symlink(link: Path, target: Path, *, folder: bool) -> None:
    try:
        os.symlink(target, link, target_is_directory=folder)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")  # Windows without the privilege to create them


def test_a_symlinked_folder_inside_the_output_folder_is_not_written_through(tmp_path: Path) -> None:
    raw_root, outside = tmp_path / "raw", tmp_path / "outside"
    (raw_root / extraction.OUTPUT_DIR).mkdir(parents=True)
    outside.mkdir()
    _symlink(raw_root / extraction.OUTPUT_DIR / "link", outside, folder=True)
    with pytest.raises(ValueError, match=r"^sponspeech/link/x\.txt: archive entry would be written outside"):
        extraction._write(raw_root, "sponspeech/link/x.txt", b"x")
    assert list(outside.iterdir()) == []


def test_a_symlinked_file_inside_the_output_folder_is_not_written_through(tmp_path: Path) -> None:
    raw_root, victim = tmp_path / "raw", tmp_path / "victim.txt"
    (raw_root / extraction.OUTPUT_DIR).mkdir(parents=True)
    victim.write_bytes(b"keep me")
    _symlink(raw_root / extraction.OUTPUT_DIR / "link.txt", victim, folder=False)
    with pytest.raises(ValueError, match="outside"):
        extraction._write(raw_root, "sponspeech/link.txt", b"x")
    assert victim.read_bytes() == b"keep me"


def test_a_manifest_that_is_not_utf8_is_one_error_naming_it(tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.sha256"
    manifest.write_bytes(b"\xe9\xff not utf-8\n")
    with pytest.raises(ValueError, match=r"^MANIFEST\.sha256: not valid UTF-8$"):
        extraction.verify_extraction(tmp_path, manifest)


def test_an_archive_that_is_a_folder_is_named_as_a_folder_before_its_size_is_read(tmp_path: Path) -> None:
    (tmp_path / "adir").mkdir()
    with pytest.raises(IsADirectoryError) as caught:
        extraction.extract_archive(tmp_path / "adir", tmp_path / "raw")
    assert str(caught.value) == "adir: is a folder, not a file"


def test_a_missing_archive_is_named_by_file_name_only(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        extraction.extract_archive(tmp_path / "deep" / "gone.tar.gz", tmp_path / "raw")
    assert str(caught.value) == "gone.tar.gz: archive not found"


def test_a_member_that_is_not_a_regular_file_is_refused_by_name() -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        folder = tarfile.TarInfo("sponspeech/adir")
        folder.type = tarfile.DIRTYPE
        tar.addfile(folder)
    buffer.seek(0)
    with tarfile.open(fileobj=buffer, mode="r:") as tar, pytest.raises(ValueError) as caught:
        extraction._file_bytes(tar, tar.getmembers()[0])
    assert str(caught.value) == "sponspeech/adir: not a regular file"


def test_a_manifest_that_is_not_directly_in_raw_root_is_named_first_and_without_its_folder(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "elsewhere").mkdir()
    manifest = tmp_path / "elsewhere" / "MANIFEST.sha256"
    manifest.write_text("", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        extraction.verify_extraction(tmp_path / "raw", manifest)
    assert str(caught.value) == (
        "MANIFEST.sha256: must be placed directly in raw_root (its paths start with sponspeech_text/)")
