"""Archive extraction never writes outside raw_root/sponspeech_text, whatever names the tar entries carry."""

from __future__ import annotations

import io
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
