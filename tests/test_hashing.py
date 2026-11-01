"""punct_restore.hashing: the one streaming SHA-256 for files and for archive members."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from punct_restore.hashing import sha256_file, sha256_stream

BLOCK = 1 << 20


def _payload(size: int) -> bytes:
    """Bytes with CRLF, a NUL and high bytes, so text-mode reading or newline translation changes the digest."""
    return (b"a\r\nb\n\x00\xff\xfe" * (size // 8 + 1))[:size]


@pytest.mark.parametrize("size", [0, 1, BLOCK - 1, BLOCK, BLOCK + 1, 3 * BLOCK + 17])
def test_sha256_file_equals_hashlib_on_the_whole_file_across_block_boundaries(tmp_path: Path, size: int) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(_payload(size))
    assert sha256_file(path) == hashlib.sha256(_payload(size)).hexdigest()


def test_sha256_file_accepts_a_str_path(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc")
    assert sha256_file(str(path)) == hashlib.sha256(b"abc").hexdigest()


def test_sha256_file_of_a_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "ghost.bin")


class _RecordingStream(io.RawIOBase):
    """A binary stream that remembers how many bytes each read asked for."""

    def __init__(self, data: bytes) -> None:
        self._inner = io.BytesIO(data)
        self.asked: list[int] = []

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        self.asked.append(size)
        return self._inner.read(size)


def test_sha256_stream_reads_in_blocks_never_the_whole_stream() -> None:
    stream = _RecordingStream(_payload(2 * BLOCK + 5))
    assert sha256_stream(stream) == hashlib.sha256(_payload(2 * BLOCK + 5)).hexdigest()
    assert len(stream.asked) >= 3 and all(0 < size <= BLOCK for size in stream.asked)


def test_sha256_stream_hashes_an_archive_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("top/member.bin", _payload(BLOCK + 123))
    with zipfile.ZipFile(archive_path) as archive, archive.open("top/member.bin") as member:
        assert sha256_stream(member) == hashlib.sha256(_payload(BLOCK + 123)).hexdigest()
