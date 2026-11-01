"""The one SHA-256 for files and archive members, read in blocks so a large file never sits in memory."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO

BLOCK_SIZE = 1 << 20


def sha256_stream(handle: BinaryIO) -> str:
    """Hex SHA-256 of what is left of a binary stream, read in blocks."""
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(BLOCK_SIZE), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    with open(path, "rb") as handle:
        return sha256_stream(handle)
