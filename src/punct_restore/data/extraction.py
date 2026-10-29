"""Recreate the raw SponSpeech transcripts from the official archive (owned by Member 1).

Two methods; both write <raw_root>/sponspeech_text/ and must match the published raw MANIFEST.sha256 exactly.

* ranges (default): downloads only the byte ranges that contain the transcripts (about 97 MB instead of 63.5 GB).
  The archive is one gzip stream, so each range is decoded from a recorded deflate block start with an empty
  32 KiB window; every range must start inside audio data so the window is refilled before the first text file.
  The archive size and GCS generation and the SHA-256 of every range are pinned: any change stops with an error.
* archive (fallback): streams a fully downloaded sponspeech.tar.gz and keeps only the non-audio members.

The ranges and their checksums were measured and verified in Member 1's source inspection (docs/DATASET_PLAN.md).
"""
import hashlib
import io
import os
import subprocess
import tarfile
import urllib.error
import urllib.request
import zlib
from pathlib import Path

ARCHIVE_URL = "https://storage.googleapis.com/sponspeech/sponspeech.tar.gz"
ARCHIVE_BYTES = 63547617096
ARCHIVE_GENERATION = "1716975130398953"
# First 20 MiB: decoded normally from byte 0; holds sponspeech/train/utt2spk.
HEAD_RANGE = {"start": 0, "length": 20971520,
              "sha256": "af20d9b74f34de479ad858e6d37d5e817f0e984f993135a8edf2983c555ee9b8"}
# Text sections at the end of each split: (range start, length, SHA-256, deflate block byte, bit offset).
TEXT_RANGES = [
    {"name": "train", "start": 44774419920, "length": 33000000, "deflate_byte": 44774431736, "bit": 2,
     "sha256": "913d478fa6843f56737caea7cf8faac50d0031a345c94151cd626cc08a27a8f8"},
    {"name": "dev", "start": 52292099325, "length": 18000000, "deflate_byte": 52292131665, "bit": 1,
     "sha256": "8ab19b0cf72759215e730ea92ef827ffda1d8c3c401833057003fc3b40e80661"},
    {"name": "test-amb", "start": 58007690922, "length": 17000000, "deflate_byte": 58007693111, "bit": 5,
     "sha256": "2d62ec2ab0a17470bff97ba69603258fcd9387fad02d43dee531b5cecd10b702"},
    {"name": "test", "start": 63539617096, "length": 8000000, "deflate_byte": 63539644808, "bit": 2,
     "sha256": "8223f5f6804818b2011de5c06ebe0bdee0ed98459abaa0f6eaf52d8376b65070"},
]
WINDOW = 32768
OUTPUT_DIR = "sponspeech_text"


class SourceChanged(RuntimeError):
    """The remote archive or a downloaded range no longer matches the pinned values."""


def _http(url, headers=None, method="GET"):
    """Return (headers, body). Falls back to the curl command if Python cannot verify TLS on this machine."""
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.URLError as e:
        if "CERTIFICATE_VERIFY_FAILED" not in str(e):
            raise
    args = ["curl", "-sSf", "-D", "-", "-o", "-"] + (["-I"] if method == "HEAD" else [])
    args += [x for k, v in (headers or {}).items() for x in ("-H", f"{k}: {v}")] + [url]
    out = subprocess.run(args, capture_output=True, check=True).stdout
    head, _, body = out.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")[1:]
    return {k.strip().lower(): v.strip() for k, _, v in (l.partition(":") for l in lines)}, body


def check_remote(url=ARCHIVE_URL):
    """Stop unless the remote object has the pinned size and generation."""
    headers, _ = _http(url, method="HEAD")
    size, gen = headers.get("content-length"), headers.get("x-goog-generation")
    if size != str(ARCHIVE_BYTES) or gen != ARCHIVE_GENERATION:
        raise SourceChanged(f"archive changed: size {size}, generation {gen}; expected {ARCHIVE_BYTES}, {ARCHIVE_GENERATION}")


def fetch_range(start, length, sha256, url=ARCHIVE_URL):
    """Download one byte range and verify it against its pinned SHA-256."""
    _, body = _http(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    if hashlib.sha256(body).hexdigest() != sha256:
        raise SourceChanged(f"range {start}+{length}: checksum differs from the pinned value")
    return body


def _shift(buf, bit):
    return (int.from_bytes(buf, "little") >> bit).to_bytes(len(buf), "little")


def decode_from(buf, byte_offset, bit):
    """Raw-deflate decode from a block start inside `buf`, with an all-zero preset window."""
    data = _shift(buf[byte_offset:], bit)
    d = zlib.decompressobj(wbits=-15, zdict=b"\0" * WINDOW)
    out = bytearray()
    for k in range(0, len(data), 1 << 16):
        try:
            out += d.decompress(data[k:k + (1 << 16)])
        except zlib.error:
            break            # the range ends inside the next block
    return bytes(out)


def _valid_header(h):
    if len(h) < 512 or h[257:262] != b"ustar":
        return False
    try:
        stored = int(h[148:156].rstrip(b"\0 ").decode() or "x", 8)
    except ValueError:
        return False
    return stored == sum(h[:148]) + 32 * 8 + sum(h[156:512])


def members_from_decoded(out):
    """(name, data) of complete regular members, walking checksum-valid tar headers after the first 32 KiB."""
    h = WINDOW
    while h + 512 <= len(out) and not _valid_header(out[h:h + 512]):
        h += 1
    while h + 512 <= len(out) and _valid_header(out[h:h + 512]):
        hdr = out[h:h + 512]
        name = hdr[0:100].rstrip(b"\0").decode()
        prefix = hdr[345:500].rstrip(b"\0").decode()
        name = f"{prefix}/{name}" if prefix else name
        size = int(hdr[124:136].rstrip(b"\0 ").decode() or "0", 8)
        data = out[h + 512:h + 512 + size]
        if len(data) < size:
            break
        if hdr[156:157] in (b"0", b"\0"):
            yield name, data
        h += 512 + ((size + 511) // 512) * 512


def _keep(name):
    """Non-audio members below sponspeech/: transcripts, utt2spk, README.txt, .build."""
    return name.startswith("sponspeech/") and not name.endswith(".wav") and not name.endswith("/")


def _write(raw_root, name, data):
    path = os.path.join(raw_root, OUTPUT_DIR, name[len("sponspeech/"):])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def extract_ranges(raw_root, url=ARCHIVE_URL, fetch=None, log=print):
    """Recreate the transcripts from pinned byte ranges. Returns the number of files written."""
    if fetch is None:                                  # real download: verify the remote object first
        check_remote(url)
        fetch = lambda start, length, sha: fetch_range(start, length, sha, url)
    written = set()
    head = fetch(HEAD_RANGE["start"], HEAD_RANGE["length"], HEAD_RANGE["sha256"])
    with tarfile.open(fileobj=io.BytesIO(head), mode="r|gz") as tf:
        try:
            for m in tf:
                if m.isfile() and m.name == "sponspeech/train/utt2spk":
                    _write(raw_root, m.name, tf.extractfile(m).read())
                    written.add(m.name)
                    break
        except (tarfile.ReadError, EOFError, zlib.error):
            pass
    for r in TEXT_RANGES:
        buf = fetch(r["start"], r["length"], r["sha256"])
        n = 0
        for name, data in members_from_decoded(decode_from(buf, r["deflate_byte"] - r["start"], r["bit"])):
            if _keep(name):
                _write(raw_root, name, data)
                written.add(name)
                n += 1
        log(f"range {r['name']}: {n} files")
    return len(written)


def extract_archive(archive_path, raw_root, log=print):
    """Fallback: stream a fully downloaded sponspeech.tar.gz and keep only the non-audio members."""
    size = os.path.getsize(archive_path)
    if size != ARCHIVE_BYTES:
        raise SourceChanged(f"{archive_path} has {size} bytes; expected {ARCHIVE_BYTES}")
    n = 0
    with tarfile.open(archive_path, mode="r|gz") as tf:
        for m in tf:
            if m.isfile() and _keep(m.name):
                _write(raw_root, m.name, tf.extractfile(m).read())
                n += 1
    log(f"archive: {n} files")
    return n


def verify_extraction(raw_root, manifest_path):
    """Compare <raw_root>/sponspeech_text with the raw manifest: (missing or different paths, extra paths)."""
    from .preprocessing import verify_raw_manifest
    base = os.path.dirname(os.path.abspath(manifest_path))
    if os.path.abspath(base) != os.path.abspath(raw_root):
        raise ValueError("MANIFEST.sha256 must be placed directly in raw_root (its paths start with sponspeech_text/)")
    with open(manifest_path, encoding="utf-8") as f:
        listed = {line.rstrip("\n").split("  ", 1)[1] for line in f}
    present = {Path(d, x).relative_to(raw_root).as_posix()
               for d, _, fs in os.walk(os.path.join(raw_root, OUTPUT_DIR)) for x in fs}
    missing = sorted(listed - present)
    bad = [] if missing else verify_raw_manifest(manifest_path)
    return missing + bad, sorted(present - listed)
