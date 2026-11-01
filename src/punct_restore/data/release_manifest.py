"""Manifest and provenance of a data release (owned by Member 1): what the release is and where its data came from."""
import json
import os
import platform
import shutil
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

from ..hashing import sha256_file
from . import extraction, preprocessing, splits, validation
from .release_constants import RELEASE_FILES, SOURCE_REFERENCE


def describe_files(folder: str | os.PathLike[str],
                   skip: Collection[str] = ("release_manifest.json", "SHA256SUMS")) -> dict[str, dict[str, str | int]]:
    """{relative path: {sha256, bytes, lines}} for every file below `folder`."""
    out = {}
    for d, _, fs in os.walk(folder):
        for x in fs:
            p = os.path.join(d, x)
            rel = Path(p).relative_to(folder).as_posix()
            if rel in skip:
                continue
            with open(p, "rb") as f:
                lines = sum(1 for _ in f)
            out[rel] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p), "lines": lines}
    return dict(sorted(out.items()))


def copy_license(raw_dir: str, out: str) -> None:
    """License and attribution travel with the release (CC BY-NC 4.0 requires attribution)."""
    license_src = os.path.join(raw_dir, "README.txt")
    if os.path.exists(license_src):
        shutil.copyfile(license_src, os.path.join(out, "SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt"))


@dataclass(frozen=True)
class ManifestInputs:
    """What the release manifest records about one build; `code_sha256` is called when the manifest is built."""

    release_id: str
    label_sets_path: str
    raw_manifest: str | None
    titles_used: bool
    code_sha256: Callable[[], str]
    source_count: int
    released_count: int


def build_manifest(inputs: ManifestInputs) -> dict:
    """The release manifest without `provenance` and the file inventories."""
    return {"release_id": inputs.release_id, "status": "final candidate: policies D1-D6 approved by the team lead",
            "schema_version": validation.SCHEMA_VERSION, "policy_version": preprocessing.POLICY_VERSION,
            "source": {"id": preprocessing.SOURCE_ID, "revision": preprocessing.SOURCE_REVISION,
                       "license": "CC BY-NC 4.0", "reference": SOURCE_REFERENCE, "decision": "D1 approved"},
            "annotation_views": {"canonical": "mode_A, mode_B, mode_C (D3, D5 approved projection)",
                                 "native": "benchmark_native/mode_A (SponSpeech four-class view, comparison only)"},
            "split_mapping": splits.SPLIT_OF, "label_sets_sha256": sha256_file(inputs.label_sets_path),
            "code_sha256": inputs.code_sha256(), "python": platform.python_version(),
            "raw_manifest_sha256": sha256_file(inputs.raw_manifest) if inputs.raw_manifest else None,
            "titles_used_for_tts_check": inputs.titles_used,
            "records": {"source": inputs.source_count, "released_per_view": inputs.released_count,
                        "rejected": inputs.source_count - inputs.released_count},
            "manifest_revision": 2}


def build_provenance(raw_manifest_sha256: str | None, titles_path: str | None, contexts: dict,
                     by_file: dict) -> dict:
    """Where the data came from: archive, raw manifest and the dated metadata snapshot."""
    tts_videos = sorted(v for v, c in contexts.items() if "tts_source" in c["flags"])
    return {
        "archive": {"url": extraction.ARCHIVE_URL, "bytes": extraction.ARCHIVE_BYTES,
                    "gcs_generation": extraction.ARCHIVE_GENERATION, "extraction": "punct_restore.data.extraction",
                    "head_range": extraction.HEAD_RANGE, "text_ranges": extraction.TEXT_RANGES},
        "raw_manifest": {"file": "MANIFEST.sha256", "sha256": raw_manifest_sha256},
        "metadata_snapshot": {
            "file": "video_metadata_oembed.jsonl", "sha256": sha256_file(titles_path) if titles_path else None,
            "fetched": "2026-10-28 (YouTube oEmbed; do not re-fetch, titles can change)",
            "use": "a video whose title ends with '| Wikipedia audio article' is flagged tts_source",
            "tts_videos": tts_videos,
            "tts_records_by_file": {n: sum("tts_source" in r["unresolved"]["A"] for r in by_file[n])
                                    for n in RELEASE_FILES}}}


def with_file_inventory(manifest: dict, files: dict, out: str, split_out: str) -> dict:
    """The manifest plus the sha256, size and line count of every released file (views also get records, valid)."""
    view_info = {p: {"records": v["records"], "valid": v.get("valid")} for p, v in files.items()}
    return {**manifest,
            "files": {p: {**d, **view_info.get(p, {})} for p, d in describe_files(out).items()},
            "split_files": describe_files(split_out)}


def write_manifest(out: str, manifest: dict) -> None:
    """release_manifest.json (LF line endings, sorted keys)."""
    with open(os.path.join(out, "release_manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
