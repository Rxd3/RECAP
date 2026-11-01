"""Recreate the SponSpeech transcripts and build a data release (owned by Member 1).

Run from the repository root. RAW is data/raw/sponspeech-gcs-generation-1716975130398953, containing the provenance
files MANIFEST.sha256 and video_metadata_oembed.jsonl from the release's provenance package.

    recap prepare-data extract --raw-root RAW                       # pinned byte ranges (~97 MB)
    recap prepare-data extract --raw-root RAW --archive sponspeech.tar.gz   # full-archive fallback
    recap prepare-data build --raw-root RAW --release-id sponspeech-m1-v2

`extract` writes RAW/sponspeech_text/ and fails unless every file matches RAW/MANIFEST.sha256. `build` writes
data/processed/<release-id>/ and data/splits/<release-id>/ (ignored by Git) following the approved policies D1-D6
(docs/DECISIONS.md). See data/README.md.
"""
import argparse
import json
import os
import sys

from punct_restore.data import extraction
from punct_restore.data.release import build_release
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import one_line_error


def run_extract(a):
    if a.archive:
        extraction.extract_archive(a.archive, a.raw_root)
    else:
        extraction.extract_ranges(a.raw_root)
    problems, extra = extraction.verify_extraction(a.raw_root, os.path.join(a.raw_root, "MANIFEST.sha256"))
    if problems or extra:
        sys.exit(f"extraction does not match MANIFEST.sha256: {len(problems)} missing/different, {len(extra)} extra "
                 f"(first: {(problems or extra)[:3]})")
    print("extraction matches MANIFEST.sha256")


def run_build(a):
    manifest = os.path.join(a.raw_root, "MANIFEST.sha256")
    titles = os.path.join(a.raw_root, "video_metadata_oembed.jsonl")
    for path in (manifest, titles):
        if not os.path.exists(path):   # without them the build is not the published release
            sys.exit(f"missing {os.path.basename(path)}: copy it from the release's provenance package")
    result = build_release(os.path.join(a.raw_root, extraction.OUTPUT_DIR), a.processed_dir, a.splits_dir, a.release_id,
                           raw_manifest=manifest, titles_path=titles)
    print(json.dumps(result["records"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recap prepare-data", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract", help="recreate the raw transcripts from the official archive")
    ex.add_argument("--raw-root", required=True)
    ex.add_argument("--archive", help="local copy of sponspeech.tar.gz (full-archive fallback)")
    bd = sub.add_parser("build", help="build a release from the raw transcripts")
    bd.add_argument("--raw-root", required=True)
    bd.add_argument("--release-id", required=True)
    bd.add_argument("--processed-dir", default=os.path.join(ROOT, "data", "processed"))
    bd.add_argument("--splits-dir", default=os.path.join(ROOT, "data", "splits"))
    a = ap.parse_args(argv)
    try:
        if a.command == "extract":
            run_extract(a)
        else:
            run_build(a)
    except (OSError, ValueError, extraction.SourceChanged) as exc:
        print(f"error: {one_line_error(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
