"""Recreate the SponSpeech transcripts and build a data release.

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
from punct_restore.data.release import ReleaseSources, build_release_from
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import one_line_error


def run_extract(args: argparse.Namespace) -> None:
    if args.archive:
        extraction.extract_archive(args.archive, args.raw_root, log=print)
    else:
        extraction.extract_ranges(args.raw_root, log=print)
    problems, extra = extraction.verify_extraction(args.raw_root, os.path.join(args.raw_root, "MANIFEST.sha256"))
    if problems or extra:
        raise ValueError(f"extraction does not match MANIFEST.sha256: {len(problems)} missing/different, "
                         f"{len(extra)} extra (first: {(problems or extra)[:3]})")
    print("extraction matches MANIFEST.sha256")


def run_build(args: argparse.Namespace) -> None:
    manifest = os.path.join(args.raw_root, "MANIFEST.sha256")
    titles = os.path.join(args.raw_root, "video_metadata_oembed.jsonl")
    for path in (manifest, titles):
        if not os.path.exists(path):   # without them the build is not the published release
            raise ValueError(f"{os.path.basename(path)}: not found; copy it from the release's provenance package")
    raw_dir = os.path.join(args.raw_root, extraction.OUTPUT_DIR)
    if not os.path.isdir(raw_dir):
        raise ValueError(
            f"{extraction.OUTPUT_DIR}: folder not found in the raw root; run recap prepare-data extract first")
    sources = ReleaseSources(raw_dir=raw_dir,
                             processed_dir=args.processed_dir, splits_dir=args.splits_dir,
                             release_id=args.release_id, raw_manifest=manifest, titles_path=titles)
    result = build_release_from(sources, log=print)
    print(json.dumps(result["records"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recap prepare-data", description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract", help="recreate the raw transcripts from the official archive")
    extract_parser.add_argument("--raw-root", required=True)
    extract_parser.add_argument("--archive", help="local copy of sponspeech.tar.gz (full-archive fallback)")
    build_parser = subparsers.add_parser("build", help="build a release from the raw transcripts")
    build_parser.add_argument("--raw-root", required=True)
    build_parser.add_argument("--release-id", required=True)
    build_parser.add_argument("--processed-dir", default=os.path.join(ROOT, "data", "processed"))
    build_parser.add_argument("--splits-dir", default=os.path.join(ROOT, "data", "splits"))
    args = parser.parse_args(argv)
    try:
        if args.command == "extract":
            run_extract(args)
        else:
            run_build(args)
    except (OSError, ValueError, extraction.SourceChanged) as exc:
        print(f"error: {one_line_error(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
