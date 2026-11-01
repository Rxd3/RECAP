"""Split an evaluation split into leakage subsets (kept / excluded by cause) as ID lists.

Usage:
    recap build-leakage-subsets --data-root data --release sponspeech-m1-v2 \\
        --mode A --split test --out DIR

Reads ``processed/<rel>/mode_<M>/<split>.jsonl``, ``splits/<rel>/leakage_controlled/<split>.ids`` and
``<split>_excluded.tsv`` plus the default-train IDs, and writes to the new directory DIR:
``all.ids``, ``kept.ids``, ``excluded_train_video.ids`` (excluded, video also in default train),
``excluded_other_leak.ids``, ``excluded_quality.ids`` (reason ``low_comma_or_tts_source``) and
``counts.json``. Every ``.ids`` file lists one ID per line in the order of ``<split>.jsonl``.
DIR must not exist; it is created only after all inputs were read and checked.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from punct_restore.file_io import iter_jsonl, write_new_folder
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import one_line_error, parse_sample_record, read_ids

QUALITY_REASON = "low_comma_or_tts_source"
BUCKETS = ("excluded_train_video", "excluded_other_leak", "excluded_quality")


def _records(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Stream ``(sample_id, record)`` per JSONL line; errors name the file and line."""
    for _, record in iter_jsonl(path, parse_sample_record):
        yield record["sample_id"], record


def _group(record: dict[str, Any]) -> str | None:
    meta = record.get("metadata")
    group = meta.get("group_id") if isinstance(meta, dict) else None
    return group if isinstance(group, str) else None


def read_split(path: Path) -> dict[str, str | None]:
    """Map every sample_id of the split file to its group_id (None if absent), in file order."""
    groups: dict[str, str | None] = {}
    for sample_id, record in _records(path):
        if sample_id in groups:
            raise ValueError(f"{path.name}: duplicate sample_id {sample_id}")
        groups[sample_id] = _group(record)
    return groups


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: not valid UTF-8") from exc


def read_excluded(path: Path) -> dict[str, str]:
    """Map sample_id to reason from the ``sample_id<TAB>reason`` file (header required)."""
    lines = _read_text(path).splitlines()
    if not lines or lines[0].split("\t") != ["sample_id", "reason"]:
        raise ValueError(f"{path.name}: expected the header 'sample_id<TAB>reason'")
    reasons: dict[str, str] = {}
    for number, line in enumerate(lines[1:], 2):
        parts = line.split("\t")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"{path.name} line {number}: expected 'sample_id<TAB>reason'")
        if parts[0] in reasons:
            raise ValueError(f"{path.name}: ID {parts[0]} is listed twice")
        reasons[parts[0]] = parts[1]
    return reasons


def partition(
    groups: dict[str, str | None], kept_ids: Sequence[str], reasons: dict[str, str], split: str
) -> tuple[list[str], list[str]]:
    """Return ``(kept, excluded)`` in split order after checking the two ID files against it."""
    for sample_id in kept_ids:
        if sample_id not in groups:
            raise ValueError(f"{split}.ids: ID {sample_id} is not in {split}.jsonl")
    kept_set = set(kept_ids)
    excluded = [sample_id for sample_id in groups if sample_id not in kept_set]
    excluded_set = set(excluded)
    for sample_id in reasons:
        if sample_id not in excluded_set:
            raise ValueError(f"{split}_excluded.tsv: ID {sample_id} is not an excluded ID of {split}.jsonl")
    for sample_id in excluded:
        if sample_id not in reasons:
            raise ValueError(f"{split}_excluded.tsv: excluded ID {sample_id} is not listed")
    return [sample_id for sample_id in groups if sample_id in kept_set], excluded


def train_videos(train_path: Path, ids_path: Path) -> set[str]:
    """Group IDs of the default-train records (streams ``train.jsonl``, holds only IDs)."""
    ids = read_ids(ids_path)
    wanted = set(ids)
    found: set[str] = set()
    videos: set[str] = set()
    for sample_id, record in _records(train_path):
        if sample_id not in wanted:
            continue
        group = _group(record)
        if group is None:
            raise ValueError(f"{train_path.name}: ID {sample_id} has no string metadata.group_id")
        found.add(sample_id)
        videos.add(group)
    for sample_id in ids:
        if sample_id not in found:
            raise ValueError(f"{train_path.name}: ID {sample_id} of {ids_path.name} is missing")
    return videos


def bucket_excluded(
    excluded: Sequence[str], reasons: dict[str, str], groups: dict[str, str | None],
    videos: set[str], split: str,
) -> dict[str, list[str]]:
    """Sort the excluded IDs into quality / train-video / other-leak, keeping their order."""
    buckets: dict[str, list[str]] = {name: [] for name in BUCKETS}
    for sample_id in excluded:
        if reasons[sample_id] == QUALITY_REASON:
            buckets["excluded_quality"].append(sample_id)
            continue
        group = groups[sample_id]
        if group is None:
            raise ValueError(f"{split}.jsonl: ID {sample_id} has no string metadata.group_id")
        buckets["excluded_train_video" if group in videos else "excluded_other_leak"].append(sample_id)
    return buckets


def write_outputs(out: Path, id_lists: dict[str, list[str]], counts: dict[str, Any]) -> None:
    """Write every file into a sibling temp folder, then rename it to ``out``: never a partial ``out``."""
    files = {f"{name}.ids": "".join(f"{sample_id}\n" for sample_id in ids) for name, ids in id_lists.items()}
    write_new_folder(out, {**files, "counts.json": json.dumps(counts, indent=2) + "\n"})


def build(data_root: Path, release: str, mode: str, split: str, out: Path) -> dict[str, Any]:
    """Read and check every input, then write the outputs; returns the counts."""
    if out.exists():
        raise ValueError(f"output path {out.name} already exists; refusing to overwrite")
    processed = data_root / "processed" / release / f"mode_{mode}"
    splits = data_root / "splits" / release
    groups = read_split(processed / f"{split}.jsonl")
    # the release writes this list empty when every record of the split is excluded
    kept_ids = read_ids(splits / "leakage_controlled" / f"{split}.ids", allow_empty=True)
    reasons = read_excluded(splits / "leakage_controlled" / f"{split}_excluded.tsv")
    kept, excluded = partition(groups, kept_ids, reasons, split)
    videos = train_videos(processed / "train.jsonl", splits / "default_train" / f"mode_{mode}.ids")
    buckets = bucket_excluded(excluded, reasons, groups, videos, split)
    id_lists = {"all": list(groups), "kept": kept, **buckets}
    counts = {"release": release, "mode": mode, "split": split, "all": len(groups), "kept": len(kept),
              **{name: len(buckets[name]) for name in BUCKETS}}
    write_outputs(out, id_lists, counts)
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recap build-leakage-subsets",
                                     description="Build leakage-controlled ID subsets of an eval split.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--release", default="sponspeech-m1-v2")
    parser.add_argument("--mode", choices=["A", "B", "C"], default="A")
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--out", type=Path, required=True, help="new directory (must not exist)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: 0 on success, 1 with one ``error:`` line on stderr, 2 on usage errors."""
    args = build_parser().parse_args(argv)
    try:
        counts = build(args.data_root, args.release, args.mode, args.split, args.out)
    except (OSError, ValueError) as exc:
        print(f"error: {one_line_error(exc)}", file=sys.stderr)
        return 1
    print(" ".join(f"{key}={value}" for key, value in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
