"""Build a word-cluster map for the CRF from the default-train records of one label mode.

Usage:
    recap build-word-clusters [--data-root data] [--release sponspeech-m1-v2] [--mode {A,B,C}] \\
        --k K [--seed 2026] [--min-count 5] [--dims 100] [--window 2] --out MAP.json

Reads only ``splits/<rel>/default_train/mode_<M>.ids`` and the matching records of
``processed/<rel>/mode_<M>/train.jsonl`` (streamed); validation and test files are never opened. The map
(see ``punct_restore.features.word_clusters``) records the SHA-256 of the ID file. ``--out`` must not exist.
Exit codes: 0 done, 1 one ``error:`` line on stderr, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from punct_restore.features.word_clusters import (
    ClusterMap,
    induce_clusters,
    map_to_json,
)
from punct_restore.hashing import sha256_file
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import one_line_error, read_ids, reconfigure_stdout, write_new_file


def _parse(line: str, name: str, number: int) -> tuple[str, dict[str, Any]]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} line {number}: invalid JSON ({exc.msg})") from exc
    sample_id = record.get("sample_id") if isinstance(record, dict) else None
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{name} line {number}: record without a string sample_id")
    return sample_id, record


def _tokens(record: dict[str, Any], name: str, number: int) -> list[str]:
    tokens = record.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise ValueError(f"{name} line {number}: tokens must be a list of strings")
    return tokens


def train_tokens(path: Path, wanted: set[str]) -> Iterator[list[str]]:
    """Stream the token list of every wanted record of the JSONL file; fail at the end if some are missing."""
    seen: set[str] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                sample_id, record = _parse(line, path.name, number)
                if sample_id not in wanted:
                    continue
                if sample_id in seen:
                    raise ValueError(f"{path.name} line {number}: duplicate sample_id")
                seen.add(sample_id)
                yield _tokens(record, path.name, number)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: not valid UTF-8") from exc
    if len(seen) != len(wanted):
        raise ValueError(f"{path.name}: {len(wanted) - len(seen)} ID(s) of the ids file not found")


def build(args: argparse.Namespace) -> ClusterMap:
    """Read the train data of ``args.mode``, induce the clusters and write the map file."""
    if args.out.exists():
        raise ValueError(f"{args.out.name} already exists; refusing to overwrite")
    ids_path = args.data_root / "splits" / args.release / "default_train" / f"mode_{args.mode}.ids"
    train_path = args.data_root / "processed" / args.release / f"mode_{args.mode}" / "train.jsonl"
    ids, ids_sha256 = set(read_ids(ids_path)), sha256_file(ids_path)
    clusters = induce_clusters(
        train_tokens(train_path, ids), k=args.k, seed=args.seed,
        min_count=args.min_count, dims=args.dims, window=args.window,
    )
    cluster_map = ClusterMap(
        k=args.k, seed=args.seed, min_count=args.min_count,
        dims=min(args.dims, len(clusters) - 1),  # the SVD cannot use more components than the vocabulary allows
        window=args.window, vocabulary_size=len(clusters), train_ids_sha256=ids_sha256, release=args.release,
        mode=args.mode, clusters=clusters,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_new_file(args.out, map_to_json(cluster_map))
    return cluster_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recap build-word-clusters",
                                     description="Build a word-cluster map from the default-train records.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--release", default="sponspeech-m1-v2")
    parser.add_argument("--mode", choices=["A", "B", "C"], default="A")
    parser.add_argument("--k", type=int, required=True, help="number of clusters")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-count", type=int, default=5, help="smallest word count that enters the vocabulary")
    parser.add_argument("--dims", type=int, default=100, help="SVD dimensions")
    parser.add_argument("--window", type=int, default=2, help="co-occurrence window on each side")
    parser.add_argument("--out", type=Path, required=True, help="new map file (must not exist)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: 0 on success, 1 with one ``error:`` line on stderr, 2 on usage errors."""
    reconfigure_stdout()
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    try:
        cluster_map = build(args)
    except (OSError, ValueError) as exc:
        print(f"error: {one_line_error(exc)}", file=sys.stderr)
        return 1
    seconds = time.perf_counter() - started
    print(f"vocabulary_size={cluster_map.vocabulary_size} k={cluster_map.k} seconds={seconds:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
