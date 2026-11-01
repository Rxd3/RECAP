"""Find gold records that carry non-speech material (speaker labels, SRT cue residue, auto-caption videos).

Usage:
    recap find-gold-noise [--data-root data] [--release sponspeech-m1-v2] [--mode B] --out DIR

Reads ``processed/<rel>/mode_B/{validation,test,test_amb,train}.jsonl``, the primary ID lists
``splits/<rel>/primary_eval/<split>_mode_B.ids`` and ``splits/<rel>/default_train/mode_B.ids``. The rules are
documented in ``punct_restore.data.gold_noise``: speaker labels are looked for in ``metadata.normalized_text`` (the text
``metadata.offsets`` index), cue residue in ``metadata.original_text``. The per-video speaker inventory is built from
every record of the split file, the counts and the table cover only the listed records.

Writes to the new directory DIR, per split in validation, test and test_amb:
    <split>_B_noisy.tsv      headerless ``sample_id<TAB>rule``, input-file order, one row per rule of a record
    <split>_B_summary.json   records, records_per_rule, records_flagged, gold_colon_labels, speaker_label_colons,
                             speaker_label_colon_share (null without COLON labels), release, mode, split and the
                             SHA-256 of every input
and for the default-train list only ``default_train_B_summary.json`` (train.jsonl is streamed, never held in memory).
DIR must not exist and its parent must be an existing folder (so must the data root); both are checked before any
scoring. DIR is built in a temporary sibling folder and renamed at the end.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

from punct_restore.data.gold_noise import (
    RULES,
    build_inventories,
    rules_for,
    speaker_label_colons,
    validate_record,
)
from punct_restore.file_io import iter_jsonl, require_folder, write_new_folder
from punct_restore.hashing import sha256_file
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import one_line_error, parse_json_object, read_ids

SPLITS = ("validation", "test", "test_amb", "default_train")


class Scored(NamedTuple):
    """One listed record: its rules, its COLON gold labels and how many of them sit on a speaker label."""

    sample_id: str
    rules: list[str]
    colons: int
    speaker_colons: int


def _checked_record(line: str, where: str) -> dict[str, Any]:
    """The JSON object on ``line``, checked by ``validate_record``; the ValueError starts with ``where``."""
    record = parse_json_object(line, where)
    try:
        validate_record(record)
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from exc
    return record


def _records(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Stream ``(sample_id, record)`` per JSONL line; errors name the file and line."""
    for _, record in iter_jsonl(path, _checked_record):
        yield record["sample_id"], record


def _unique_records(path: Path) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    for sample_id, record in _records(path):
        if sample_id in seen:
            raise ValueError(f"{path.name}: ID {sample_id} is listed twice")
        seen.add(sample_id)
        yield record


def score_records(jsonl: Path, ids_path: Path) -> Iterator[Scored]:
    """Score the listed records in file order; the speaker inventory comes from every record of the file."""
    ids = read_ids(ids_path)
    wanted = set(ids)
    inventories = build_inventories(_unique_records(jsonl))                      # pass 1: the whole file
    found: set[str] = set()
    for sample_id, record in _records(jsonl):                                    # pass 2: the listed records
        if sample_id not in wanted:
            continue
        found.add(sample_id)
        inventory = inventories.get(record["metadata"]["group_id"], frozenset())
        yield Scored(sample_id, rules_for(record, inventory), record["punct_labels"].count("COLON"),
                     speaker_label_colons(record, inventory))
    if len(found) != len(wanted):
        missing = next(sample_id for sample_id in ids if sample_id not in found)
        raise ValueError(f"{ids_path.name}: ID {missing} is not in {jsonl.name}")


def summarize(scored: Iterable[Scored]) -> dict[str, Any]:
    """Counts over the scored records."""
    per_rule = dict.fromkeys(RULES, 0)
    records = flagged = colons = speaker_colons = 0
    for row in scored:
        records += 1
        flagged += bool(row.rules)
        colons += row.colons
        speaker_colons += row.speaker_colons
        for rule in row.rules:
            per_rule[rule] += 1
    share = round(speaker_colons / colons, 6) if colons else None
    return {"records": records, "records_per_rule": per_rule, "records_flagged": flagged,
            "gold_colon_labels": colons, "speaker_label_colons": speaker_colons,
            "speaker_label_colon_share": share}


def input_paths(root: Path, release: str, mode: str, split: str) -> tuple[Path, Path]:
    """The JSONL file and the ID list that define ``split``."""
    processed = root / "processed" / release / f"mode_{mode}"
    if split == "default_train":
        return processed / "train.jsonl", root / "splits" / release / "default_train" / f"mode_{mode}.ids"
    return processed / f"{split}.jsonl", root / "splits" / release / "primary_eval" / f"{split}_mode_{mode}.ids"


def process_split(root: Path, release: str, mode: str, split: str) -> dict[str, str]:
    """The output files of one split as ``{file name: text}``; nothing is written yet."""
    jsonl, ids_path = input_paths(root, release, mode, split)
    scored = score_records(jsonl, ids_path)
    rows = list(scored) if split != "default_train" else None
    counts = summarize(rows if rows is not None else scored)
    summary = {"release": release, "mode": mode, "split": split, **counts,
               "input_sha256": {p.relative_to(root).as_posix(): sha256_file(p) for p in (jsonl, ids_path)}}
    files = {f"{split}_B_summary.json": json.dumps(summary, indent=2) + "\n"}
    if rows is not None:
        files[f"{split}_B_noisy.tsv"] = "".join(f"{row.sample_id}\t{rule}\n" for row in rows for rule in row.rules)
    return files


def check_folders(data_root: Path, out: Path) -> None:
    """Refuse a data root that is not a folder, or an ``out`` whose parent is not an existing folder."""
    require_folder(data_root, "data root")
    require_folder(out.parent, "output parent")


def build(data_root: Path, release: str, mode: str, out: Path) -> list[str]:
    """Score every split, then write the outputs; returns one status line per split."""
    check_folders(data_root, out)
    if out.exists():
        raise ValueError(f"output path {out.name} already exists; refusing to overwrite")
    files: dict[str, str] = {}
    lines = []
    for split in SPLITS:
        made = process_split(data_root, release, mode, split)
        files.update(made)
        counts = json.loads(made[f"{split}_B_summary.json"])
        lines.append(f"{split} records={counts['records']} flagged={counts['records_flagged']} "
                     f"speaker_label_colons={counts['speaker_label_colons']}/{counts['gold_colon_labels']} "
                     f"share={counts['speaker_label_colon_share']}")
    write_new_folder(out, files)
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recap find-gold-noise",
                                     description="Flag gold records with speaker labels, cue residue or auto captions.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--release", default="sponspeech-m1-v2")
    parser.add_argument("--mode", choices=["B"], default="B", help="only mode B has a COLON label")
    parser.add_argument("--out", type=Path, required=True, help="new directory (must not exist)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: 0 on success, 1 with one ``error:`` line on stderr, 2 on usage errors."""
    args = build_parser().parse_args(argv)
    try:
        lines = build(args.data_root, args.release, args.mode, args.out)
    except (OSError, ValueError) as exc:
        print(f"error: {one_line_error(exc)}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
