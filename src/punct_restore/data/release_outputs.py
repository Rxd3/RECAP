"""Writers for the files of a data release (owned by Member 1): views, masks, audit trail, split lists, reports.

`write_lines` is the single place every line-oriented release file goes through (LF endings, folders created), so a
test can wrap it to see or damage every such file.
"""
import collections
import json
import os
from collections.abc import Iterable
from pathlib import Path

from ..hashing import sha256_file
from . import splits
from .release_constants import JSON, RELEASE_FILES
from .release_notes import release_notes_lines


def write_lines(path: str, lines: Iterable[str]) -> None:
    """Write `lines` (each ending in its newline) to `path` with LF endings, creating the folder."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)


def write_views(out: str, by_file: dict) -> dict:
    """Write the four views (rejected records are never written); returns {relative path: {"records": n}}."""
    files = {}
    for folder, key in (("mode_A", "A"), ("mode_B", "B"), ("mode_C", "C"), ("benchmark_native/mode_A", "native_A")):
        for name in RELEASE_FILES:
            path = os.path.join(out, folder, name + ".jsonl")
            write_lines(path, (json.dumps(r["views"][key], **JSON) + "\n" for r in by_file[name]))
            files[Path(path).relative_to(out).as_posix()] = {"records": len(by_file[name])}
    return files


def write_masks(out: str, by_file: dict) -> None:
    """Casing/joint evaluation masks: one line per released record, positions excluded from casing and joint metrics."""
    for name in RELEASE_FILES:
        write_lines(os.path.join(out, "masks", "case_joint", name + ".jsonl"),
                    (json.dumps({"sample_id": r["sample_id"], "excluded_positions": r["case_masked"],
                                 "reason": "case_mixed_approximation" if r["case_masked"] else None}, **JSON) + "\n"
                     for r in by_file[name]))


def write_audit(out: str, results: list, kept: list) -> None:
    """Audit trail: rejected, unresolved (kept in the views) and every normalization removal."""
    write_lines(os.path.join(out, "audit", "rejected.jsonl"),
                (json.dumps({k: r.get(k) for k in ("sample_id", "source_split", "group_id", "reasons",
                                                    "validation_errors", "original_text")}, **JSON) + "\n"
                 for r in results if r["status"] == "rejected"))
    write_lines(os.path.join(out, "audit", "unresolved.jsonl"),
                (json.dumps({"sample_id": r["sample_id"], "source_split": r["source_split"],
                             "unresolved_by_mode": r["unresolved"], "token_positions": r["positions"]}, **JSON) + "\n"
                 for r in kept if r["status"] == "flagged"))
    write_lines(os.path.join(out, "audit", "normalization.jsonl"),
                (json.dumps({"sample_id": r["sample_id"], "removals": r["removals"]}, **JSON) + "\n"
                 for r in results if r["removals"]))


def write_split_map(split_out: str, results: list) -> None:
    """split_map.tsv: one row per source record, rejected ones included."""
    write_lines(os.path.join(split_out, "split_map.tsv"),
                ["sample_id\tsplit\tsource_split\tfile\tgroup_id\tstatus\n"] +
                [f"{r['sample_id']}\t{splits.SPLIT_OF[r['source_split']]}\t{r['source_split']}\t{r['file']}\t"
                 f"{r['group_id']}\t{r['status']}\n" for r in results])


def _screen_source_split(s: str, refs: tuple, by_source: dict) -> tuple:
    """(reason counts, kept records, excluded-TSV rows) for source split `s` against its reference splits."""
    ref = splits.Reference([r for name in refs for r in by_source[name]])
    reasons: collections.Counter[str] = collections.Counter()
    keep: list = []
    dropped: list[str] = []
    for r in by_source[s]:
        why = splits.overlap_reason(r, ref)
        if why is None and {"low_comma_video", "tts_source"} & set(r["unresolved"]["A"]):
            why = "low_comma_or_tts_source"
        if why:
            reasons[why] += 1
            dropped.append(f"{r['sample_id']}\t{why}\n")
        else:
            keep.append(r)
    return reasons, keep, dropped


def write_leakage_controlled(split_out: str, by_source: dict) -> tuple[dict, dict]:
    """Frozen leakage-controlled subsets: writes <file>.ids and <file>_excluded.tsv; returns (records, report)."""
    controlled, controlled_report = {}, {}
    for s, refs in splits.DISJOINT_REFERENCE.items():
        reasons, keep, dropped = _screen_source_split(s, refs, by_source)
        name = splits.FILE_OF[s]
        controlled[name] = keep
        controlled_report[name] = {"reference": [splits.FILE_OF[x] for x in refs],
                                   "official_records": len(by_source[s]),
                                   "kept": len(keep), "kept_videos": len({r["group_id"] for r in keep}),
                                   "dropped_by_reason": dict(sorted(reasons.items()))}
        write_lines(os.path.join(split_out, "leakage_controlled", name + ".ids"),
                    (r["sample_id"] + "\n" for r in keep))
        write_lines(os.path.join(split_out, "leakage_controlled", name + "_excluded.tsv"),
                    ["sample_id\treason\n"] + dropped)
    return controlled, controlled_report


def write_id_lists(split_out: str, default: dict) -> None:
    """Default training and primary evaluation ID lists: only records resolved for the mode."""
    for mode in ("A", "B", "C"):
        write_lines(os.path.join(split_out, "default_train", f"mode_{mode}.ids"),
                    (r["sample_id"] + "\n" for r in default["train"] if not r["unresolved"][mode]))
        for name in ("validation", "test", "test_amb"):
            write_lines(os.path.join(split_out, "primary_eval", f"{name}_mode_{mode}.ids"),
                        (r["sample_id"] + "\n" for r in default[name] if not r["unresolved"][mode]))


def write_json_reports(out: str, split_out: str, stats: dict, controlled_report: dict, overlap: dict) -> None:
    """stats.json and the leakage-controlled report.json (LF line endings, sorted keys)."""
    with open(os.path.join(out, "stats.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=1, sort_keys=True)
    with open(os.path.join(split_out, "leakage_controlled", "report.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump({"subsets": controlled_report, "overlap": overlap}, f, indent=1, sort_keys=True)


def write_release_notes(out: str, release_id: str, stats: dict, manifest: dict) -> None:
    """Short human-readable notes stored with the release (all numbers come from stats)."""
    write_lines(os.path.join(out, "RELEASE_NOTES.md"), release_notes_lines(release_id, stats, manifest))


def write_checksums(folders: tuple[str, ...]) -> None:
    """SHA256SUMS in every folder, listing every other file below it."""
    for folder in folders:
        paths = sorted(Path(d, x).relative_to(folder).as_posix()
                       for d, _, fs in os.walk(folder) for x in fs if x != "SHA256SUMS")
        write_lines(os.path.join(folder, "SHA256SUMS"),
                    (f"{sha256_file(os.path.join(folder, p))}  {p}\n" for p in paths))
