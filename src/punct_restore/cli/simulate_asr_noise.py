"""Simulate ASR-like word errors in gold data while keeping every label consistent (robustness study).

Usage:
    recap simulate-asr-noise --split {validation,test} --wer W [--seed 2026] --out DIR
        [--edit-model {v1,v2}] [--data-root data] [--release sponspeech-m1-v2] [--masks FILE]

Reads, for each mode M in A, B, C, the records of processed/<release>/mode_M/<split>.jsonl named by
splits/<release>/primary_eval/<split>_mode_M.ids, and the mode C casing masks (default
processed/<release>/masks/case_joint/<split>.jsonl). Writes DIR/mode_M/<split>.jsonl (listed records only,
in file order), DIR/<split>_mode_M.ids (a byte copy), DIR/masks/<split>.jsonl (remapped) and
DIR/summary.json. DIR must not exist; it is built in a temporary sibling folder and renamed at the end.

One edit plan is drawn from the tokens of the union of listed samples (mode A's first; every mode that lists
a sample must agree on its tokens) and applied to each mode: under v1 (default), deletions of function words,
homophone substitutions and swaps of adjacent tokens, taken round-robin by type up to the budget
round(wer * tokens); under v2, insertions, confusion and homophone substitutions (from the default_train
vocabulary with count >= 5), deletions and swaps (capped at 10% of the budget) with target cost shares
45% / 30% / 15% / 10%. The edit model and the label rules are documented in punct_restore.data.asr_noise.
summary.json reports the edit count of each type and its share of the edit cost (and for v2, edit_model and
vocabulary_source). Masks drop deleted positions, follow swapped tokens and mask inserted tokens. Left
unchanged, since they cannot follow the edits: metadata.original_text, metadata.normalized_text,
metadata.condition and the mask "reason". The evaluator reads only sample_id, label_mode, tokens,
punct_labels, case_labels and metadata.group_id / policy_version.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from punct_restore.data.asr_noise import (
    COST,
    EDIT_MODELS,
    NO_EDITS,
    Edits,
    align,
    apply_edits,
    build_plan,
    edit_budget,
    edit_counts,
    remap_positions,
    train_vocabulary,
)
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import one_line_error, parse_json_object, read_ids, reconfigure_stdout

MODES = ("A", "B", "C")


class InputError(ValueError):
    """Input that cannot be processed; the message names the file."""


def _display_name(path: Path) -> str:
    """Folder and file name for messages, never the full path; the folder is left out when it has no name."""
    return "/".join(part for part in (path.parent.name, path.name) if part)


def _lines(path: Path) -> Iterator[tuple[int, str]]:
    """(line number, text) of the non-blank lines of a UTF-8 file; problems name the file."""
    try:
        with path.open(encoding="utf-8-sig") as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    yield number, line
    except UnicodeDecodeError:
        raise InputError(f"{_display_name(path)}: not UTF-8 text") from None
    except OSError as exc:  # Windows reports EACCES for a folder, POSIX EISDIR
        reason = "is a folder, not a file" if path.is_dir() else (exc.strerror or "cannot be read")
        raise InputError(f"{_display_name(path)}: {reason}") from None


def scan(path: Path, wanted: frozenset[str]) -> Iterator[tuple[int, dict[str, Any]]]:
    """(line number, record) of the wanted sample_ids in file order; each must occur exactly once."""
    label, seen = _display_name(path), set()
    for number, line in _lines(path):
        record = parse_json_object(line, f"{label} line {number}")
        sid = record.get("sample_id")
        if not isinstance(sid, str) or sid not in wanted:
            continue
        if sid in seen:
            raise InputError(f"{label} line {number}: duplicate sample_id")
        seen.add(sid)
        yield number, record
    if len(seen) != len(wanted):
        raise InputError(f"{label}: {len(wanted) - len(seen)} listed ID(s) not found")


def _is_span(span: Any) -> bool:
    """[start, end] with two real ints (not bool) and 0 <= start <= end."""
    return (isinstance(span, list) and len(span) == 2 and all(type(x) is int for x in span)
            and 0 <= span[0] <= span[1])


def check_record(record: Mapping[str, Any], mode: str, where: str) -> None:
    if record.get("label_mode") != mode:
        raise InputError(f"{where}: label_mode must be {mode}")
    tokens = record.get("tokens")
    if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) or not t for t in tokens):
        raise InputError(f"{where}: tokens must be a nonempty list of nonempty strings")
    for name in ("punct_labels", "case_labels"):
        labels = record.get(name)
        if not isinstance(labels, list) or len(labels) != len(tokens) or any(not isinstance(x, str) for x in labels):
            raise InputError(f"{where}: {name} must hold one string per token")
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and "offsets" in metadata:
        offsets = metadata["offsets"]
        if not isinstance(offsets, list) or len(offsets) != len(tokens):
            raise InputError(f"{where}: metadata.offsets must hold one entry per token")
        for k, span in enumerate(offsets):
            if not _is_span(span):
                raise InputError(
                    f"{where}: metadata.offsets[{k}] must be [start, end] with 0 <= start <= end, both integers")


@dataclass(frozen=True)
class ModeInput:
    mode: str
    data_path: Path
    ids_path: Path
    ids: list[str]

    def records(self) -> Iterator[dict[str, Any]]:
        """The listed records of the data file, validated, in file order."""
        for number, record in scan(self.data_path, frozenset(self.ids)):
            check_record(record, self.mode, f"{_display_name(self.data_path)} line {number}")
            yield record


def shared_tokens(inputs: Sequence[ModeInput]) -> dict[str, list[str]]:
    """Tokens by sample_id over the union of listed samples; modes must agree on every sample."""
    tokens: dict[str, list[str]] = {}
    for item in inputs:
        for record in item.records():
            if tokens.setdefault(record["sample_id"], record["tokens"]) != record["tokens"]:
                raise InputError(f"{_display_name(item.data_path)}: tokens differ from another mode")
    return tokens


def remapped_masks(masks_path: Path, c_ids: Sequence[str], tokens: Mapping[str, Sequence[str]],
                   plan: Mapping[str, Edits]) -> list[dict[str, Any]]:
    """The mode C masks of the listed records, positions moved through the edits, in file order."""
    out = []
    for number, mask in scan(masks_path, frozenset(c_ids)):
        n, where = len(tokens[mask["sample_id"]]), f"{_display_name(masks_path)} line {number}"
        positions = mask.get("excluded_positions")
        if (not isinstance(positions, list) or any(type(p) is not int or not 0 <= p < n for p in positions)
                or len(set(positions)) != len(positions)):
            raise InputError(f"{where}: excluded_positions must be unique integers from 0 to {n - 1}")
        order = align(n, plan.get(mask["sample_id"], NO_EDITS))
        out.append({**mask, "excluded_positions": remap_positions(positions, order)})
    return out


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def summarize(args: argparse.Namespace, inputs: Sequence[ModeInput], tokens: Mapping[str, Sequence[str]],
              plan: Mapping[str, Edits]) -> dict[str, Any]:
    edit_model = getattr(args, "edit_model", "v1")
    counts, before = edit_counts(plan, edit_model=edit_model), sum(len(t) for t in tokens.values())
    kind_to_cost = {"deletion": "delete", "insertion": "insert", "substitution": "replace", "swap": "swap"}
    spent = {name: counts[name] * COST[kind_to_cost[name]] for name in counts}
    cost = sum(spent.values())
    after = before - counts["deletion"] + counts.get("insertion", 0)
    summary: dict[str, Any] = {
        "release": args.release, "split": args.split, "wer": args.wer, "seed": args.seed,
        "samples": len(tokens), "records": {item.mode: len(item.ids) for item in inputs},
        "tokens_before": before, "tokens_after": after,
        "budget": edit_budget(tokens, args.wer), "edits": counts, "edit_cost": cost,
        "cost_share": {kind: part / (cost or 1) for kind, part in spent.items()},  # all 0 when no edits
        "achieved_wer": cost / before,
    }
    if edit_model != "v1":
        summary["edit_model"] = edit_model
        summary["vocabulary_source"] = "default_train"
    return summary


def write_outputs(folder: Path, args: argparse.Namespace, inputs: Sequence[ModeInput],
                  masks: Sequence[Mapping[str, Any]], plan: Mapping[str, Edits],
                  summary: Mapping[str, Any]) -> None:
    for item in inputs:
        rows = (apply_edits(r, plan.get(r["sample_id"], NO_EDITS)) for r in item.records())
        write_jsonl(folder / f"mode_{item.mode}" / f"{args.split}.jsonl", rows)
        shutil.copyfile(item.ids_path, folder / f"{args.split}_mode_{item.mode}.ids")
    write_jsonl(folder / "masks" / f"{args.split}.jsonl", masks)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (folder / "summary.json").write_text(text, encoding="utf-8", newline="\n")


def wer_value(text: str) -> float:
    value = float(text)  # argparse reports a ValueError as an invalid value
    if not 0 < value <= 0.5:  # also rejects nan
        raise argparse.ArgumentTypeError(f"must be above 0 and at most 0.5, got {text}")
    return value


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="recap simulate-asr-noise",
                                     description="Simulate ASR word errors in gold data, labels kept consistent.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--release", default="sponspeech-m1-v2")
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--wer", type=wer_value, required=True, help="target word error rate, above 0 up to 0.5")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--edit-model", choices=list(EDIT_MODELS), default="v1",
                        help="word-error model (default: v1)")
    parser.add_argument("--masks", type=Path, help="mode C casing masks (default: the release's case_joint file)")
    parser.add_argument("--out", type=Path, required=True, help="new folder; it must not exist")
    return parser.parse_args(argv)


def load_inputs(args: argparse.Namespace) -> list[ModeInput]:
    processed = args.data_root / "processed" / args.release
    lists = args.data_root / "splits" / args.release / "primary_eval"
    ids = {mode: lists / f"{args.split}_mode_{mode}.ids" for mode in MODES}
    return [ModeInput(mode, processed / f"mode_{mode}" / f"{args.split}.jsonl", ids[mode], read_ids(ids[mode]))
            for mode in MODES]


def load_train_vocab(args: argparse.Namespace) -> frozenset[str]:
    """Words occurring at least MIN_TRAIN_COUNT times in default_train (mode A)."""
    train_path = args.data_root / "processed" / args.release / "mode_A" / "train.jsonl"
    ids_path = args.data_root / "splits" / args.release / "default_train" / "mode_A.ids"
    wanted = frozenset(read_ids(ids_path))

    def token_lists() -> Iterator[list[str]]:
        for number, record in scan(train_path, wanted):
            check_record(record, "A", f"{_display_name(train_path)} line {number}")
            yield record["tokens"]

    return train_vocabulary(token_lists())


def run(args: argparse.Namespace) -> str:
    """Build the noisy release next to args.out and rename it into place; returns a status line."""
    out = args.out
    if os.path.lexists(out):
        raise InputError(f"{out.name}: already exists; refusing to overwrite")
    if not out.parent.is_dir():
        raise InputError(f"{out.name}: the folder to create it in does not exist")
    inputs = load_inputs(args)
    tokens = shared_tokens(inputs)
    edit_model = getattr(args, "edit_model", "v1")
    vocab = load_train_vocab(args) if edit_model == "v2" else frozenset()
    plan = build_plan(tokens, args.wer, args.seed, edit_model=edit_model, vocab=vocab)
    default_masks = args.data_root / "processed" / args.release / "masks" / "case_joint" / f"{args.split}.jsonl"
    masks_path = args.masks or default_masks
    c_ids = next(item.ids for item in inputs if item.mode == "C")
    masks = remapped_masks(masks_path, c_ids, tokens, plan)
    summary = summarize(args, inputs, tokens, plan)
    folder = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))
    try:
        write_outputs(folder, args, inputs, masks, plan, summary)
        folder.rename(out)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return (f"wrote {out.name}: edit cost {summary['edit_cost']} of budget {summary['budget']} "
            f"(WER {summary['achieved_wer']:.4f}), tokens {summary['tokens_before']} -> {summary['tokens_after']}")


def main(argv: list[str] | None = None) -> int:
    """0 on success, 1 with one ``error:`` line on stderr, 2 on usage errors."""
    reconfigure_stdout()
    args = parse_args(argv)
    try:
        print(run(args))
    except (OSError, ValueError) as exc:
        print(f"error: {one_line_error(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
