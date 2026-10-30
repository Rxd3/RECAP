"""One-group-off feature ablations of the improved CRF preset (validation only).

Usage:
    python scripts/run_ablations.py --mode {A,B,C} --out DIR [--sample 50000]
        [--variants improved,improved_minus_window2,...,improved_plus_filler]
        [--data-root DATA] [--release sponspeech-m1-v2]

Trains this repository's CRFModel (c1 = c2 = 0.1, 100 L-BFGS iterations) on a fixed seeded sample
of default_train, once with the full improved preset, once per removed feature group and once with
the filler cues added. Primary-validation predictions are written to
DIR/mode_<M>/<variant>/validation.jsonl, and the sampled training IDs to DIR/mode_<M>/sample.ids.
Test data is never read.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Support the documented source-checkout command before an editable install.
sys.path.insert(0, str(ROOT / "src"))
from punct_restore.features.crf_features import FeatureConfig  # noqa: E402
from punct_restore.models.crf import CRFModel  # noqa: E402

SEED = 2026
# group name -> FeatureConfig.improved() override that removes it
GROUPS: dict[str, dict[str, Any]] = {
    "window2": {"window_size": 1},
    "ngrams": {"use_token_ngrams": False},
    "question": {"use_question_cues": False},
    "clause": {"use_clause_cues": False},
    "casing": {"use_casing_cues": False},
    "position": {"use_positional_buckets": False},
}
VARIANTS: dict[str, dict[str, Any]] = {
    "improved": {},
    **{f"improved_minus_{group}": override for group, override in GROUPS.items()},
    "improved_plus_filler": {"use_filler_cues": True},
}


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def variant_list(text: str) -> list[str]:
    names = text.split(",")
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown or empty variant {unknown[0]!r}; choose from {', '.join(VARIANTS)}"
        )
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("duplicate variant names")
    return names


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=["A", "B", "C"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample", type=positive_int, default=50000)
    parser.add_argument("--variants", type=variant_list, default=list(VARIANTS))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--release", default="sponspeech-m1-v2")
    return parser.parse_args(argv)


def read_ids(path: Path) -> set[str]:
    return set(path.read_text(encoding="utf-8").split())


def read_records(path: Path, ids: set[str]) -> dict[str, dict[str, Any]]:
    """Stream a JSONL file and keep only the rows whose sample_id is in ids."""
    keep: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"{path.name} line {number}: invalid JSON") from None
            if not isinstance(row, dict) or "sample_id" not in row:
                raise ValueError(f"{path.name} line {number}: record without sample_id")
            sid = row["sample_id"]
            if not isinstance(sid, str) or not sid:
                raise ValueError(f"{path.name} line {number}: sample_id must be a non-empty string")
            if sid in seen:
                raise ValueError(f"{path.name} line {number}: duplicate sample_id {sid!r}")
            seen.add(sid)
            if sid in ids:
                keep[sid] = row
    missing = ids - keep.keys()
    if missing:
        raise ValueError(f"{len(missing)} IDs missing from {path.name}")
    return keep


def load_inputs(
    args: argparse.Namespace,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (sorted sample IDs, training records, gold-free validation inputs)."""
    mode, root, release = args.mode, args.data_root, args.release
    train_ids = sorted(read_ids(root / "splits" / release / "default_train" / f"mode_{mode}.ids"))
    if args.sample > len(train_ids):
        raise ValueError(f"--sample {args.sample} exceeds the {len(train_ids)} default_train IDs")
    sample = sorted(random.Random(SEED).sample(train_ids, args.sample))
    processed = root / "processed" / release / f"mode_{mode}"
    train = read_records(processed / "train.jsonl", set(sample))
    val_ids = read_ids(root / "splits" / release / "primary_eval" / f"validation_mode_{mode}.ids")
    val = read_records(processed / "validation.jsonl", val_ids)
    gold_free = [{"sample_id": k, "tokens": v.get("tokens")} for k, v in sorted(val.items())]
    return sample, [train[i] for i in sample], gold_free


def run_variant(
    mode: str,
    variant: str,
    records: list[dict[str, Any]],
    gold_free: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    model = CRFModel(
        label_mode=mode,
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        feature_config=FeatureConfig.improved(**VARIANTS[variant]),
        model_name=f"crf_{variant}",
    )
    model.fit(records)
    preds = model.predict(gold_free)
    out_dir.mkdir()
    with open(out_dir / "validation.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(json.dumps(pred) + "\n" for pred in preds)
    del model, preds
    gc.collect()


def describe(exc: Exception) -> str:
    """One-line message that names a file by its name only, never its full path."""
    if isinstance(exc, OSError) and exc.filename:
        return f"{exc.strerror}: {Path(str(exc.filename)).name}"
    return str(exc)


def _already_exists(root: Path) -> ValueError:
    return ValueError(f"{root.name} already exists under --out; remove it or pick another --out")


def run(args: argparse.Namespace) -> None:
    root = args.out / f"mode_{args.mode}"
    if root.exists():  # fail early, before reading the large train.jsonl
        raise _already_exists(root)
    sample, records, gold_free = load_inputs(args)
    try:
        root.mkdir(parents=True, exist_ok=False)  # the real guard against a concurrent run
    except FileExistsError:
        raise _already_exists(root) from None
    (root / "sample.ids").write_text("\n".join(sample) + "\n", encoding="utf-8", newline="\n")
    for variant in args.variants:
        start = time.perf_counter()
        run_variant(args.mode, variant, records, gold_free, root / variant)
        print(f"{args.mode} {variant}: {time.perf_counter() - start:.0f}s", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError) as exc:
        print(f"error: {describe(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
