"""One-group-off feature ablations of the improved CRF preset (validation only).

Usage:
    python scripts/run_ablations.py --mode {A,B,C} --out DIR [--sample 50000]
        [--variants improved,improved_minus_window2,...,improved_plus_filler,
         improved_plus_question2,gap_improved,case_improved,factored]
        [--data-root DATA] [--release sponspeech-m1-v2]

Trains this repository's CRFModel (c1 = c2 = 0.1, 100 L-BFGS iterations) on a fixed seeded sample
of default_train, once with the full improved preset, once per removed feature group, once with
the filler cues added and once with the question-lookback features added (improved_plus_question2:
wh_aux_lookback and is_tag_question_cue). The last variant, gap_improved, trains GapClassifier
("CRF without transitions": per-token logistic regression, C = 1, 200 L-BFGS iterations) on the
same improved features and the same sample, so its difference from improved isolates the value of
label transitions. Primary-validation predictions are written to
DIR/mode_<M>/<variant>/validation.jsonl, and the sampled training IDs to DIR/mode_<M>/sample.ids.
Test data is never read.

Two variants exist for --mode C only, to compare the joint CRF (improved) with a factored one on the same
sample. case_improved trains a casing-only CRF (label mode CASE, from the records' case_labels).
factored trains a mode A improved CRF on the same records (their punct_labels), writes its predictions to
factored/punct_A.jsonl, and joins them with the case_improved predictions into mode C labels
(factored/validation.jsonl); it needs case_improved earlier in --variants. Without --variants, modes A and B
run the first ten variants and mode C runs all twelve.
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
from punct_restore.features.crf_features import FeatureConfig
from punct_restore.models.crf import CRFModel
from punct_restore.models.factored import combine
from punct_restore.models.gap_classifier import GapClassifier

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
GAP_VARIANT = "gap_improved"
CASE_VARIANT = "case_improved"
FACTORED_VARIANT = "factored"
C_ONLY_VARIANTS = (CASE_VARIANT, FACTORED_VARIANT)
VARIANTS: dict[str, dict[str, Any]] = {
    "improved": {},
    **{f"improved_minus_{group}": override for group, override in GROUPS.items()},
    "improved_plus_filler": {"use_filler_cues": True},
    "improved_plus_question2": {"use_question_lookback": True},
    GAP_VARIANT: {},  # improved features, trained with GapClassifier instead of the CRF
    CASE_VARIANT: {},  # mode C only: casing-only CRF
    FACTORED_VARIANT: {},  # mode C only: mode A CRF + the case_improved predictions -> joint labels
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
    parser.add_argument("--variants", type=variant_list, default=None)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--release", default="sponspeech-m1-v2")
    args = parser.parse_args(argv)
    problem = variants_problem(args.mode, args.variants)
    if problem:
        parser.error(f"--variants: {problem}")
    if args.variants is None:
        args.variants = [v for v in VARIANTS if args.mode == "C" or v not in C_ONLY_VARIANTS]
    return args


def variants_problem(mode: str, variants: list[str] | None) -> str | None:
    """Why an explicit --variants list does not fit --mode, or None (the C-only variants need mode C)."""
    names = variants or []
    for name in names:
        if name in C_ONLY_VARIANTS and mode != "C":
            return f"{name} is only valid with --mode C"
    if FACTORED_VARIANT in names and CASE_VARIANT not in names[: names.index(FACTORED_VARIANT)]:
        return f"{FACTORED_VARIANT} needs {CASE_VARIANT} earlier in --variants"
    return None


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


def build_crf(label_mode: str, feature_config: FeatureConfig, model_name: str) -> CRFModel:
    return CRFModel(
        label_mode=label_mode,
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        feature_config=feature_config,
        model_name=model_name,
    )


def write_predictions(path: Path, preds: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(json.dumps(pred) + "\n" for pred in preds)


def run_factored(
    records: list[dict[str, Any]],
    gold_free: list[dict[str, Any]],
    out_dir: Path,
    feature_config: FeatureConfig,
) -> None:
    """Mode A CRF on the mode C sample, joined with the case_improved predictions already on disk."""
    punct_model = build_crf("A", feature_config, "crf_improved_on_C_sample")
    punct_model.fit(records)
    punct = punct_model.predict(gold_free)
    out_dir.mkdir()
    write_predictions(out_dir / "punct_A.jsonl", punct)
    ids = [row["sample_id"] for row in gold_free]
    case_rows = read_records(out_dir.parent / CASE_VARIANT / "validation.jsonl", set(ids))
    joint = combine(punct, [case_rows[i] for i in ids], model_name="crf_factored")
    write_predictions(out_dir / "validation.jsonl", joint)


def run_variant(
    mode: str,
    variant: str,
    records: list[dict[str, Any]],
    gold_free: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    feature_config = FeatureConfig.improved(**VARIANTS[variant])
    if variant == FACTORED_VARIANT:
        run_factored(records, gold_free, out_dir, feature_config)
        return
    model: CRFModel | GapClassifier
    if variant == GAP_VARIANT:
        model = GapClassifier(mode, feature_config=feature_config, C=1.0, max_iter=200, model_name=variant)
    else:
        model = build_crf("CASE" if variant == CASE_VARIANT else mode, feature_config, f"crf_{variant}")
    model.fit(records)
    preds = model.predict(gold_free)
    out_dir.mkdir()
    write_predictions(out_dir / "validation.jsonl", preds)
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
