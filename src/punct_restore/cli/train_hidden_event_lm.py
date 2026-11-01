"""Train and apply the hidden-event language model (comparison method H).

Usage:
  recap train-hidden-event-lm train --mode A --input TRAIN.jsonl --ids TRAIN.ids \\
      --output MODEL.json [--order 3] [--discount 0.75] [--min-count 1] [--model-name hidden_event_lm]
  recap train-hidden-event-lm predict --model MODEL.json --input IN.jsonl [--ids IDS] \\
      --output PRED.jsonl [--manifest-out PRED.manifest.json]

``train`` keeps only the records named by the ID list (each must be present, split "train" and of the requested
mode) and writes a JSON model artifact. ``predict`` writes gold-free predictions in the model contract format and,
with --manifest-out, the sidecar manifest the evaluator checks. Outputs are never overwritten, and an output may
not be another output or an input; both are refused before anything is read. A failure prints one ``error:`` line
on stderr (files by name only) and exits 1; a usage error exits 2.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from punct_restore.models.hidden_event_lm import HiddenEventLM
from punct_restore.prediction_io import (
    check_new_output_paths,
    one_line_error,
    read_frozen_ids,
    read_observations,
    reconfigure_stdout,
    write_prediction_outputs,
)

PATH_OPTIONS = ("input", "ids", "output", "model", "manifest_out")


def _refuse_existing(*named: tuple[str, Path | None]) -> None:
    for flag, path in named:
        if path is not None and os.path.lexists(path):
            raise ValueError(f"{flag}: {path.name} already exists, refusing to overwrite it")


def _require_files(*named: tuple[str, Path | None]) -> None:
    for flag, path in named:
        if path is not None and not path.is_file():
            raise ValueError(f"{flag}: {path.name} is not an existing file")


@contextlib.contextmanager
def _utf8_file(path: Path) -> Iterator[None]:
    """Turn a UnicodeDecodeError raised while reading ``path`` into a one-line ValueError naming the file."""
    try:
        yield
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: not valid UTF-8") from exc


def _interned(value: Any) -> Any:
    """Share equal strings of a list, so a million-token training set stays small in memory."""
    if isinstance(value, list):
        return [sys.intern(item) if type(item) is str else item for item in value]
    return value


def _slim(record: dict[str, Any], mode: str) -> dict[str, Any]:
    fields = ("sample_id", "split", "label_mode", "tokens", "punct_labels")
    fields += ("case_labels",) if mode == "C" else ()
    return {field: _interned(record.get(field)) for field in fields}


def _keep_line(line: str, where: str, wanted: set[str], kept: dict[str, dict[str, Any]], mode: str) -> None:
    try:
        record = json.loads(line)
    except (ValueError, RecursionError) as exc:  # a deeply nested line raises RecursionError
        raise ValueError(f"{where}: not valid JSON") from exc
    if not isinstance(record, dict):
        raise ValueError(f"{where}: not a JSON object")  # noqa: TRY004 - main() reports ValueError only
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"{where}: sample_id must be a nonempty string")
    if sample_id not in wanted:
        return
    if sample_id in kept:
        raise ValueError(f"{where}: duplicate sample_id {sample_id!r}")
    if record.get("split") != "train":
        raise ValueError(f"{where}: {sample_id!r} has split {record.get('split')!r}; only train records can be used")
    if record.get("label_mode") != mode:
        raise ValueError(f"{where}: {sample_id!r} has label_mode {record.get('label_mode')!r}, expected {mode!r}")
    kept[sample_id] = _slim(record, mode)


def read_training_records(input_path: Path, ids_path: Path, mode: str) -> list[dict[str, Any]]:
    """The listed records of ``input_path`` in file order, reduced to the fields training reads.

    ValueError if a listed ID is absent, is not split "train" or has another label mode.
    """
    with _utf8_file(ids_path):
        ids = read_frozen_ids(ids_path)
    wanted = set(ids)
    kept: dict[str, dict[str, Any]] = {}
    with _utf8_file(input_path), open(input_path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                _keep_line(line, f"{input_path.name} line {number}", wanted, kept, mode)
    missing = [sample_id for sample_id in ids if sample_id not in kept]
    if missing:
        raise ValueError(f"{len(missing)} listed ID(s) are missing from {input_path.name}, first {missing[0]!r}")
    return list(kept.values())


def _cmd_train(args: argparse.Namespace) -> str:
    input_path, ids_path, output = Path(args.input), Path(args.ids), Path(args.output)
    check_new_output_paths({"--output": output}, {"--input": input_path, "--ids": ids_path})
    _refuse_existing(("--output", output))
    _require_files(("--input", input_path), ("--ids", ids_path))
    model = HiddenEventLM(
        args.mode, order=args.order, discount=args.discount, min_count=args.min_count, model_name=args.model_name
    )
    model.fit(read_training_records(input_path, ids_path, args.mode))
    model.save_model(output)
    summary = model.training_summary
    return (f"trained mode {args.mode} order {args.order}: {summary['num_sequences']} sequences, "
            f"{summary['num_tokens']} tokens, {summary['vocabulary_size']} word types -> {output.name}")


def _read_named(input_path: Path, ids_path: Path | None) -> list[dict]:
    """``read_observations`` with the input file named in a JSON error, as train does."""
    try:
        return read_observations(input_path, ids_path)
    except RecursionError as exc:
        raise ValueError(f"{input_path.name}: not valid JSON (nested too deeply)") from exc
    except UnicodeDecodeError:
        raise  # the caller's _utf8_file names it
    except ValueError as exc:
        if str(input_path) in str(exc):  # read_observations names files by full path; _one_line shortens it
            raise
        raise ValueError(f"{input_path.name}: {exc}") from exc


def _cmd_predict(args: argparse.Namespace) -> str:
    model_path, input_path, output = Path(args.model), Path(args.input), Path(args.output)
    ids_path = Path(args.ids) if args.ids else None
    manifest = Path(args.manifest_out) if args.manifest_out else None
    check_new_output_paths({"--output": output, "--manifest-out": manifest},
                           {"--model": model_path, "--input": input_path, "--ids": ids_path})
    outputs = (("--output", output), ("--manifest-out", manifest))
    _refuse_existing(*outputs)
    _require_files(("--model", model_path), ("--input", input_path), ("--ids", ids_path))
    model = HiddenEventLM.load_model(model_path)
    if ids_path is not None:
        with _utf8_file(ids_path):  # read_observations reads the list again; this names a bad ID file correctly
            read_frozen_ids(ids_path)
    with _utf8_file(input_path):
        observations = _read_named(input_path, ids_path)
    predictions = model.predict(observations)
    _refuse_existing(*outputs)  # decoding is long: an output may have appeared since the first check
    write_prediction_outputs(predictions, output, manifest, input_path=input_path, model_name=model.model_name,
                             label_mode=model.label_mode, ordered_labels=model.vocabulary)
    return f"predicted {len(predictions)} sequences -> {output.name}"


def _one_line(exc: ValueError | OSError | RecursionError, args: argparse.Namespace) -> str:
    """The message on one line, with files named by name only (an OSError reports just its file's name)."""
    return one_line_error(exc, *(getattr(args, option, None) for option in PATH_OPTIONS))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recap train-hidden-event-lm",
        description="Hidden-event language model (comparison method H): train and predict.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Fit the model on the listed training records.")
    train.add_argument("--mode", required=True, choices=["A", "B", "C"], help="Label mode.")
    train.add_argument("--input", required=True, help="Training JSONL.")
    train.add_argument("--ids", required=True, help="ID list of the records to train on.")
    train.add_argument("--output", required=True, help="New model JSON (never overwritten).")
    train.add_argument("--order", type=int, default=3, help="N-gram order, at least 2 (default 3).")
    train.add_argument("--discount", type=float, default=0.75, help="Absolute discount, 0 < D < 1 (default 0.75).")
    train.add_argument("--min-count", type=int, default=1, help="Words rarer than this become <unk> (default 1).")
    train.add_argument("--model-name", default="hidden_event_lm", help="model_name written to predictions.")
    train.set_defaults(handler=_cmd_train)

    predict = commands.add_parser("predict", help="Predict labels with a trained model.")
    predict.add_argument("--model", required=True, help="Model JSON written by 'train'.")
    predict.add_argument("--input", required=True, help="JSONL with sample_id and tokens.")
    predict.add_argument("--ids", default=None, help="ID list restricting the input records.")
    predict.add_argument("--output", required=True, help="New predictions JSONL (never overwritten).")
    predict.add_argument("--manifest-out", default=None, help="New prediction manifest JSON (never overwritten).")
    predict.set_defaults(handler=_cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: 0 on success, 1 with one ``error:`` line on stderr, 2 on usage errors."""
    # A path the console code page cannot encode must not crash the run after files are written.
    reconfigure_stdout()
    args = _build_parser().parse_args(argv)
    try:
        summary = args.handler(args)
    except (ValueError, OSError, RecursionError) as exc:
        print(f"error: {_one_line(exc, args)}", file=sys.stderr)
        return 1
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
