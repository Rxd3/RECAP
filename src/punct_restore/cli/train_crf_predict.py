"""The `predict` subcommand of train_crf: gold-free predictions from a saved CRF model artifact."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from punct_restore.models.crf import (
    CRFModel,
    require_scoring_mode,
)
from punct_restore.prediction_io import (
    check_new_output_paths,
    one_line_error,
    read_observations,
    write_prediction_outputs,
)


def _predict_path_error(args: argparse.Namespace, manifest_path: Path | None) -> str | None:
    """Return a one-line error for missing, unreadable or clashing paths, else None."""
    model_path, input_path, output_path = Path(args.model), Path(args.input), Path(args.output)
    try:
        if not model_path.is_file():
            return f"Error: Model artifact not found at {model_path.name}"
        if not input_path.is_file():
            return f"Error: Input JSONL not found at {input_path.name}"
        check_new_output_paths(
            {"--output": output_path, "--manifest-out": manifest_path},
            {"--input": input_path, "--model": model_path,
             "--ids": Path(args.ids) if args.ids else None},
        )
        if output_path.exists() and not args.overwrite:
            return f"Error: Output file already exists at {output_path.name}. Use --overwrite to replace."
        if manifest_path is not None and manifest_path.exists() and not args.overwrite:
            return f"Error: Manifest already exists at {manifest_path.name}. Use --overwrite to replace."
    except ValueError as e:
        return f"Error: {e}"
    except OSError as e:  # e.g. a folder on the way cannot be read
        shown = Path(e.filename).name if e.filename else "a path"
        return f"Error: cannot check {shown}: {e.strerror or e}"
    return None


def _export_predictions(
    model: CRFModel,
    predictions: Sequence[Mapping[str, Any]],
    input_path: Path,
    output_path: Path,
    manifest_path: Path | None,
) -> None:
    """Write the predictions JSONL and, when requested, its sidecar manifest; report both after."""
    write_prediction_outputs(
        predictions,
        output_path,
        manifest_path,
        input_path=input_path,
        model_name=model.model_name,
        label_mode=model.label_mode,
        ordered_labels=model.vocabulary,
    )
    print(f"[RECAP CRF] Predictions exported to {output_path}")
    if manifest_path is not None:
        print(f"[RECAP CRF] Manifest written to {manifest_path}")


def cmd_predict(args: argparse.Namespace) -> int:
    """Generate contract-compliant predictions from a saved CRF model artifact."""
    model_path, input_path, output_path = Path(args.model), Path(args.input), Path(args.output)
    manifest_path = Path(args.manifest_out) if args.manifest_out else None

    path_error = _predict_path_error(args, manifest_path)
    if path_error:
        sys.stderr.write(path_error + "\n")
        return 1

    print(f"[RECAP CRF] Loading model from {model_path}...")
    try:
        model = CRFModel.load_model(model_path)
    except (OSError, ValueError) as e:  # an unreadable, corrupt or incompatible artifact
        sys.stderr.write(f"Failed to load model: {one_line_error(e, model_path)}\n")
        return 1
    try:
        require_scoring_mode(model.label_mode)  # a CASE helper artifact loads fine but must not be scored here
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1

    ids_path = Path(args.ids) if args.ids else None
    try:
        records_to_predict = read_observations(input_path, ids_path)
    except (OSError, ValueError) as e:
        sys.stderr.write(f"Error: {one_line_error(e, input_path, ids_path)}\n")
        return 1
    if ids_path is not None:
        print(f"[RECAP CRF] Restricting inference to {len(records_to_predict):,} IDs from {ids_path.name}...")

    print(f"[RECAP CRF] Running batch inference on {len(records_to_predict):,} sequences...")
    start_time = time.perf_counter()
    try:
        predictions = model.predict(records_to_predict)
    except ValueError as e:  # the model predicted a label sequence the contract rejects
        sys.stderr.write(f"Prediction failed: {one_line_error(e)}\n")
        return 1
    duration = time.perf_counter() - start_time
    total_tokens = sum(len(p["predicted_labels"]) for p in predictions)
    tok_per_sec = total_tokens / duration if duration > 0 else 0.0
    print(f"[RECAP CRF] Inferred {len(predictions):,} sequences ({total_tokens:,} tokens) "
          f"in {duration:.2f}s ({tok_per_sec:.1f} tok/s).")

    try:
        _export_predictions(model, predictions, input_path, output_path, manifest_path)
    except (OSError, ValueError) as e:
        sys.stderr.write(f"Error: Could not write prediction outputs: {e}\n")
        return 1
    return 0
