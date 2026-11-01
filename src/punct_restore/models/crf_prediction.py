"""CRF prediction steps that are not CLI-specific: path checks, loading, reading, predicting and writing.

Used by punct_restore.cli.train_crf (`predict`), which keeps the order of the steps, the progress lines on stdout and
the exit code. A step that fails for a reason the user can fix raises PredictionFailed, whose text is the one-line
reason to show after ``error: `` (``<file name>: <reason>`` when a file is at fault); any other exception is a bug
and propagates. Nothing here prints.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from punct_restore.models.crf import CRFModel, require_scoring_mode
from punct_restore.prediction_io import (
    check_new_output_paths,
    one_line_error,
    one_line_error_naming,
    read_observations,
    require_file,
    write_prediction_outputs,
)


class PredictionFailed(Exception):
    """A prediction step failed; ``str(error)`` is the one-line reason for the user, without the ``error: `` prefix."""


def _named(exc: BaseException, path: Path) -> str:
    """The one-line reason of ``exc``, led by the file name of ``path`` unless the text already names it."""
    text = one_line_error_naming(exc, path)
    return text if path.name in text else f"{path.name}: {text}"


def check_predict_paths(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    manifest_path: Path | None,
    ids_path: Path | None,
    overwrite: bool,
) -> None:
    """Raise PredictionFailed for a missing, unreadable or clashing path; files are named by name only."""
    try:
        require_file(model_path, "model artifact")
        require_file(input_path, "input JSONL")
        check_new_output_paths(
            {"--output": output_path, "--manifest-out": manifest_path},
            {"--input": input_path, "--model": model_path, "--ids": ids_path},
        )
        if output_path.exists() and not overwrite:
            raise PredictionFailed(f"{output_path.name}: output file already exists; use --overwrite to replace")
        if manifest_path is not None and manifest_path.exists() and not overwrite:
            raise PredictionFailed(f"{manifest_path.name}: manifest already exists; use --overwrite to replace")
    except ValueError as e:
        raise PredictionFailed(str(e)) from e
    except OSError as e:  # e.g. a folder on the way cannot be read
        raise PredictionFailed(one_line_error(e)) from e


def load_scoring_model(model_path: Path) -> CRFModel:
    """Load the artifact; PredictionFailed if it is unreadable, corrupt, incompatible or not a scoring model."""
    try:
        model = CRFModel.load_model(model_path)
    except (OSError, ValueError) as e:  # an unreadable, corrupt or incompatible artifact
        raise PredictionFailed(_named(e, model_path)) from e
    try:
        require_scoring_mode(model.label_mode)  # a CASE helper artifact loads fine but must not be scored here
    except ValueError as e:
        raise PredictionFailed(f"{model_path.name}: {e}") from e
    return model


def read_input_records(input_path: Path, ids_path: Path | None) -> list[dict[str, Any]]:
    """The gold-free records of the input file, restricted to the ID list when given; PredictionFailed if unreadable."""
    try:
        return read_observations(input_path, ids_path)
    except (OSError, ValueError) as e:
        raise PredictionFailed(one_line_error_naming(e, input_path, ids_path)) from e


def predict_records(model: CRFModel, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The model's predictions; PredictionFailed if it predicted a label sequence the contract rejects."""
    try:
        return model.predict(records)
    except ValueError as e:
        raise PredictionFailed(f"prediction failed: {one_line_error(e)}") from e


def inference_summary(predictions: Sequence[Mapping[str, Any]], duration: float) -> str:
    """The sequence and token counts of a batch of predictions and the rate they were made at."""
    total_tokens = sum(len(p["predicted_labels"]) for p in predictions)
    tok_per_sec = total_tokens / duration if duration > 0 else 0.0
    return (f"Inferred {len(predictions):,} sequences ({total_tokens:,} tokens) "
            f"in {duration:.2f}s ({tok_per_sec:.1f} tok/s).")


def write_model_predictions(
    model: CRFModel,
    predictions: Sequence[Mapping[str, Any]],
    input_path: Path,
    output_path: Path,
    manifest_path: Path | None,
) -> None:
    """Write the predictions JSONL and, when a path is given, its sidecar manifest naming the model's labels."""
    try:
        write_prediction_outputs(
            predictions,
            output_path,
            manifest_path,
            input_path=input_path,
            model_name=model.model_name,
            label_mode=model.label_mode,
            ordered_labels=model.vocabulary,
        )
    except (OSError, ValueError) as e:
        reason = one_line_error_naming(e, output_path, manifest_path, input_path)
        raise PredictionFailed(f"could not write prediction outputs: {reason}") from e
