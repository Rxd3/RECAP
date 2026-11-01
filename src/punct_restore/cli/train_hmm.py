"""Train Member 2's HMM from shared-contract JSONL, without downloading data."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple

from punct_restore.hashing import sha256_file
from punct_restore.models.hmm import HMMModel
from punct_restore.models.hmm_records import SelectedRecords
from punct_restore.models.hmm_workflow import (
    ReleaseInputs,
    code_identity,
    file_identity,
    portable_path,
    verify_training_release,
)
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import one_line_error, read_ids


def load_labels(path: Path, mode: str) -> list[str]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ValueError(
            'YAML configuration needs PyYAML; install with pip install -e ".[dev]"'
        ) from exc
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if type(config["schema_version"]) is not int or config["schema_version"] != 1:
            raise ValueError("unsupported label configuration version")
        mode_config = config["modes"][mode]
        punctuation = config["punctuation"][mode_config["punctuation_set"]]
        if mode == "C":
            if mode_config["separator"] != "|":
                raise ValueError("joint labels must use the | separator")
            labels = [f"{p}|{c}" for p in punctuation for c in config["casing"]]
            if labels != mode_config["labels"]:
                raise ValueError("explicit joint labels do not match the ordered 4-by-3 product")
        else:
            labels = punctuation
        if mode_config["num_classes"] != len(labels):
            raise ValueError("num_classes does not match label vocabulary")
        return labels
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid label configuration: {exc}") from exc


def nonempty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be blank")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recap train-hmm", description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="shared-contract train JSONL")
    parser.add_argument(
        "--train-ids",
        type=Path,
        required=True,
        help="explicit frozen training ID selection; never train on the whole release implicitly",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="new run directory; never overwritten"
    )
    parser.add_argument("--label-mode", choices=["A", "B", "C"], default="A")
    parser.add_argument("--run-id", type=nonempty, help="stable experiment/candidate identifier")
    parser.add_argument("--label-config", type=Path, default=ROOT / "configs" / "label_sets.yaml")
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="positive additive smoothing (untuned default)"
    )
    parser.add_argument(
        "--min-token-count", type=int, default=2, help="train frequency below which words use UNK"
    )
    parser.add_argument("--dataset-release-id", type=nonempty, required=True)
    parser.add_argument("--preprocessing-version", type=nonempty, required=True)
    parser.add_argument("--tokenizer-version", type=nonempty, required=True)
    parser.add_argument("--label-policy-version", type=nonempty, required=True)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        required=True,
        help="frozen manifest to checksum; schema owned by Member 1",
    )
    parser.add_argument(
        "--predict-input", type=Path, help="optional records to predict; no gold labels are read"
    )
    parser.add_argument("--predict-ids", type=Path, help="optional prediction ID selection")
    parser.add_argument(
        "--release-manifest",
        type=Path,
        help="verify canonical source files, IDs, versions and label config against a data release",
    )
    return parser


class _Prediction(NamedTuple):
    """The predictions of a run, with the selection counts and the seconds it took to read the input and predict."""

    records: list[dict[str, Any]]
    counts: dict[str, int]
    seconds: float


def _verify_release(args: argparse.Namespace) -> dict[str, Any] | None:
    """Verify the inputs against the data release manifest, when one is given."""
    if args.release_manifest is None:
        return None
    if not (
        args.preprocessing_version == args.tokenizer_version == args.label_policy_version
    ):
        raise ValueError(
            "this release uses one policy version for preprocessing/tokens/labels"
        )
    return verify_training_release(
        args.release_manifest,
        ReleaseInputs(
            train=args.train,
            train_ids=args.train_ids,
            split_manifest=args.split_manifest,
            label_config=args.label_config,
            mode=args.label_mode,
            release_id=args.dataset_release_id,
            policy_version=args.label_policy_version,
            prediction=args.predict_input,
            prediction_ids=args.predict_ids,
        ),
    )


def _read_selections(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...], tuple[str, ...] | None, list[str], dict[str, Any] | None]:
    """Check the run arguments; read the ID selections, the labels and the optional release record."""
    if args.output_dir.exists():
        raise ValueError("output-dir already exists; choose a new run directory")
    train_ids = tuple(read_ids(args.train_ids, strict=True))
    predict = args.predict_ids
    prediction_ids = None if predict is None else tuple(read_ids(predict, strict=True))
    labels = load_labels(args.label_config, args.label_mode)
    release = _verify_release(args)
    return train_ids, prediction_ids, labels, release


def _build_provenance(args: argparse.Namespace, train_ids: tuple[str, ...], labels: list[str]) -> dict[str, Any]:
    """Describe the run: versions, identities of every input file, code, environment and hyperparameters."""
    return {
        "schema_version": 1,
        "run_id": args.run_id or args.output_dir.name,
        "dataset_release_id": args.dataset_release_id,
        "preprocessing_version": args.preprocessing_version,
        "tokenizer_version": args.tokenizer_version,
        "label_policy_version": args.label_policy_version,
        "training_input": file_identity(args.train),
        "training_ids": file_identity(args.train_ids),
        "selected_training_sequences": len(train_ids),
        "release_manifest": (
            file_identity(args.release_manifest) if args.release_manifest else None
        ),
        "prediction_input": file_identity(args.predict_input) if args.predict_input else None,
        "prediction_ids": file_identity(args.predict_ids) if args.predict_ids else None,
        "split_manifest": file_identity(args.split_manifest),
        "label_configuration": file_identity(args.label_config),
        "code": code_identity(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "implementation": "stdlib_supervised_hmm_v1",
        },
        "random_seed": None,
        "deterministic": True,
        "label_mode": args.label_mode,
        "ordered_labels": labels,
        "smoothing": args.smoothing,
        "min_token_count": args.min_token_count,
    }


def _training_command(args: argparse.Namespace, run_id: str) -> list[str]:
    """The portable command line that reproduces this run."""
    command = [
        "python",
        "scripts/train_hmm.py",
        "--train",
        portable_path(args.train),
        "--output-dir",
        portable_path(args.output_dir),
        "--train-ids",
        portable_path(args.train_ids),
        "--label-mode",
        args.label_mode,
        "--run-id",
        run_id,
        "--label-config",
        portable_path(args.label_config),
        "--smoothing",
        str(args.smoothing),
        "--min-token-count",
        str(args.min_token_count),
        "--dataset-release-id",
        args.dataset_release_id,
        "--preprocessing-version",
        args.preprocessing_version,
        "--tokenizer-version",
        args.tokenizer_version,
        "--label-policy-version",
        args.label_policy_version,
        "--split-manifest",
        portable_path(args.split_manifest),
    ]
    if args.predict_input is not None:
        command.extend(["--predict-input", portable_path(args.predict_input)])
    if args.predict_ids is not None:
        command.extend(["--predict-ids", portable_path(args.predict_ids)])
    if args.release_manifest is not None:
        command.extend(["--release-manifest", portable_path(args.release_manifest)])
    return command


def _fit_model(
    args: argparse.Namespace,
    labels: list[str],
    train_ids: tuple[str, ...],
    release: dict[str, Any] | None,
    provenance: dict[str, Any],
) -> tuple[HMMModel, dict[str, int], float]:
    """Train the HMM on the selected records; return the model, the selection counts and the training seconds."""
    model = HMMModel(
        label_mode=args.label_mode,
        labels=labels,
        smoothing=args.smoothing,
        min_token_count=args.min_token_count,
    )
    start = perf_counter()
    training = SelectedRecords(
        args.train,
        train_ids,
        training_mode=args.label_mode,
        policy_version=release["policy_version"] if release else None,
        source_revision=release["source"]["revision"] if release else None,
    )
    model.fit(training, metadata=provenance)
    training_seconds = perf_counter() - start
    return model, training.counts, training_seconds


def _predict(
    args: argparse.Namespace, model: HMMModel, prediction_ids: tuple[str, ...] | None
) -> _Prediction | None:
    """Predict the optional prediction input (no gold labels are read); None when there is none."""
    if args.predict_input is None:
        return None
    start = perf_counter()
    selection = SelectedRecords(args.predict_input, prediction_ids)
    records = model.predict(selection)
    return _Prediction(records, selection.counts, perf_counter() - start)


def _verify_inputs_unchanged(args: argparse.Namespace, provenance: dict[str, Any]) -> None:
    """Raise ValueError when an input file changed since its identity was recorded."""
    identities = [
        (args.train, provenance["training_input"]),
        (args.train_ids, provenance["training_ids"]),
        (args.split_manifest, provenance["split_manifest"]),
        (args.label_config, provenance["label_configuration"]),
    ]
    if args.release_manifest is not None:
        identities.append((args.release_manifest, provenance["release_manifest"]))
    if args.predict_input is not None:
        identities.append((args.predict_input, provenance["prediction_input"]))
    if args.predict_ids is not None:
        identities.append((args.predict_ids, provenance["prediction_ids"]))
    for path, identity in identities:
        if sha256_file(path) != identity["sha256"]:
            raise ValueError(f"{path.name}: input changed during the run")


def _save_model(output_dir: Path, model: HMMModel) -> str:
    """Create the run directory, save the model in it and return the model's SHA-256."""
    output_dir.mkdir(parents=True, exist_ok=False)
    model_path = output_dir / "model.json"
    model.save_model(model_path)
    return sha256_file(model_path)


def _write_predictions(
    args: argparse.Namespace,
    labels: list[str],
    provenance: dict[str, Any],
    model_hash: str,
    prediction: _Prediction,
) -> None:
    """Write predictions.jsonl and its manifest into the run directory."""
    prediction_path = args.output_dir / "predictions.jsonl"
    with prediction_path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in prediction.records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    prediction_manifest = {
        "schema_version": 1,
        "run_id": provenance["run_id"],
        "model_name": "hmm",
        "label_mode": args.label_mode,
        "ordered_labels": labels,
        "model_sha256": model_hash,
        "input": provenance["prediction_input"],
        "ids": file_identity(args.predict_ids) if args.predict_ids else None,
        "predictions_sha256": sha256_file(prediction_path),
        "num_sequences": len(prediction.records),
        "num_tokens": prediction.counts["selected_tokens"],
        "prediction_seconds_including_input": prediction.seconds,
        "selection_counts": prediction.counts,
        "provenance": provenance,
    }
    (args.output_dir / "predictions.manifest.json").write_text(
        json.dumps(prediction_manifest, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _run(args: argparse.Namespace) -> None:
    """Validate the inputs, train, optionally predict, then write every run output."""
    train_ids, prediction_ids, labels, release = _read_selections(args)
    provenance = _build_provenance(args, train_ids, labels)
    provenance["command"] = _training_command(args, provenance["run_id"])
    started_at = datetime.now(UTC).isoformat()
    model, training_counts, training_seconds = _fit_model(args, labels, train_ids, release, provenance)
    prediction = _predict(args, model, prediction_ids)
    _verify_inputs_unchanged(args, provenance)
    # All input validation finishes before creating any run outputs.
    model_hash = _save_model(args.output_dir, model)
    if prediction is not None:
        _write_predictions(args, labels, provenance, model_hash, prediction)
    run = {
        "model_name": "hmm",
        "started_at_utc": started_at,
        "training_seconds": training_seconds,
        "training_summary": model.training_summary,
        "selection_counts": training_counts,
        "model_file": "model.json",
        "model_sha256": model_hash,
        "provenance": provenance,
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.predict_ids is not None and args.predict_input is None:
        parser.error("--predict-ids requires --predict-input")
    try:
        _run(args)
    except (OSError, ValueError) as exc:
        print(f"error: {one_line_error(exc)}", file=sys.stderr)
        return 1
    print(f"Saved HMM to {args.output_dir / 'model.json'}")
    print("No evaluation metrics calculated; score the predictions with: "
          "python scripts/evaluate_models.py score --gold ... --ids ... --predictions ... --out ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
