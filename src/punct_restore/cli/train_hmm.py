"""Train Member 2's HMM from shared-contract JSONL, without downloading data."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from punct_restore.models.hmm import HMMModel
from punct_restore.models.hmm_workflow import (
    code_identity,
    file_identity,
    portable_path,
    read_id_selection,
    selected_records,
    sha256,
    verify_training_release,
)
from punct_restore.paths import REPO_ROOT as ROOT


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise ValueError("output-dir already exists; choose a new run directory")
        if args.predict_ids is not None and args.predict_input is None:
            raise ValueError("predict-ids requires predict-input")
        train_ids = read_id_selection(args.train_ids)
        prediction_ids = None if args.predict_ids is None else read_id_selection(args.predict_ids)
        labels = load_labels(args.label_config, args.label_mode)
        release = None
        if args.release_manifest is not None:
            if not (
                args.preprocessing_version == args.tokenizer_version == args.label_policy_version
            ):
                raise ValueError(
                    "this release uses one policy version for preprocessing/tokens/labels"
                )
            release = verify_training_release(
                args.release_manifest,
                train=args.train,
                train_ids=args.train_ids,
                split_manifest=args.split_manifest,
                label_config=args.label_config,
                mode=args.label_mode,
                release_id=args.dataset_release_id,
                policy_version=args.label_policy_version,
                prediction=args.predict_input,
                prediction_ids=args.predict_ids,
            )
        provenance = {
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
            provenance["run_id"],
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
        provenance["command"] = command
        started_at = datetime.now(UTC).isoformat()
        model = HMMModel(
            label_mode=args.label_mode,
            labels=labels,
            smoothing=args.smoothing,
            min_token_count=args.min_token_count,
        )
        start = perf_counter()
        training_counts = {}
        model.fit(
            selected_records(
                args.train,
                train_ids,
                training_mode=args.label_mode,
                policy_version=release["policy_version"] if release else None,
                source_revision=release["source"]["revision"] if release else None,
                counts=training_counts,
            ),
            metadata=provenance,
        )
        training_seconds = perf_counter() - start
        predictions = None
        prediction_input = None
        prediction_seconds = None
        prediction_counts = {}
        if args.predict_input is not None:
            prediction_input = provenance["prediction_input"]
            start = perf_counter()
            predictions = model.predict(
                selected_records(args.predict_input, prediction_ids, counts=prediction_counts)
            )
            prediction_seconds = perf_counter() - start
        identities = [
            (args.train, provenance["training_input"]),
            (args.train_ids, provenance["training_ids"]),
            (args.split_manifest, provenance["split_manifest"]),
            (args.label_config, provenance["label_configuration"]),
        ]
        if args.release_manifest is not None:
            identities.append((args.release_manifest, provenance["release_manifest"]))
        if args.predict_input is not None:
            identities.append((args.predict_input, prediction_input))
        if args.predict_ids is not None:
            identities.append((args.predict_ids, provenance["prediction_ids"]))
        for path, identity in identities:
            if sha256(path) != identity["sha256"]:
                raise ValueError(f"{path.name}: input changed during the run")
        # All input validation finishes before creating any run outputs.
        args.output_dir.mkdir(parents=True, exist_ok=False)
        model_path = args.output_dir / "model.json"
        model.save_model(model_path)
        model_hash = sha256(model_path)
        if predictions is not None:
            prediction_path = args.output_dir / "predictions.jsonl"
            with prediction_path.open("x", encoding="utf-8", newline="\n") as handle:
                for record in predictions:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            prediction_manifest = {
                "schema_version": 1,
                "run_id": provenance["run_id"],
                "model_name": "hmm",
                "label_mode": args.label_mode,
                "ordered_labels": labels,
                "model_sha256": model_hash,
                "input": prediction_input,
                "ids": file_identity(args.predict_ids) if args.predict_ids else None,
                "predictions_sha256": sha256(prediction_path),
                "num_sequences": len(predictions),
                "num_tokens": prediction_counts["selected_tokens"],
                "prediction_seconds_including_input": prediction_seconds,
                "selection_counts": prediction_counts,
                "provenance": provenance,
            }
            (args.output_dir / "predictions.manifest.json").write_text(
                json.dumps(prediction_manifest, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
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
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Saved HMM to {args.output_dir / 'model.json'}")
    print("No evaluation metrics calculated; score the predictions with: "
          "python scripts/evaluate_models.py score --gold ... --ids ... --predictions ... --out ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
