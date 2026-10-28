"""Train Member 2's HMM from shared-contract JSONL, without downloading data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Support the documented source-checkout command before an editable install.
sys.path.insert(0, str(ROOT / "src"))
from punct_restore.models.hmm import HMMModel  # noqa: E402


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number}: blank JSONL record")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path.name}:{line_number}: record must be a JSON object")
            yield record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    try:
        name = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        name = path.name
    return name


def file_identity(path: Path) -> dict[str, str]:
    return {"path": portable_path(path), "sha256": sha256(path)}


def code_identity() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="shared-contract train JSONL")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="new run directory; never overwritten"
    )
    parser.add_argument("--label-mode", choices=["A", "B", "C"], default="A")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise ValueError("output-dir already exists; choose a new run directory")
        labels = load_labels(args.label_config, args.label_mode)
        provenance = {
            "schema_version": 1,
            "dataset_release_id": args.dataset_release_id,
            "preprocessing_version": args.preprocessing_version,
            "tokenizer_version": args.tokenizer_version,
            "label_policy_version": args.label_policy_version,
            "training_input": file_identity(args.train),
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
            "--label-mode",
            args.label_mode,
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
        provenance["command"] = command
        started_at = datetime.now(UTC).isoformat()
        model = HMMModel(
            label_mode=args.label_mode,
            labels=labels,
            smoothing=args.smoothing,
            min_token_count=args.min_token_count,
        )
        start = perf_counter()
        model.fit(read_jsonl(args.train), metadata=provenance)
        training_seconds = perf_counter() - start
        predictions = None
        prediction_input = None
        if args.predict_input is not None:
            prediction_input = file_identity(args.predict_input)
            predictions = model.predict(read_jsonl(args.predict_input))
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
                "model_name": "hmm",
                "label_mode": args.label_mode,
                "ordered_labels": labels,
                "model_sha256": model_hash,
                "input": prediction_input,
                "predictions_sha256": sha256(prediction_path),
                "num_sequences": len(predictions),
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
            "model_file": "model.json",
            "model_sha256": model_hash,
            "provenance": provenance,
        }
        (args.output_dir / "run.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Saved HMM to {args.output_dir / 'model.json'}")
    print("No evaluation metrics calculated; use the shared evaluation pipeline when available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
