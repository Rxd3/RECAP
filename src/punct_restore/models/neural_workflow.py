"""Train/validation selection and frozen inference for the optional neural comparisons.

No test file is read during training or selection. Neural imports live inside
the operations that need them, so the classical installation stays independent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

from punct_restore.evaluation.inputs import sha256_file
from punct_restore.prediction_io import (check_new_output_paths, read_frozen_ids,
                                        read_observations, write_prediction_outputs)
from .neural_utils import labels_for_record, observations

ROOT = Path(__file__).resolve().parents[3]


def relative(path):
    """Persist portable paths; artifact/selection references use their own base instead."""
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def identity(path):
    return {"path": relative(path), "sha256": sha256_file(path)}


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def new_outputs(outputs, inputs):
    check_new_output_paths(outputs, inputs)
    for path in outputs.values():
        if path is not None and Path(path).exists():
            raise ValueError(f"refusing to overwrite {Path(path).name}")


def training_records(path, ids_path, mode):
    selected = set(read_frozen_ids(ids_path))
    result, seen = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("sample_id")
            if sample_id in seen:
                raise ValueError("duplicate training sample_id")
            seen.add(sample_id)
            if sample_id not in selected:
                continue
            if record.get("split") != "train" or record.get("label_mode") != mode:
                raise ValueError("training requires the selected mode and split train")
            observed = observations([record])[0]
            labels_for_record(record, mode)
            result.append({**observed, "punct_labels": record["punct_labels"],
                           "case_labels": record.get("case_labels", [])})
    if len(result) != len(selected):
        raise ValueError("training source is missing frozen IDs")
    return result


def environment(device):
    import torch

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "sentencepiece", "tokenizers", "safetensors")},
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "cuda": torch.version.cuda, "device": device,
            "code_commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "source_sha256": {relative(p): sha256_file(p) for p in [
                *Path(__file__).parent.glob("neural*.py"), Path(__file__).with_name("seq2seq.py"),
                ROOT / "scripts/train_neural_tagger.py", ROOT / "scripts/train_seq2seq.py",
                ROOT / "configs/label_sets.yaml", ROOT / "pyproject.toml"]}}


def synchronize(model):
    if model.device == "cuda":
        import torch
        torch.cuda.synchronize()


def benchmark(model, records, batch_size):
    """Full validation warm-up, then median of three end-to-end in-process passes."""
    model.predict(records, batch_size=batch_size)
    synchronize(model)
    passes = []
    for _ in range(3):
        synchronize(model)
        start = time.perf_counter()
        model.predict(records, batch_size=batch_size)
        synchronize(model)
        passes.append(1000 * (time.perf_counter() - start) / len(records))
    return {"scope": "tokenization + inference/generation + alignment; excludes model/data loading",
            "warmup_passes": 1, "timed_passes": 3, "batch_size": batch_size,
            "ms_per_utterance_passes": passes, "median_ms_per_utterance": statistics.median(passes)}


def export_predictions(model, records, input_path, ids_path, output, manifest, provenance):
    new_outputs({"--output": output, "--manifest-out": manifest},
                {"--input": input_path, "--ids": ids_path})
    synchronize(model)
    start = time.perf_counter()
    predictions = model.predict(records, batch_size=provenance["batch_size"])
    synchronize(model)
    seconds = time.perf_counter() - start
    write_prediction_outputs(predictions, output, manifest, input_path=input_path,
                             model_name=model.model_name, label_mode=model.label_mode,
                             ordered_labels=model.labels)
    # Keep the official writer's mandatory fields; add reproducibility evidence.
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["input"]["path"] = relative(input_path)
    document.update({"schema_version": 1, "ids": identity(ids_path),
                     "diagnostics": model.last_diagnostics, "prediction_seconds": seconds,
                     **provenance})
    manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def score(input_path, ids_path, predictions, manifest, output, masks=None, bootstrap=0):
    command = [sys.executable, str(ROOT / "scripts/evaluate_models.py"), "score",
               "--gold", str(input_path), "--ids", str(ids_path), "--predictions", str(predictions),
               "--manifest", str(manifest), "--out", str(output)]
    if masks:
        command.extend(["--masks", str(masks)])
    if bootstrap:
        command.extend(["--bootstrap", str(bootstrap)])
    subprocess.run(command, check=True, cwd=ROOT)
    return json.loads(output.read_text(encoding="utf-8"))


def choose(candidates):
    # Stable ties: higher official primary, fewer epochs, then smaller LR.
    return min(candidates, key=lambda c: (-c["primary"]["value"], c["epoch"], c["learning_rate"]))


def train_epoch(model, loader, optimizer, scaler, accumulation):
    import torch

    model.backend.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for index, batch in enumerate(loader):
        # The final short accumulation group uses its actual batch count.
        group_start = (index // accumulation) * accumulation
        group_size = min(accumulation, len(loader) - group_start)
        with model.autocast():
            loss = model.backend(**model.to_device(batch)).loss
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss; no checkpoint selected")
        scaler.scale(loss / group_size).backward()
        total_loss += loss.item()
        if (index + 1) % accumulation == 0 or index + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.backend.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if (index + 1) % 200 == 0:
            print(f"batches {index + 1}/{len(loader)} loss {total_loss / (index + 1):.4f}", flush=True)
    return total_loss / len(loader)


def train(args, model_class):
    import torch

    # These checks run before any data is opened. Held-out inputs are never accepted.
    if args.input.name != "train.jsonl" or args.ids.name != f"mode_{args.mode}.ids" or args.ids.parent.name != "default_train":
        raise ValueError("training requires train.jsonl and default_train/mode_<X>.ids")
    if args.validation_input.name != "validation.jsonl" or args.validation_ids.name != f"validation_mode_{args.mode}.ids" or args.validation_ids.parent.name != "primary_eval":
        raise ValueError("selection requires validation.jsonl and primary_eval/validation_mode_<X>.ids")
    if (args.mode == "C") != (args.masks is not None):
        raise ValueError("--masks is required for C and only C")
    if args.masks and args.masks.name != "validation.jsonl":
        raise ValueError("selection masks must be validation.jsonl")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; install the CUDA build documented in REPRODUCIBILITY.md")
    outputs = {"--output": args.output, "--manifest-out": args.manifest_out,
               "--selection-out": args.selection_out}
    new_outputs(outputs, {"--input": args.input, "--ids": args.ids,
                         "--validation-input": args.validation_input, "--validation-ids": args.validation_ids,
                         "--masks": args.masks})
    # Freeze references are relative to the run directory; keep the three outputs together.
    if args.selection_out.parent.resolve() != args.output.resolve() or args.manifest_out.parent.resolve() != args.output.resolve():
        raise ValueError("selection and training manifest must be directly inside --output")
    args.output.mkdir(parents=True, exist_ok=False)
    records = training_records(args.input, args.ids, args.mode)
    validation = read_observations(args.validation_input, args.validation_ids)
    settings = {"seed": 2026, "epochs": args.epochs, "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation, "optimizer": "AdamW",
                "weight_decay": 0.01, "gradient_clip_norm": 1.0, "schedule": "constant",
                "precision": "float32" if args.device == "cpu" else
                    ("fp16" if model_class.model_name == "bert_tagger" else
                     ("bf16" if torch.cuda.is_bf16_supported() else "float32")),
                "num_beams": args.num_beams,
                "word_alignment": "first complete subword; truncate at 512; default cut words" if
                    model_class.model_name == "bert_tagger" else "whitespace + SequenceMatcher matching blocks; D5 projection",
                "pretrained": args.pretrained or model_class.default_pretrained,
                "dataset_release": "sponspeech-m1-v2", "policy_version": "m1-v2"}
    evidence = {"schema_version": 1, "model_name": model_class.model_name, "label_mode": args.mode,
                "run_id": args.run_id, "settings": settings, "environment": environment(args.device),
                "command": [relative(item) if Path(item).is_absolute() else item for item in sys.argv],
                "inputs": {"training": identity(args.input), "training_ids": identity(args.ids),
                           "validation": identity(args.validation_input), "validation_ids": identity(args.validation_ids),
                           "masks": identity(args.masks) if args.masks else None},
                "training_sequences": len(records), "training_words": sum(len(r["tokens"]) for r in records)}
    candidates = []
    rates = [2e-5, 5e-5] if model_class.model_name == "bert_tagger" else [args.learning_rate]
    for rate in rates:
        random.seed(2026)
        torch.manual_seed(2026)
        if args.device == "cuda":
            torch.cuda.manual_seed_all(2026)
        model = model_class.from_pretrained(args.mode, args.pretrained, revision=args.revision)
        model.num_beams = args.num_beams
        model.configure_device(args.device)
        model.backend.gradient_checkpointing_enable()
        if model.model_name == "t5_seq2seq":
            model.backend.config.use_cache = False
        loader = torch.utils.data.DataLoader(records, batch_size=args.batch_size, shuffle=True,
                                             collate_fn=model.training_batch, num_workers=0,
                                             generator=torch.Generator().manual_seed(2026))
        optimizer = torch.optim.AdamW(model.backend.parameters(), lr=rate, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=model.fp16)
        cumulative = 0.0
        for epoch in range(1, args.epochs + 1):
            model.training_diagnostics = {"words": 0, "cutoff_words": 0}
            synchronize(model)
            start = time.perf_counter()
            loss = train_epoch(model, loader, optimizer, scaler, args.gradient_accumulation)
            synchronize(model)
            duration = time.perf_counter() - start
            cumulative += duration
            prefix = f"lr{rate:g}-epoch{epoch}"
            directory = args.output / prefix
            directory.mkdir()
            artifact = directory / "model"
            model.save_model(artifact, metadata={**evidence, "learning_rate": rate, "epoch": epoch,
                                                "training_seconds": cumulative})
            predictions, manifest = directory / "validation.jsonl", directory / "validation.manifest.json"
            info = export_predictions(model, validation, args.validation_input, args.validation_ids,
                                      predictions, manifest, {"batch_size": args.batch_size,
                                      "run_id": args.run_id, "environment": evidence["environment"],
                                      "artifact": identity(artifact / "neural_metadata.json")})
            report = score(args.validation_input, args.validation_ids, predictions, manifest,
                           directory / "validation.score.json", args.masks)
            candidate = {"learning_rate": rate, "epoch": epoch, "primary": report["primary"],
                         "pretrained": model.pretrained, "pretrained_revision": model.revision,
                         "training_seconds": cumulative, "epoch_seconds": duration, "mean_batch_loss": loss,
                         "training_diagnostics": dict(model.training_diagnostics),
                         "artifact": f"{prefix}/model", "diagnostics": info["diagnostics"],
                         "score": identity(directory / "validation.score.json"),
                         "artifact_metadata_sha256": sha256_file(artifact / "neural_metadata.json")}
            candidates.append(candidate)
            write_json(directory / "candidate.json", candidate)
            print(json.dumps(candidate), flush=True)
        del optimizer, scaler, loader, model
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    selected = choose(candidates)
    selection = {**evidence, "frozen": True, "frozen_at": datetime.now(timezone.utc).isoformat(),
                 "training_manifest": args.manifest_out.name,
                 "criterion": "official evaluator primary macro F1; ties: fewer epochs, smaller LR",
                 "candidates": candidates, "selected": selected,
                 "search_training_seconds": sum(c["epoch_seconds"] for c in candidates)}
    write_json(args.selection_out, selection)  # Durable freeze precedes any final split work.
    model = model_class.load_model(args.output / selected["artifact"]).configure_device(args.device)
    runtime = benchmark(model, validation, args.batch_size)
    export_predictions(model, validation, args.validation_input, args.validation_ids,
                       args.output / "validation.jsonl", args.output / "validation.manifest.json",
                       {"batch_size": args.batch_size, "run_id": args.run_id,
                        "selection": identity(args.selection_out), "runtime": runtime,
                        "environment": evidence["environment"], "artifact": selected})
    score(args.validation_input, args.validation_ids, args.output / "validation.jsonl",
          args.output / "validation.manifest.json", args.output / "validation.score.json", args.masks, 1000)
    append_notes(args.notes_out, {**selection, "validation_runtime": runtime})
    write_json(args.manifest_out, {**selection, "validation_runtime": runtime})


def append_notes(path, evidence):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"\n## Frozen run {evidence['run_id']}\n\n```json\n")
        handle.write(json.dumps(evidence, indent=2, sort_keys=True) + "\n```\n")


def frozen_selection(path, model_path, model_class):
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("frozen") is not True or selection.get("model_name") != model_class.model_name:
        raise ValueError("a frozen selection for this model family is required")
    selected = selection["selected"]
    if (path.parent / selected["artifact"]).resolve() != model_path.resolve():
        raise ValueError("--model must be the frozen selected artifact")
    if sha256_file(model_path / "neural_metadata.json") != selected["artifact_metadata_sha256"]:
        raise ValueError("selected model metadata checksum mismatch")
    completion = json.loads((path.parent / selection["training_manifest"]).read_text(encoding="utf-8"))
    if completion.get("selected") != selected or "validation_runtime" not in completion:
        raise ValueError("selection has no completed training and runtime record")
    return selection


def predict(args, model_class):
    # Validate the freeze before reading the requested input, including held-out files.
    selection = frozen_selection(args.selection, args.model, model_class)
    split = args.input.stem
    mode = selection["label_mode"]
    if split not in ("validation", "test", "test_amb") or args.ids.name != f"{split}_mode_{mode}.ids" or args.ids.parent.name != "primary_eval":
        raise ValueError("prediction requires the matching primary_eval ID list")
    new_outputs({"--output": args.output, "--manifest-out": args.manifest_out},
                {"--input": args.input, "--ids": args.ids, "--selection": args.selection})
    receipt = args.selection.parent / f"{split}.prediction-receipt.json"
    if split != "validation" and receipt.exists():
        raise ValueError(f"{split} inference already attempted for this frozen model")
    model = model_class.load_model(args.model).configure_device(args.device)
    if model.label_mode != mode:
        raise ValueError("selection label mode differs from artifact")
    if split != "validation":
        # Exclusive reservation also prevents concurrent duplicate held-out inference.
        write_json(receipt, {"status": "started", "selection": identity(args.selection),
                             "output": relative(args.output)})
    records = read_observations(args.input, args.ids)
    export_predictions(model, records, args.input, args.ids, args.output, args.manifest_out,
                       {"batch_size": args.batch_size, "run_id": selection["run_id"],
                        "selection": identity(args.selection), "artifact": selection["selected"],
                        "environment": environment(args.device)})
    if split != "validation":
        receipt.write_text(json.dumps({"status": "complete", "selection": identity(args.selection),
                                      "predictions": identity(args.output), "manifest": identity(args.manifest_out)},
                                     indent=2) + "\n", encoding="utf-8")


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(model_class, argv=None):
    parser = argparse.ArgumentParser(description=f"{model_class.model_name}: validation-selected neural comparison")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="train and select on validation only")
    inference = commands.add_parser("predict", help="predict with a frozen selected model")
    for command in (training, inference):
        for flag in ("input", "ids", "output", "manifest-out"):
            command.add_argument(f"--{flag}", type=Path, required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--batch-size", type=positive, default=4)
    training.add_argument("--mode", choices=("A", "B", "C"), required=True)
    for flag in ("validation-input", "validation-ids", "selection-out"):
        training.add_argument(f"--{flag}", type=Path, required=True)
    training.add_argument("--masks", type=Path)
    training.add_argument("--pretrained", choices=(model_class.default_pretrained,) if model_class.model_name == "bert_tagger"
                          else ("google/flan-t5-base", "t5-small"))
    training.add_argument("--revision", default="main")
    training.add_argument("--epochs", type=int, choices=(1, 2, 3), default=3)
    training.add_argument("--gradient-accumulation", type=positive, default=4)
    training.add_argument("--learning-rate", type=float, choices=(1e-4, 3e-4), default=1e-4,
                          help="seq2seq LR; BERT always searches 2e-5 and 5e-5")
    training.add_argument("--num-beams", type=int, choices=(1, 4), default=4)
    training.add_argument("--run-id", required=True)
    training.add_argument("--notes-out", type=Path, default=ROOT / "experiments/NEURAL_NOTES.md")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        (train if args.command == "train" else predict)(args, model_class)
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0
