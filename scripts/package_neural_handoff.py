"""Verify all 18 frozen prediction pairs with the official scorer, then ZIP with SHA256SUMS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from punct_restore.models.neural_tagger import BertTagger
from punct_restore.models.seq2seq import T5Seq2Seq
from punct_restore.models.neural_workflow import frozen_selection, new_outputs, score
from punct_restore.evaluation.inputs import sha256_file


def package(run_root, data_root, output):
    score_dir = output.parent / f"{output.stem}-scores"
    new_outputs({"--output": output, "scores": score_dir}, {})
    files, frozen = [], []
    # Every family/mode must be frozen before this command can read any test gold.
    for model_class in (BertTagger, T5Seq2Seq):
        for mode in ("A", "B", "C"):
            directory = run_root / f"{model_class.model_name}_{mode}"
            selection_path = directory / "selection.json"
            selected = json.loads(selection_path.read_text(encoding="utf-8"))["selected"]
            frozen_selection(selection_path, directory / selected["artifact"], model_class)
            frozen.append((directory, mode))
    for directory, mode in frozen:
        for split in ("validation", "test", "test_amb"):
            for suffix in ("jsonl", "manifest.json"):
                path = directory / f"{split}.{suffix}"
                if not path.is_file():
                    raise ValueError(f"missing {directory.name}/{path.name}")
                files.append(path)
    score_dir.mkdir(parents=True)
    for directory, mode in frozen:
        for split in ("validation", "test", "test_amb"):
            release = data_root / "processed/sponspeech-m1-v2"
            score(release / f"mode_{mode}/{split}.jsonl",
                  data_root / f"splits/sponspeech-m1-v2/primary_eval/{split}_mode_{mode}.ids",
                  directory / f"{split}.jsonl", directory / f"{split}.manifest.json",
                  score_dir / f"{directory.name}_{split}.json",
                  release / f"masks/case_joint/{split}.jsonl" if mode == "C" else None, 1000)
    checksums = "".join(f"{sha256_file(path)}  {path.relative_to(run_root).as_posix()}\n" for path in files)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(run_root).as_posix())
        archive.writestr("SHA256SUMS", checksums)
    print(f"{sha256_file(output)}  {output.name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("experiments/runs/neural"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        package(args.run_root, args.data_root, args.output)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
