"""How far was CRF training from converging? Read the loss trace stored in each saved model.

Usage: python scripts/reporting/crf_convergence.py --models PATH [PATH ...] --out OUT_JSON
Each PATH is a CRF .joblib file, or a folder whose *.joblib files are all read (sorted by name). The report holds
one entry per model: file name, model_sha256, model_name, label_mode, the number of optimizer iterations, the final
loss, the loss 10 iterations earlier and the relative drop over those last 10 iterations
((loss_10_before - final_loss) / loss_10_before). With fewer than 11 iterations the last two are null.
The loss trace is ``backend_state.training_log_.iterations`` of the saved bundle (sklearn-crfsuite).
.joblib files are pickles and can run code when opened: trusted files only. OUT_JSON must not exist.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib

sys.path.insert(0, str(Path(__file__).parent))
from _cli import check_out, run, write_new_text  # puts ROOT/src on sys.path

from punct_restore.evaluation.inputs import sha256_file

LOOKBACK = 10  # the drop is measured between iteration n - 10 and iteration n


def model_files(paths: Sequence[Path]) -> list[Path]:
    """The model files named by ``paths``: a file as given, a folder as its sorted ``*.joblib`` files."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            found = sorted(p for p in path.glob("*.joblib") if p.is_file())
            if not found:
                raise ValueError(f"{path.name}: no .joblib files in this folder")
            files.extend(found)
        elif path.is_file():
            files.append(path)
        else:
            raise ValueError(f"{path.name}: model file not found")
    return files


def read_bundle(path: Path) -> dict[str, Any]:
    """The saved CRF bundle (a dict); anything else is refused by file name."""
    try:
        bundle = joblib.load(path)  # trusted files only: this is a pickle
    except OSError:
        raise  # run() names the file and the system reason (e.g. "is a folder, not a file")
    except ImportError as exc:  # the pickle names a class from a package that is not installed
        raise ValueError(f"{path.name}: needs the Python package {exc.name} to open it") from exc
    except Exception as exc:  # a pickle can fail in many ways; the message would quote the whole path
        raise ValueError(f"{path.name}: not a readable joblib file") from exc
    if not isinstance(bundle, dict) or bundle.get("model_family") != "crf":
        raise ValueError(f"{path.name}: not a CRF model file")
    return bundle


def read_losses(bundle: dict[str, Any], path: Path) -> list[float]:
    """The loss of every optimizer iteration, in order."""
    log = getattr(bundle.get("backend_state"), "training_log_", None)
    iterations = getattr(log, "iterations", None)
    if not isinstance(iterations, list) or not iterations:
        raise ValueError(f"{path.name}: the model holds no training log")
    losses = [step.get("loss") if isinstance(step, dict) else None for step in iterations]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in losses):
        raise ValueError(f"{path.name}: the training log has an iteration without a finite loss")
    return losses


def convergence_entry(path: Path) -> dict[str, Any]:
    """The report entry of one model file."""
    bundle = read_bundle(path)
    losses = read_losses(bundle, path)
    final = losses[-1]
    before = losses[-1 - LOOKBACK] if len(losses) > LOOKBACK else None
    drop = (before - final) / before if before else None  # no drop to report from a missing or zero loss
    return {"file": path.name, "model_sha256": sha256_file(path), "model_name": bundle.get("model_name"),
            "label_mode": bundle.get("label_mode"), "iterations": len(losses), "final_loss": final,
            "loss_10_before": before, "relative_drop_last_10": drop}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report the CRF training loss trace (iterations, final loss, drop over the last 10 iterations). "
                    ".joblib files are pickles: trusted files only.")
    parser.add_argument("--models", type=Path, nargs="+", required=True, metavar="PATH",
                        help="CRF .joblib files, or folders whose *.joblib files are all read (trusted files only)")
    parser.add_argument("--out", type=Path, required=True, help="JSON file to create (must not exist)")
    return parser


def write_report(args: argparse.Namespace) -> None:
    files = model_files(args.models)
    check_out(args.out, {f"--models ({path.name})": path for path in files})  # before loading anything
    entries = [convergence_entry(path) for path in files]
    write_new_text(args.out, json.dumps(entries, indent=1) + "\n")
    print(f"wrote {len(entries)} model {'entry' if len(entries) == 1 else 'entries'} to {args.out.name}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(lambda: write_report(args), json_errors=False)  # it opens pickles: no input-JSON relabelling


if __name__ == "__main__":
    sys.exit(main())
