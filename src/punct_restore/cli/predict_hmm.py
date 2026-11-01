"""Export HMM predictions from a saved artifact, without training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from punct_restore.models.hmm_workflow import portable_path, predict_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recap predict-hmm", description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--ids", type=Path)
    selection.add_argument(
        "--all-records", action="store_true", help="explicitly predict every input sequence"
    )
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument(
        "--selection-record", type=Path, help="bind artifact to frozen validation winner"
    )
    args = parser.parse_args(argv)
    command = [
        "python",
        "scripts/predict_hmm.py",
        "--model",
        portable_path(args.model),
        "--input",
        portable_path(args.input),
        "--output-dir",
        portable_path(args.output_dir),
    ]
    command += ["--ids", portable_path(args.ids)] if args.ids else ["--all-records"]
    if args.release_manifest:
        command += ["--release-manifest", portable_path(args.release_manifest)]
    if args.selection_record:
        command += ["--selection-record", portable_path(args.selection_record)]
    try:
        report = predict_run(
            args.model,
            args.input,
            args.output_dir,
            ids_path=args.ids,
            release_manifest=args.release_manifest,
            selection_record=args.selection_record,
            command=command,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Exported {report['num_sequences']} HMM predictions; no training performed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
