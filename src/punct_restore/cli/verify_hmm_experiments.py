"""Independently audit HMM runs, refit frozen winners and reproduce every prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from punct_restore.models.hmm_experiment_verify import verify
from punct_restore.prediction_io import one_line_error

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recap verify-hmm-experiments", description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        verify(args.run_dir, args.evaluator_root, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"error: {one_line_error(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
