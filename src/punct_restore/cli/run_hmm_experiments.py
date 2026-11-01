"""Run the HMM study: frozen validation search (tune) and held-out evaluation (finalize).

The evaluator is the evaluation code (punct_restore.evaluation) in a separate checkout pinned by config.
Tuning never opens test records. Finalization requires every configured mode's selection.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

from punct_restore.models.hmm_experiment_commands import HmmCommandFailed
from punct_restore.models.hmm_experiment_finalize import finalize
from punct_restore.models.hmm_experiment_tune import tune
from punct_restore.prediction_io import one_line_error

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recap run-hmm-experiments", description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    tune_parser = commands.add_parser("tune")
    tune_parser.add_argument("--config", type=Path, required=True)
    tune_parser.add_argument("--output-dir", type=Path, required=True)
    tune_parser.add_argument("--evaluator-root", type=Path, required=True)
    final_parser = commands.add_parser("finalize")
    final_parser.add_argument("--run-dir", type=Path, required=True)
    final_parser.add_argument("--evaluator-root", type=Path, required=True)
    args = parser.parse_args(argv)
    progress = partial(print, flush=True)
    try:
        if args.phase == "tune":
            tune(args.config, args.evaluator_root, args.output_dir, log=progress)
        else:
            finalize(args.run_dir, args.evaluator_root, log=progress)
    except (OSError, ValueError, HmmCommandFailed) as exc:
        print(f"error: {one_line_error(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
