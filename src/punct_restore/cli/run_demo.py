#!/usr/bin/env python3
"""CLI demo: restore punctuation and casing in unpunctuated text with a trained model.

Usage:
    recap run-demo --model-path <model>.json|.joblib --text "..."

Without --text it starts an interactive prompt (type quit, exit or q to leave).
Only load trusted artifacts: .joblib/.pkl model files are pickles and can run code on load.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from punct_restore.pipeline import ModelAdapter, RestorationPipeline
from punct_restore.prediction_io import one_line_error, reconfigure_stdout


def _interactive(pipeline: RestorationPipeline) -> None:
    print("Type unpunctuated text (or 'quit' to exit):\n")
    while True:
        try:
            line = input(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            return
        if line.lower() in ("quit", "exit", "q"):
            return
        if not line:
            continue
        try:
            print(f"Restored: {pipeline.process_text(line)['restored_text']}\n")
        except ValueError as exc:
            print(f"error: {one_line_error(exc)}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    reconfigure_stdout()  # a cp1252 console must not crash on characters outside its code page
    parser = argparse.ArgumentParser(prog="recap run-demo", description="RECAP punctuation and casing restoration demo")
    parser.add_argument("--model-path", type=Path, required=True, help="trained model (.json, .joblib or .pkl)")
    parser.add_argument("--text", type=str, default=None, help="text to restore (omit for interactive mode)")
    args = parser.parse_args(argv)

    try:
        pipeline = RestorationPipeline(ModelAdapter(args.model_path))
        if args.text is None:
            _interactive(pipeline)
            return 0
        result = pipeline.process_text(args.text)
    except (ValueError, OSError) as exc:  # OSError covers FileNotFoundError and directories
        print(f"error: {one_line_error(exc)}", file=sys.stderr)
        return 1
    print(f"Input:    {result['input_text']}")
    print(f"Restored: {result['restored_text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
