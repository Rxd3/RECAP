#!/usr/bin/env python3
"""CLI demo: restore punctuation and casing in unpunctuated text with a trained model.

Usage:
    python scripts/run_demo.py --model-path artifacts/models/<model>.json|.joblib --text "..."

Without --text it starts an interactive prompt (type quit, exit or q to leave).
Only load trusted artifacts: .joblib/.pkl model files are pickles and can run code on load.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Support the documented source-checkout command before an editable install.
sys.path.insert(0, str(ROOT / "src"))
from punct_restore.pipeline import ModelAdapter, RestorationPipeline  # noqa: E402


def _error_line(exc: Exception) -> str:
    return "error: " + " ".join(str(exc).splitlines())


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
            print(_error_line(exc), file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RECAP punctuation and casing restoration demo")
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
        print(_error_line(exc), file=sys.stderr)
        return 1
    print(f"Input:    {result['input_text']}")
    print(f"Restored: {result['restored_text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
