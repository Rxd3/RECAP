#!/usr,bin/env python3
"""
CLI Demonstration Runner for Sequence Labeling.
Usage: python scripts/run_demo.py --model-path artifacts/models/crf.joblib --config configs/mode_a.yaml
"""

import argparse
import sys
from pathlib import Path
from punct_restore.pipeline import ModelAdapter, RestorationPipeline


def main():
    parser = argparse.ArgumentParser(description="RECAP Punctuation & Casing Restoration CLI Demo")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to trained model artifact")
    parser.add_argument("--config-path", type=Path, required=True, help="Path to experiment YAML config")
    parser.add_argument("--input-text", type=str, default=None, help="Text to process")
    
    args = parser.parse_args()

    try:
        adapter = ModelAdapter(model_artifact_path=args.model_path, config_path=args.config_path)
        pipeline = RestorationPipeline(adapter=adapter)
    except Exception as e:
        print(f"[ERROR] Failed to initialize pipeline: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- RECAP Interactive Demonstration ---")
    if args.input_text:
        result = pipeline.process_text(args.input_text)
        print(f"Input:    {result['input_text']}")
        print(f"Restored: {result['restored_text']}\n")
    else:
        print("Type unpunctuated text (or 'quit' to exit):\n")
        while True:
            try:
                user_input = input(">> ").strip()
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if not user_input:
                    continue
                result = pipeline.process_text(user_input)
                print(f"Restored: {result['restored_text']}\n")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()