"""Train/validation-select T5 or export its frozen, word-aligned predictions."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from punct_restore.models.seq2seq import T5Seq2Seq
from punct_restore.models.neural_workflow import main

if __name__ == "__main__":
    raise SystemExit(main(T5Seq2Seq))
