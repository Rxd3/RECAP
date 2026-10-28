"""Shared evaluation for RECAP models, owned by Member 5 (docs/EVALUATION_PROTOCOL.md).

load_bundle() checks and aligns gold, ID list, predictions and masks; evaluate() scores every
block of the mode; to_markdown() summarizes a report; shortcut_predictions() builds the
shortcut reference baseline.
"""

from .baseline import shortcut_predictions
from .inputs import EvaluationInputError, load_bundle, load_scored_gold, load_vocab
from .report import evaluate, to_markdown

__all__ = ["EvaluationInputError", "evaluate", "load_bundle", "load_scored_gold", "load_vocab",
           "shortcut_predictions", "to_markdown"]
