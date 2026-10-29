"""Shared evaluation for RECAP models, owned by Member 5 (docs/EVALUATION_PROTOCOL.md).

load_bundle() checks and aligns gold, ID list, predictions and masks; evaluate() scores every
block of the mode (optionally with grouped bootstrap CIs); compare() pairs two models;
error_examples() counts error pairs and draws examples; check_manifest() verifies a model's
prediction manifest; shortcut_predictions() builds the shortcut reference baseline.
"""

from .baseline import shortcut_predictions
from .comparison import compare, comparison_markdown
from .error_analysis import error_examples, errors_markdown
from .inputs import EvaluationInputError, load_bundle, load_scored_gold, load_vocab
from .manifest import check_manifest
from .report import evaluate, to_markdown
from .uncertainty import BootstrapSpec

__all__ = ["BootstrapSpec", "EvaluationInputError", "check_manifest", "compare",
           "comparison_markdown", "error_examples", "errors_markdown", "evaluate", "load_bundle",
           "load_scored_gold", "load_vocab", "shortcut_predictions", "to_markdown"]
