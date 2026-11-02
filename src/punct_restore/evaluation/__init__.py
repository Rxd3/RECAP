"""Shared evaluation for RECAP models, owned by Member 5 (docs/EVALUATION_PROTOCOL.md).

load_bundle() checks and aligns gold, ID list, predictions and masks; evaluate() scores every
block of the mode (optionally with grouped bootstrap CIs); compare() pairs two models;
error_examples() counts error pairs and draws examples; check_manifest() verifies a model's
prediction manifest; shortcut_predictions() builds the shortcut reference baseline.

The names below load their module on first use (`punct_restore.lazy_exports`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..lazy_exports import lazy_exports

if TYPE_CHECKING:  # what the lazy names are, for type checkers and editors
    from .baseline import shortcut_predictions
    from .comparison import compare, comparison_markdown
    from .error_analysis import error_examples, errors_markdown
    from .inputs import EvaluationInputError, load_bundle, load_scored_gold, load_vocab
    from .manifest import check_manifest
    from .report import evaluate, to_markdown
    from .uncertainty import BootstrapSpec

_EXPORTS = {
    "BootstrapSpec": "uncertainty", "EvaluationInputError": "inputs", "check_manifest": "manifest",
    "compare": "comparison", "comparison_markdown": "comparison", "error_examples": "error_analysis",
    "errors_markdown": "error_analysis", "evaluate": "report", "load_bundle": "inputs",
    "load_scored_gold": "inputs", "load_vocab": "inputs", "shortcut_predictions": "baseline",
    "to_markdown": "report"}
__all__ = ["BootstrapSpec", "EvaluationInputError", "check_manifest", "compare",
           "comparison_markdown", "error_examples", "errors_markdown", "evaluate", "load_bundle",
           "load_scored_gold", "load_vocab", "shortcut_predictions", "to_markdown"]  # a literal, which linters read

if not TYPE_CHECKING:  # the imports above are what type checkers see; this lookup would make every name valid to them
    __getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
