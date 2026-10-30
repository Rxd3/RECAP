"""Inference pipeline: load a trained HMM/CRF, clean raw text, predict labels, rebuild text.

Member 6 — Integration & Reproducibility (Abdullah Jamal)

Input text is lowercased and stripped of punctuation (the models only ever see
unpunctuated lowercase tokens), labelled by the loaded model, and the labels are
turned back into punctuated, cased text. Only load model artifacts you trust:
.joblib/.pkl files are pickles.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from punct_restore.data.labeling import LABEL_SETS_PATH, load_label_sets
from punct_restore.models.crf import CRFModel
from punct_restore.models.hmm import HMMModel

STRIP_CHARS = ',.?!:;-"'
PUNCT_MAP = {
    "COMMA": ",",
    "PERIOD": ".",
    "QUESTION": "?",
    "EXCLAMATION": "!",
    "COLON": ":",
    "SEMICOLON": ";",
    "DASH": " -",
}
_MODES = ("A", "B", "C")


def load_model(path: Path) -> HMMModel | CRFModel:
    """Load a saved model, choosing the family from the file suffix."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return HMMModel.load_model(path)
    if suffix in (".joblib", ".pkl"):
        return CRFModel.load_model(path)
    raise ValueError(f"unsupported model file: {path.name}")


def clean_tokens(tokens: Sequence[str]) -> list[str]:
    """Lowercase, strip edge punctuation and drop tokens that end up empty."""
    cleaned = [token for token in (t.lower().strip(STRIP_CHARS) for t in tokens) if token]
    if not cleaned:
        raise ValueError("empty input")
    return cleaned


def _split_label(label: str) -> tuple[str, str]:
    """Split ``PUNCT|CASE`` into its parts; a plain label is punctuation with LOWER case."""
    punct, _, case = label.partition("|")
    return punct, case or "LOWER"


def _apply_case(word: str, case: str) -> str:
    if case == "FIRST_UPPER":
        return word[:1].upper() + word[1:]
    if case == "ALL_UPPER":
        return word.upper()
    return word


def reconstruct(tokens: Sequence[str], labels: Sequence[str]) -> str:
    """Apply one label per token and join the words with single spaces."""
    if len(tokens) != len(labels):
        raise ValueError(f"{len(labels)} labels for {len(tokens)} tokens")
    words = []
    for token, label in zip(tokens, labels, strict=True):
        punct, case = _split_label(label)
        words.append(_apply_case(token, case) + PUNCT_MAP.get(punct, ""))
    return " ".join(words)


def _predicted_labels(result: Any) -> list[str] | None:
    """Return the labels of a single-prediction result, or None if the shape is wrong."""
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
        labels = result[0].get("predicted_labels")
        if isinstance(labels, list):
            return labels
    return None


class ModelAdapter:
    """Wraps a loaded model behind one text-in, labels-out call with contract checks."""

    def __init__(self, model_path: Path, config_path: Path | str = LABEL_SETS_PATH) -> None:
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"label sets config not found: {config_path.name}")
        self.model = load_model(Path(model_path))
        mode = self.model.label_mode
        if mode not in _MODES:
            raise ValueError(f"unsupported label mode {mode!r} in model")
        self.vocabulary = tuple(load_label_sets(config_path)[mode])

    def predict_tokens(self, tokens: Sequence[str]) -> list[str]:
        """Clean the tokens, run the model and return one validated label per cleaned token."""
        cleaned = clean_tokens(tokens)
        record = {"schema_version": 1, "sample_id": "demo-0", "tokens": cleaned}
        result = self.model.predict([record])
        labels = _predicted_labels(result)
        if labels is None:
            raise ValueError("model must return exactly one prediction with a predicted_labels list")
        if len(labels) != len(cleaned):
            raise ValueError(f"model returned {len(labels)} labels for {len(cleaned)} tokens")
        for label in labels:
            if label not in self.vocabulary:
                mode = self.model.label_mode
                raise ValueError(f"model returned label {label!r} outside the mode {mode} vocabulary")
        return list(labels)


class RestorationPipeline:
    """Raw text in, punctuated and cased text out."""

    def __init__(self, adapter: ModelAdapter) -> None:
        self.adapter = adapter

    def process_text(self, text: str) -> dict[str, Any]:
        tokens = clean_tokens(text.split())
        labels = self.adapter.predict_tokens(tokens)
        return {
            "input_text": text,
            "tokens": tokens,
            "predicted_labels": labels,
            "restored_text": reconstruct(tokens, labels),
        }
