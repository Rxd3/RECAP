"""Inference pipeline: load a trained model, clean raw text, predict labels, rebuild text.

Loadable models: HMM and hidden-event LM (.json), CRF and gap classifier (.joblib/.pkl).

Member 6 — Integration & Reproducibility (Abdullah Jamal)

Input text is lowercased and stripped of punctuation (the models only ever see
unpunctuated lowercase tokens), labelled by the loaded model, and the labels are
turned back into punctuated, cased text. Only load model artifacts you trust:
.joblib/.pkl files are pickles.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib

from punct_restore.data.labeling import LABEL_SETS_PATH, load_label_sets
from punct_restore.models.crf import CRFModel
from punct_restore.models.gap_classifier import MODEL_FAMILY as GAP_CLASSIFIER_FAMILY
from punct_restore.models.gap_classifier import GapClassifier
from punct_restore.models.hidden_event_lm import MODEL_FAMILY as HIDDEN_EVENT_LM_FAMILY
from punct_restore.models.hidden_event_lm import HiddenEventLM
from punct_restore.models.hmm import HMMModel
from punct_restore.prediction_io import require_file

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


# crf.py and hmm.py define no family constant: the CRF bundle says "crf", an HMM artifact has no
# model_family and is recognised by its model_name.
_HMM_FAMILY = "hmm"
_CRF_FAMILY = "crf"
_JSON_FAMILIES = {_HMM_FAMILY: HMMModel, HIDDEN_EVENT_LM_FAMILY: HiddenEventLM}
_JOBLIB_FAMILIES = {_CRF_FAMILY: CRFModel, GAP_CLASSIFIER_FAMILY: GapClassifier}
_FAMILIES_BY_SUFFIX = {".json": _JSON_FAMILIES, ".joblib": _JOBLIB_FAMILIES, ".pkl": _JOBLIB_FAMILIES}


def _unsupported(path: Path, why: str) -> ValueError:
    return ValueError(f"{path.name}: unsupported model file ({why})")


def _read_payload(path: Path) -> Any:
    """Parse the artifact only to read its family marker; the class loader parses it again."""
    try:
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        return joblib.load(path)  # pickle: load trusted files only (see the module docstring)
    except OSError as exc:  # the file exists but cannot be read: an OS problem, not a bad artifact
        raise OSError(f"{path.name}: cannot read model file ({exc.strerror or 'read error'})") from exc
    except Exception as exc:  # a corrupt or foreign file; the reason is not shown to the user
        raise _unsupported(path, "unreadable") from exc


def _family_of(payload: Any) -> Any:
    """The family marker of a parsed artifact, or None when it is not a dict with one."""
    if not isinstance(payload, dict):
        return None
    family = payload.get("model_family")
    if family is None and payload.get("model_name") == _HMM_FAMILY:
        return _HMM_FAMILY
    return family


def load_model(path: Path) -> HMMModel | CRFModel | HiddenEventLM | GapClassifier:
    """Load a saved model: the suffix narrows the choice, the family marker inside the file decides."""
    path = Path(path)
    require_file(path, "model file")
    families = _FAMILIES_BY_SUFFIX.get(path.suffix.lower())
    if families is None:
        raise _unsupported(path, "expected .json, .joblib or .pkl")
    family = _family_of(_read_payload(path))
    loader = families.get(family) if isinstance(family, str) else None
    if loader is None:
        raise _unsupported(path, "unknown model family")
    # ponytail: the file is read twice (peek, then the class loader), so big artifacts are parsed
    # twice; upgrade path is class loaders that accept an already parsed payload.
    return loader.load_model(path)


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
        require_file(config_path, "label sets config")
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
