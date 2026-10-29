"""
Member 6 — Integration & Reproducibility (Abdullah Jamal)
"""

from typing import Dict, List, Any
from pathlib import Path
import yaml


class ModelAdapter:
    """Base interface wrapper for loaded models (HMM or CRF)."""
    def __init__(self, model_artifact_path: Path, config_path: Path):
        self.model_path = Path(model_artifact_path)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.model = self._load_model()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_model(self) -> Any:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        return None

    def predict_tokens(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        clean_tokens = [t.lower() for t in tokens]
        if self.model is None:
            return ["O"] * len(clean_tokens)
        return self.model.predict(clean_tokens)


class RestorationPipeline:
    """Unified pipeline connecting model adapters and text reconstruction."""
    def __init__(self, adapter: ModelAdapter):
        self.adapter = adapter

    def process_text(self, text: str) -> Dict[str, Any]:
        raw_tokens = text.strip().split()
        predicted_labels = self.adapter.predict_tokens(raw_tokens)
        reconstructed_text = self._reconstruct_text(raw_tokens, predicted_labels)
        
        return {
            "input_text": text,
            "raw_tokens": raw_tokens,
            "predicted_labels": predicted_labels,
            "restored_text": reconstructed_text,
        }

    def _reconstruct_text(self, tokens: List[str], labels: List[str]) -> str:
        punct_map = {
            "COMMA": ",",
            "PERIOD": ".",
            "QUESTION": "?",
            "EXCLAMATION": "!",
            "COLON": ":",
            "SEMICOLON": ";",
            "DASH": "-",
        }
        output_words = []
        for token, label in zip(tokens, labels):
            punct_tag = label
            casing_tag = "LOWER"
            if "|" in label:
                punct_tag, casing_tag = label.split("|", 1)

            word = token.lower()
            if casing_tag == "FIRST_UPPER":
                word = word.capitalize()
            elif casing_tag == "ALL_UPPER":
                word = word.upper()

            punct = punct_map.get(punct_tag, "")
            output_words.append(word + punct)

        return " ".join(output_words)