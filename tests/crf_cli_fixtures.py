"""Synthetic CRF training data for the train-crf command tests (explicitly synthetic, not SponSpeech data)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def create_synthetic_train_jsonl(
    path: Path,
    mode: str = "A",
    num_samples: int = 5,
    split: str = "train",
) -> list[str]:
    """Helper to create a synthetic JSONL file and return its sample IDs."""
    ids: list[str] = []
    mode_a_punct = ["O", "COMMA", "PERIOD", "QUESTION"]
    mode_b_punct = ["O", "COMMA", "PERIOD", "QUESTION", "EXCLAMATION", "COLON", "SEMICOLON", "DASH"]
    case_labels_pool = ["LOWER", "FIRST_UPPER", "ALL_UPPER"]

    tokens_pool = [
        ["hello", "world"],
        ["how", "are", "you"],
        ["this", "is", "a", "test"],
        ["nlp", "is", "great"],
        ["good", "morning", "everyone"],
        ["we", "love", "python"],
        ["crf", "sequence", "tagging"],
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for idx in range(num_samples):
            s_id = f"sample_{idx:04d}"
            tokens = tokens_pool[idx % len(tokens_pool)]
            n = len(tokens)

            if mode == "A":
                p_labels = [mode_a_punct[i % len(mode_a_punct)] for i in range(n)]
                p_labels[-1] = "PERIOD"
                rec: dict[str, Any] = {
                    "sample_id": s_id,
                    "split": split,
                    "label_mode": "A",
                    "tokens": tokens,
                    "punct_labels": p_labels,
                }
            elif mode == "B":
                p_labels = [mode_b_punct[i % len(mode_b_punct)] for i in range(n)]
                p_labels[-1] = "EXCLAMATION"
                rec = {
                    "sample_id": s_id,
                    "split": split,
                    "label_mode": "B",
                    "tokens": tokens,
                    "punct_labels": p_labels,
                }
            elif mode == "C":
                p_labels = [mode_a_punct[i % len(mode_a_punct)] for i in range(n)]
                p_labels[-1] = "PERIOD"
                c_labels = [case_labels_pool[i % len(case_labels_pool)] for i in range(n)]
                rec = {
                    "sample_id": s_id,
                    "split": split,
                    "label_mode": "C",
                    "tokens": tokens,
                    "punct_labels": p_labels,
                    "case_labels": c_labels,
                }
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            f.write(json.dumps(rec) + "\n")
            ids.append(s_id)

    return ids


def write_id_file(path: Path, ids: list[str]) -> None:
    """Helper to write an ID file with one ID per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s_id in ids:
            f.write(f"{s_id}\n")
