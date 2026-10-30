"""Contract validation and artifact persistence shared by the optional neural models."""

from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
import json
from pathlib import Path

from punct_restore.data.labeling import load_label_sets
from punct_restore.evaluation.inputs import sha256_file


@lru_cache(maxsize=1)
def label_config():
    return load_label_sets()


def labels_for_record(record, mode):
    tokens = record["tokens"]
    punct = record.get("punct_labels", [])
    case = record.get("case_labels", [])
    if len(punct) != len(tokens) or (mode == "C" and len(case) != len(tokens)):
        raise ValueError("training labels must match the token count")
    labels = [f"{p}|{c}" for p, c in zip(punct, case)] if mode == "C" else list(punct)
    vocabulary = label_config()[mode]
    if any(label not in vocabulary for label in labels):
        raise ValueError("training label outside the mode vocabulary")
    return labels


def observations(records):
    result, seen = [], set()
    for record in records:
        sample_id, tokens = record.get("sample_id"), record.get("tokens")
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen:
            raise ValueError("sample_id must be nonempty and unique")
        if not isinstance(tokens, list) or not tokens or any(
                not isinstance(t, str) or not t or t != t.lower() or any(c.isspace() for c in t)
                for t in tokens):
            raise ValueError("tokens must be nonempty lowercase words without whitespace")
        result.append({"sample_id": sample_id, "tokens": list(tokens)})
        seen.add(sample_id)
    if not result:
        raise ValueError("no observations")
    return result


def prediction(record, name, mode, labels):
    if len(labels) != len(record["tokens"]) or any(l not in label_config()[mode] for l in labels):
        raise ValueError("invalid prediction length or label")
    return {"sample_id": record["sample_id"], "model_name": name,
            "label_mode": mode, "predicted_labels": labels}


def artifact_hashes(path):
    return {p.relative_to(path).as_posix(): sha256_file(p)
            for p in sorted(Path(path).rglob("*")) if p.is_file() and p.name != "neural_metadata.json"}


class NeuralArtifact:
    def __init__(self, label_mode):
        if label_mode not in ("A", "B", "C"):
            raise ValueError("label_mode must be A, B or C")
        self.label_mode = label_mode
        self.labels = load_label_sets()[label_mode]
        self.default_label = "O|LOWER" if label_mode == "C" else "O"
        self.num_beams = 4
        self.device = "cpu"
        self.fp16 = False
        self.precision = "float32"
        self.last_diagnostics = {}
        self.training_diagnostics = {"words": 0, "cutoff_words": 0}
        self.pretrained, self.revision = None, None
        self.backend, self.tokenizer = None, None

    def require_fitted(self):
        if self.backend is None or self.tokenizer is None:
            raise RuntimeError("model is not initialized; load an artifact or pretrained model")

    def configure_device(self, device="cuda", *, fp16=True):
        import torch

        self.require_fitted()
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA GPU unavailable; install the CUDA build of the neural extra")
        self.device = device
        self.fp16 = fp16 and device == "cuda" and self.model_name == "bert_tagger"
        self.precision = "fp16" if self.fp16 else "float32"
        if device == "cuda" and self.model_name == "t5_seq2seq" and torch.cuda.is_bf16_supported():
            self.precision = "bf16"
        self.backend.to(device)
        return self

    def autocast(self):
        import torch

        if self.precision == "float32":
            return nullcontext()
        return torch.autocast("cuda", dtype=torch.float16 if self.fp16 else torch.bfloat16)

    def to_device(self, batch):
        return {key: value.to(self.device) for key, value in batch.items()}

    def save_model(self, path, *, metadata=None):
        self.require_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        self.backend.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        document = {"schema_version": 1, "model_name": self.model_name,
                    "label_mode": self.label_mode, "ordered_labels": self.labels,
                    "num_beams": self.num_beams, "pretrained": self.pretrained,
                    "revision": self.revision, "metadata": metadata or {},
                    "files": artifact_hashes(path)}
        with (path / "neural_metadata.json").open("x", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)

    @classmethod
    def load_model(cls, path):
        from transformers import AutoTokenizer

        path = Path(path)
        document = json.loads((path / "neural_metadata.json").read_text(encoding="utf-8"))
        instance = cls(document["label_mode"])
        if (document["schema_version"] != 1 or document["model_name"] != cls.model_name
                or document["ordered_labels"] != instance.labels):
            raise ValueError("incompatible neural artifact")
        if document["files"] != artifact_hashes(path):
            raise ValueError("neural artifact checksum mismatch")
        instance.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        instance.backend = cls._load_backend(path)
        if cls.model_name == "bert_tagger" and [instance.backend.config.id2label[i]
                for i in range(len(instance.labels))] != instance.labels:
            raise ValueError("classifier vocabulary differs from artifact")
        instance.num_beams = document["num_beams"]
        if instance.num_beams not in (1, 4):
            raise ValueError("unsupported saved beam count")
        instance.pretrained, instance.revision = document["pretrained"], document["revision"]
        instance.metadata = document["metadata"]
        return instance
