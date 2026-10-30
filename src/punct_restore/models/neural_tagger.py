"""Optional BERT word tagger; importing this module needs no neural packages."""

from __future__ import annotations

from collections import Counter

from .neural_utils import NeuralArtifact, labels_for_record, observations, prediction


def first_subword_labels(word_ids, word_labels, complete_words=None):
    """Supervise the first piece only; specials, repeats and cut words are ignored."""
    seen = set()
    result = []
    for word in word_ids:
        first = word is not None and word not in seen
        usable = first and (complete_words is None or word in complete_words)
        result.append(word_labels[word] if usable else -100)
        seen.add(word)
    return result


def encode_words(tokenizer, tokens, *, max_length=512):
    """Truncate at 512 pieces, counting a partially retained word as cut off too.

    Comparing full and truncated piece counts prevents a partial final word from
    being trained or decoded. Every cut word receives the mode's default at decode.
    """
    if not 3 <= max_length <= 512:
        raise ValueError("max_length must be between 3 and 512")
    full = tokenizer(tokens, is_split_into_words=True, truncation=False,
                     verbose=False)
    encoded = tokenizer(tokens, is_split_into_words=True, truncation=True,
                        max_length=max_length)
    full_counts = Counter(w for w in full.word_ids() if w is not None)
    kept_counts = Counter(w for w in encoded.word_ids() if w is not None)
    complete = {w for w, count in full_counts.items() if kept_counts[w] == count}
    return encoded, complete, len(tokens) - len(complete)


class BertTagger(NeuralArtifact):
    """One BERT token classifier per label mode, using complete first pieces."""

    model_name = "bert_tagger"
    default_pretrained = "bert-base-uncased"

    @classmethod
    def from_pretrained(cls, label_mode, pretrained=None, *, revision="main"):
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        instance = cls(label_mode)
        name = pretrained or cls.default_pretrained
        instance.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True)
        instance.backend = AutoModelForTokenClassification.from_pretrained(
            name, revision=revision, num_labels=len(instance.labels),
            id2label=dict(enumerate(instance.labels)),
            label2id={label: i for i, label in enumerate(instance.labels)})
        instance.pretrained = name
        instance.revision = instance.backend.config._commit_hash
        return instance

    @classmethod
    def _load_backend(cls, path):
        from transformers import AutoModelForTokenClassification

        return AutoModelForTokenClassification.from_pretrained(path, local_files_only=True)

    def training_batch(self, records):
        from transformers import DataCollatorForTokenClassification

        self.require_fitted()
        features = []
        for record in records:
            encoded, complete, cut = encode_words(self.tokenizer, record["tokens"])
            self.training_diagnostics["words"] += len(record["tokens"])
            self.training_diagnostics["cutoff_words"] += cut
            targets = [self.labels.index(label) for label in labels_for_record(record, self.label_mode)]
            feature = dict(encoded)
            feature["labels"] = first_subword_labels(encoded.word_ids(), targets, complete)
            features.append(feature)
        return DataCollatorForTokenClassification(self.tokenizer)(features)

    def predict(self, records, *, batch_size=8):
        import torch

        self.require_fitted()
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        records = observations(records)
        self.backend.eval()
        result, cut, words = [], 0, 0
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            encoded = [encode_words(self.tokenizer, record["tokens"]) for record in chunk]
            batch = self.tokenizer.pad([dict(e[0]) for e in encoded], return_tensors="pt")
            with torch.inference_mode(), self.autocast():
                guessed = self.backend(**self.to_device(batch)).logits.argmax(-1).cpu().tolist()
            for record, (encoding, complete, cutoff), row in zip(chunk, encoded, guessed, strict=True):
                labels = [self.default_label] * len(record["tokens"])
                seen = set()
                for word, label_id in zip(encoding.word_ids(), row):
                    if word in complete and word not in seen:
                        labels[word] = self.labels[label_id]
                    seen.add(word)
                result.append(prediction(record, self.model_name, self.label_mode, labels))
                cut += cutoff
                words += len(labels)
        self.last_diagnostics = {"words": words, "cutoff_words": cut,
                                 "cutoff_share": cut / words if words else 0.0}
        return result
