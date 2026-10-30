"""Optional T5 rewriting with the fixed whitespace/matching-block word alignment."""

from __future__ import annotations

from difflib import SequenceMatcher

from .neural_utils import NeuralArtifact, label_config, labels_for_record, observations, prediction

MARKS = {",": "COMMA", ".": "PERIOD", "?": "QUESTION", "!": "EXCLAMATION",
         ":": "COLON", ";": "SEMICOLON", "-": "DASH"}


def output_words(text):
    """Return (surface, last boundary mark) pairs; a standalone dash marks its left word.

    Attached hyphens remain lexical. With several trailing marks, the rightmost
    mark is the last one in the generated text, regardless of stripping order.
    """
    words = []
    for piece in text.split():
        if piece == "-":
            if words:
                words[-1] = (words[-1][0], "DASH")
            continue
        mark = MARKS.get(piece[-1], "O") if piece[-1] != "-" else "O"
        word = piece.rstrip(",.?!:;")
        if word:
            words.append((word, mark))
        elif words:
            words[-1] = (words[-1][0], mark)
    return words


def word_case(word):
    letters = "".join(char for char in word if char.isalpha())
    if len(word) > 1 and letters.isupper():
        return "ALL_UPPER"
    if letters and letters[0].isupper():
        return "FIRST_UPPER"
    return "LOWER"


def align_generated(tokens, text, label_mode):
    """Use matching blocks only; edits receive O (or O|LOWER), without fuzzy rescue."""
    config = label_config()
    if label_mode not in ("A", "B", "C"):
        raise ValueError("label_mode must be A, B or C")
    generated = output_words(text)
    labels = ["O|LOWER" if label_mode == "C" else "O"] * len(tokens)
    aligned = 0
    matcher = SequenceMatcher(None, list(tokens), [w.lower() for w, _ in generated], autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            surface, mark = generated[block.b + offset]
            if label_mode in ("A", "C"):
                mark = config["projection"]["mapping"][mark]
            labels[block.a + offset] = f"{mark}|{word_case(surface)}" if label_mode == "C" else mark
            aligned += 1
    return labels, len(tokens) - aligned


def target_text(record, label_mode):
    # Import the existing reconstruction only when training seq2seq.
    from punct_restore.pipeline import reconstruct

    return reconstruct(record["tokens"], labels_for_record(record, label_mode))


class T5Seq2Seq(NeuralArtifact):
    """One rewriting model per mode; generation limits are per input utterance."""

    model_name = "t5_seq2seq"
    default_pretrained = "google/flan-t5-base"

    @classmethod
    def from_pretrained(cls, label_mode, pretrained=None, *, revision="main"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        instance = cls(label_mode)
        name = pretrained or cls.default_pretrained
        instance.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        instance.backend = AutoModelForSeq2SeqLM.from_pretrained(name, revision=revision)
        instance.pretrained = name
        instance.revision = instance.backend.config._commit_hash
        return instance

    @classmethod
    def _load_backend(cls, path):
        from transformers import AutoModelForSeq2SeqLM

        return AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True)

    def training_batch(self, records):
        from transformers import DataCollatorForSeq2Seq

        self.require_fitted()
        # No source/target truncation: T5 uses relative positions. Dynamic padding.
        inputs = self.tokenizer([" ".join(r["tokens"]) for r in records], truncation=False,
                                return_token_type_ids=False)
        targets = self.tokenizer(text_target=[target_text(r, self.label_mode) for r in records],
                                 truncation=False)
        features = [{**{key: value[i] for key, value in inputs.items()}, "labels": labels}
                    for i, labels in enumerate(targets["input_ids"])]
        return DataCollatorForSeq2Seq(self.tokenizer, model=self.backend)(features)

    def predict(self, records, *, batch_size=8):
        import torch

        self.require_fitted()
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        records = observations(records)
        self.backend.eval()
        result, unaligned, words = [], 0, 0
        # Group equal word counts to apply the exact per-record generation cap.
        groups = {}
        for index, record in enumerate(records):
            groups.setdefault(len(record["tokens"]), []).append((index, record))
        ordered = {}
        for count, group in groups.items():
            for start in range(0, len(group), batch_size):
                chunk = group[start:start + batch_size]
                inputs = self.tokenizer([" ".join(r["tokens"]) for _, r in chunk],
                                        padding=True, truncation=False, return_tensors="pt",
                                        return_token_type_ids=False)
                with torch.inference_mode(), self.autocast():
                    output = self.backend.generate(**self.to_device(inputs), num_beams=self.num_beams,
                                                   do_sample=False, use_cache=True,
                                                   max_new_tokens=2 * count + 16)
                for (index, record), text in zip(chunk, self.tokenizer.batch_decode(
                        output, skip_special_tokens=True), strict=True):
                    labels, missing = align_generated(record["tokens"], text, self.label_mode)
                    ordered[index] = prediction(record, self.model_name, self.label_mode, labels)
                    unaligned += missing
                    words += count
        result = [ordered[i] for i in range(len(records))]
        self.last_diagnostics = {"words": words, "unaligned_words": unaligned,
                                 "unaligned_share": unaligned / words if words else 0.0}
        return result
