"""Tiny local models exercise real tensors, persistence and official output contracts."""

from __future__ import annotations

import json
from argparse import Namespace
import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from punct_restore.evaluation.inputs import load_bundle, load_vocab, sha256_file
from punct_restore.evaluation.manifest import check_manifest
from punct_restore.models.neural_tagger import BertTagger, encode_words, first_subword_labels
from punct_restore.models.neural_workflow import export_predictions, main, predict, train_epoch
from punct_restore.models.seq2seq import T5Seq2Seq


@pytest.fixture
def tokenizer(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                               "hello", "world", "play", "##ing", ".", ","]), encoding="utf-8")
    return transformers.BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)


@pytest.fixture
def tagger(tokenizer):
    instance = BertTagger("A")
    instance.tokenizer = tokenizer
    instance.backend = transformers.BertForTokenClassification(transformers.BertConfig(
        vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=37, num_labels=4, id2label=dict(enumerate(instance.labels)),
        label2id={label: i for i, label in enumerate(instance.labels)}))
    return instance.configure_device("cpu")


def test_truncation_masks_partial_word_and_counts_all_cut_words(tokenizer):
    tokens = ["hello"] + ["playing"] * 255
    encoded, complete, cut = encode_words(tokenizer, tokens)
    assert len(encoded["input_ids"]) == 512
    assert cut == 1 and 255 not in complete and 254 in complete
    labels = first_subword_labels(encoded.word_ids(), [1] * len(tokens), complete)
    assert labels[-2:] == [-100, -100]
    assert sum(label != -100 for label in labels) == 255
    _, complete, cut = encode_words(tokenizer, ["playing"] * 384)
    assert cut == 129 and len(complete) == 255


def test_tiny_bert_training_predict_save_load_and_fallback(tagger, tmp_path):
    record = {"sample_id": "one", "tokens": ["hello", "playing"], "punct_labels": ["O", "PERIOD"]}
    batch = tagger.training_batch([record])
    assert batch["labels"].tolist() == [[-100, 0, 2, -100, -100]]
    optimizer = torch.optim.AdamW(tagger.backend.parameters(), lr=1e-3)
    loss = train_epoch(tagger, [batch], optimizer, torch.amp.GradScaler("cuda", enabled=False), 4)
    assert loss > 0
    before = tagger.predict([record])
    assert len(before[0]["predicted_labels"]) == 2
    assert set(before[0]) == {"sample_id", "model_name", "label_mode", "predicted_labels"}
    path = tmp_path / "model"
    tagger.save_model(path)
    assert BertTagger.load_model(path).predict([record]) == before
    with pytest.raises(FileExistsError):
        tagger.save_model(path)
    long = tagger.predict([{**record, "tokens": ["playing"] * 384}])[0]
    assert len(long["predicted_labels"]) == 384
    assert long["predicted_labels"][255:] == ["O"] * 129
    assert tagger.last_diagnostics["cutoff_words"] == 129
    (path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        BertTagger.load_model(path)


def test_tiny_t5_forward_generation_cap_and_alignment(tokenizer, monkeypatch, tmp_path):
    instance = T5Seq2Seq("C")
    instance.tokenizer = tokenizer
    instance.backend = transformers.T5ForConditionalGeneration(transformers.T5Config(
        vocab_size=len(tokenizer), d_model=32, d_ff=37, num_layers=1, num_decoder_layers=1,
        num_heads=2, decoder_start_token_id=0, eos_token_id=3, pad_token_id=0))
    instance.configure_device("cpu")
    record = {"sample_id": "one", "tokens": ["hello", "world"],
              "punct_labels": ["COMMA", "PERIOD"], "case_labels": ["FIRST_UPPER", "ALL_UPPER"]}
    batch = instance.training_batch([record])
    assert torch.isfinite(instance.backend(**batch).loss)
    assert instance.backend.generate(input_ids=batch["input_ids"],
                                     attention_mask=batch["attention_mask"], max_new_tokens=2).shape[0] == 1
    # Check the real tiny model's persistence before replacing generation for exact alignment assertions.
    instance.save_model(tmp_path / "t5")
    loaded = T5Seq2Seq.load_model(tmp_path / "t5")
    assert loaded.labels == instance.labels
    seen = []
    def generate(**kwargs):
        seen.append((kwargs["num_beams"], kwargs["max_new_tokens"]))
        return torch.tensor([[5, 6]])
    monkeypatch.setattr(instance.backend, "generate", generate)
    monkeypatch.setattr(instance.tokenizer, "batch_decode", lambda *a, **k: ["Hello, WORLD."])
    result = instance.predict([record])
    assert seen == [(4, 20)]
    assert result[0]["predicted_labels"] == ["COMMA|FIRST_UPPER", "PERIOD|ALL_UPPER"]
    assert instance.last_diagnostics["unaligned_share"] == 0


@pytest.mark.parametrize("mode", ["A", "B", "C"])
def test_output_manifest_passes_official_checks_and_refuses_overwrite(tagger, tmp_path, mode):
    tagger.label_mode = mode
    tagger.labels = list(load_vocab().labels(mode))
    # A deterministic fake decode isolates export/manifest validation from model accuracy.
    labels = ["O|LOWER", "PERIOD|FIRST_UPPER"] if mode == "C" else ["O", "PERIOD"]
    record = {"sample_id": "one", "label_mode": mode, "tokens": ["hello", "world"],
              "punct_labels": ["O", "PERIOD"], "case_labels": ["LOWER", "FIRST_UPPER"]}
    tagger.predict = lambda records, **kw: [{"sample_id": "one", "model_name": "bert_tagger",
                                            "label_mode": mode, "predicted_labels": labels}]
    source, ids = tmp_path / "validation.jsonl", tmp_path / "ids"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    ids.write_text("one\n", encoding="utf-8")
    masks = tmp_path / "masks.jsonl"
    masks.write_text(json.dumps({"sample_id": "one", "excluded_positions": [], "reason": []}) + "\n", encoding="utf-8")
    output, manifest = tmp_path / "pred.jsonl", tmp_path / "pred.manifest.json"
    export_predictions(tagger, [record], source, ids, output, manifest, {"batch_size": 1})
    bundle = load_bundle(source, ids, output, masks if mode == "C" else None, load_vocab())
    check_manifest(manifest, bundle, load_vocab())
    original = output.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        export_predictions(tagger, [record], source, ids, output, manifest, {"batch_size": 1})
    assert output.read_bytes() == original
    altered = json.loads(manifest.read_text(encoding="utf-8"))
    altered["predictions_sha256"] = "0" * 64
    manifest.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError):
        check_manifest(manifest, bundle, load_vocab())


def test_tiny_full_selection_and_once_only_test_inference(tagger, tmp_path, monkeypatch):
    source = tmp_path / "processed/mode_A"
    source.mkdir(parents=True)
    splits = tmp_path / "splits"
    (splits / "default_train").mkdir(parents=True)
    (splits / "primary_eval").mkdir()
    record = {"sample_id": "one", "label_mode": "A", "tokens": ["hello", "world"],
              "punct_labels": ["O", "PERIOD"], "case_labels": ["LOWER", "FIRST_UPPER"],
              "metadata": {"group_id": "group1", "policy_version": "m1-v2"}, "split": "train"}
    for split in ("train", "validation", "test"):
        (source / f"{split}.jsonl").write_text(json.dumps({**record, "split": split}) + "\n", encoding="utf-8")
    (splits / "default_train/mode_A.ids").write_text("one\n", encoding="utf-8")
    for split in ("validation", "test"):
        (splits / f"primary_eval/{split}_mode_A.ids").write_text("one\n", encoding="utf-8")
    monkeypatch.setattr(BertTagger, "from_pretrained", classmethod(lambda cls, *a, **kw: copy.deepcopy(tagger)))
    run = tmp_path / "run"
    assert main(BertTagger, ["train", "--mode", "A", "--device", "cpu", "--epochs", "1",
        "--input", str(source / "train.jsonl"), "--ids", str(splits / "default_train/mode_A.ids"),
        "--validation-input", str(source / "validation.jsonl"),
        "--validation-ids", str(splits / "primary_eval/validation_mode_A.ids"),
        "--output", str(run), "--manifest-out", str(run / "training.manifest.json"),
        "--selection-out", str(run / "selection.json"), "--run-id", "local-fixture",
        "--notes-out", str(tmp_path / "notes.md")]) == 0
    selected = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    assert selected["frozen"] is True and len(selected["candidates"]) == 2
    assert "local-fixture" in (tmp_path / "notes.md").read_text(encoding="utf-8")
    runtime = json.loads((run / "training.manifest.json").read_text(encoding="utf-8"))["validation_runtime"]
    assert len(runtime["ms_per_utterance_passes"]) == 3
    args = Namespace(selection=run / "selection.json", model=run / selected["selected"]["artifact"],
                     input=source / "test.jsonl", ids=splits / "primary_eval/test_mode_A.ids",
                     output=run / "test.jsonl", manifest_out=run / "test.manifest.json",
                     device="cpu", batch_size=1)
    predict(args, BertTagger)
    assert json.loads((run / "test.prediction-receipt.json").read_text())["status"] == "complete"
    args.output, args.manifest_out = run / "other.jsonl", run / "other.manifest.json"
    with pytest.raises(ValueError, match="already attempted"):
        predict(args, BertTagger)
