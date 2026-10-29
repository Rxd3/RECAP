"""Case builders and helpers shared by the score-extras, compare and errors command line tests.

The case has 8 gold rows over 4 groups (g1 to g4); model_a (preds.jsonl) and model_b (preds_b.jsonl) make
different mistakes. Not a test module: pytest does not collect it.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from eval_fixtures import Case, cli_argv, gold_row, joint, mask_row, pred_row, write_case, write_jsonl

from punct_restore.evaluation import load_bundle
from punct_restore.evaluation.inputs import Bundle, Vocab, sha256_file

IDS = tuple(f"s{number}" for number in range(1, 9))
TOKENS = ("what", "is", "it")
CASING = ("FIRST_UPPER", "LOWER", "LOWER")
MODEL_A_ERRORS = {"s1": {2: "QUESTION"}, "s4": {2: "PERIOD"}, "s5": {1: "O"}}
MODEL_B_ERRORS = {"s2": {2: "O"}, "s3": {1: "O"}, "s7": {2: "COMMA"}}
MANIFEST_NAME = "predictions.manifest.json"


# ---------------------------------------------------------------- case builders
def gold_punct(index: int) -> tuple[str, ...]:
    return ("O", "COMMA", "PERIOD") if index % 2 == 0 else ("O", "O", "QUESTION")


def gold_for(mode: str, no_group: Collection[str]) -> list[dict[str, Any]]:
    """Two rows per group; the IDs in `no_group` get no metadata."""
    return [gold_row(sid, mode, TOKENS, gold_punct(index), CASING,
                     None if sid in no_group else f"g{index // 2 + 1}")
            for index, sid in enumerate(IDS)]


def predictions_for(mode: str, model: str, errors: Mapping[str, Mapping[int, str]]) -> list[dict[str, Any]]:
    """Gold labels with the given {sample_id: {position: wrong punctuation}} changed (joint labels in mode C)."""
    rows = []
    for index, sid in enumerate(IDS):
        punct = [errors.get(sid, {}).get(position, label) for position, label in enumerate(gold_punct(index))]
        rows.append(pred_row(sid, mode, joint(punct, CASING) if mode == "C" else punct, model))
    return rows


def build_case(folder: Path, mode: str = "A", no_group: Collection[str] = ()) -> Case:
    """preds.jsonl is model_a; preds_b.jsonl (model_b) sits beside it; mode C also gets masks.jsonl."""
    folder.mkdir()
    masks = [mask_row(sid, []) for sid in IDS] if mode == "C" else None
    case = write_case(folder, gold_for(mode, no_group), IDS,
                      predictions_for(mode, "model_a", MODEL_A_ERRORS), masks)
    write_jsonl(folder / "preds_b.jsonl", predictions_for(mode, "model_b", MODEL_B_ERRORS))
    return case


# ---------------------------------------------------------------- helpers
def predictions_b(case: Case) -> Path:
    return case.predictions.with_name("preds_b.jsonl")


def bundle_of(case: Case, vocab: Vocab, predictions: Path | None = None) -> Bundle:
    return load_bundle(case.gold, case.ids, predictions or case.predictions, case.masks, vocab)


def compare_argv(case: Case, out: Path, *extra: str) -> list[str]:
    return cli_argv("compare", case, out, "--predictions-a", str(case.predictions),
                    "--predictions-b", str(predictions_b(case)), *extra, predictions=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_one_line(stdout: str) -> None:
    assert stdout.endswith("\n") and stdout.count("\n") == 1, repr(stdout)


def assert_nothing_written(out: Path) -> None:
    for path in (out, out.with_suffix(".md")):
        assert not path.exists(), f"{path.name} must not exist"


def assert_usage_error(cli: ModuleType, capsys: pytest.CaptureFixture[str], argv: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(argv)
    assert raised.value.code == 2
    assert "usage:" in capsys.readouterr().err


def valid_manifest(case: Case, vocab: Vocab) -> dict[str, Any]:
    return {"model_name": "model_a", "label_mode": "A", "ordered_labels": list(vocab.A),
            "predictions_sha256": sha256_file(case.predictions), "input": {"sha256": sha256_file(case.gold)}}


def write_manifest(case: Case, text: str) -> Path:
    path = case.predictions.parent / MANIFEST_NAME
    path.write_text(text, encoding="utf-8")
    return path
