"""Tests for text cleaning, label-to-text reconstruction, the pipeline and the demo CLI."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from punct_restore import pipeline
from punct_restore.cli import run_demo
from punct_restore.data import labeling
from punct_restore.data.labeling import load_label_sets
from punct_restore.models.crf import CRFModel
from punct_restore.models.hmm import HMMModel
from punct_restore.pipeline import (
    ModelAdapter,
    RestorationPipeline,
    clean_tokens,
    reconstruct,
)

ROOT = Path(__file__).resolve().parents[1]


class StubAdapter:
    """Adapter double: label every token O."""

    def predict_tokens(self, tokens: list[str]) -> list[str]:
        return ["O"] * len(tokens)


@pytest.fixture(scope="module")
def hmm_json(tmp_path_factory: pytest.TempPathFactory) -> Path:
    records: list[dict[str, Any]] = [
        {
            "schema_version": 1,
            "sample_id": f"fixture-{index}",
            "split": "train",
            "label_mode": "A",
            "source": "fixture",
            "tokens": tokens,
            "punct_labels": ["O"] * (len(tokens) - 1) + ["PERIOD"],
            "case_labels": ["LOWER"] * len(tokens),
        }
        for index, tokens in enumerate([["hello", "world"], ["this", "is", "tiny"]])
    ]
    model = HMMModel(label_mode="A", labels=load_label_sets()["A"], min_token_count=1)
    path = tmp_path_factory.mktemp("models") / "hmm.json"
    model.fit(records).save_model(path)
    return path


def test_cleans_punctuation_and_case() -> None:
    assert clean_tokens(["Hello,", "World."]) == ["hello", "world"]

    result = RestorationPipeline(StubAdapter()).process_text("Hello, World.")  # type: ignore[arg-type]

    assert result["tokens"] == ["hello", "world"]
    assert result["input_text"] == "Hello, World."
    assert result["predicted_labels"] == ["O", "O"]
    assert result["restored_text"] == "hello world"


def test_clean_tokens_drops_punctuation_only_tokens() -> None:
    assert clean_tokens(["Hello", "-", "...", "World!"]) == ["hello", "world"]


NOT_TEXT = [pytest.param(None, id="null"), pytest.param(1, id="number"), pytest.param({}, id="object"),
            pytest.param(["x"], id="list")]


@pytest.mark.parametrize("bad", NOT_TEXT)
def test_clean_tokens_refuses_a_token_that_is_not_text_in_one_line(bad: object) -> None:
    with pytest.raises(ValueError, match=r"^tokens must be strings, got ") as caught:
        clean_tokens(["a", bad])  # type: ignore[list-item]
    assert "\n" not in str(caught.value)


@pytest.mark.parametrize("bad", NOT_TEXT)
def test_reconstruct_refuses_a_token_or_a_label_that_is_not_text_in_one_line(bad: object) -> None:
    with pytest.raises(ValueError, match=r"^tokens must be strings, got "):
        reconstruct(["a", bad], ["O", "O"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match=r"^labels must be strings, got "):
        reconstruct(["a", "b"], ["O", bad])  # type: ignore[list-item]


def test_clean_tokens_returns_new_list() -> None:
    raw = ["Hello,"]

    cleaned = clean_tokens(raw)

    assert cleaned is not raw
    assert raw == ["Hello,"]


def test_clean_tokens_strips_only_edge_punctuation() -> None:
    assert clean_tokens(['"Well-known,"', "U.S."]) == ["well-known", "u.s"]


@pytest.mark.parametrize("tokens", [[], ["-", "..."], ['";']])
def test_clean_tokens_rejects_empty_input(tokens: list[str]) -> None:
    with pytest.raises(ValueError, match="empty input"):
        clean_tokens(tokens)


def test_reconstruct_first_upper_changes_only_first_character() -> None:
    assert reconstruct(["o'neil"], ["O|FIRST_UPPER"]) == "O'neil"
    assert reconstruct(["mcDonald"], ["O|FIRST_UPPER"]) == "McDonald"  # str.capitalize() gives "Mcdonald"


def test_reconstruct_all_upper() -> None:
    assert reconstruct(["nasa"], ["O|ALL_UPPER"]) == "NASA"


def test_reconstruct_dash_renders_with_leading_space() -> None:
    assert reconstruct(["word", "next"], ["DASH", "O"]) == "word - next"


def test_reconstruct_mode_c_joint_label() -> None:
    labels = ["O|FIRST_UPPER", "PERIOD|FIRST_UPPER"]

    assert reconstruct(["hello", "world"], labels) == "Hello World."


def test_reconstruct_plain_labels_are_lowercase_punctuation() -> None:
    labels = ["COMMA", "QUESTION", "EXCLAMATION", "O"]

    assert reconstruct(["a", "b", "c", "d"], labels) == "a, b? c! d"


def test_reconstruct_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="labels"):
        reconstruct(["a", "b"], ["O"])


def test_process_text_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="empty input"):
        RestorationPipeline(StubAdapter()).process_text("  ")  # type: ignore[arg-type]


def test_process_text_rejects_model_label_count_mismatch(hmm_json: Path) -> None:
    adapter = ModelAdapter(hmm_json)
    adapter.model.predict = lambda records: [{"predicted_labels": ["O"]}]  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="1 labels for 2 tokens"):
        RestorationPipeline(adapter).process_text("hello world")


def test_adapter_rejects_a_casing_helper_model(tmp_path: Path) -> None:
    records = [
        {"sample_id": f"c{index}", "split": "train", "tokens": ["we", "agree"],
         "punct_labels": ["O", "PERIOD"], "case_labels": ["FIRST_UPPER", "LOWER"]}
        for index in range(3)
    ]
    path = tmp_path / "case.joblib"
    CRFModel(label_mode="CASE", max_iterations=5).fit(records).save_model(path)

    with pytest.raises(ValueError, match="unsupported label mode 'CASE'"):
        ModelAdapter(path)


def test_end_to_end_with_saved_hmm(hmm_json: Path) -> None:
    result = RestorationPipeline(ModelAdapter(hmm_json)).process_text("Hello world")

    assert len(result["tokens"]) == len(result["predicted_labels"]) == 2
    assert result["restored_text"].startswith("hello world")


def test_empty_input_exits_1(hmm_json: Path) -> None:
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_demo.py"), "--model-path", str(hmm_json), "--text", "  "],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )

    assert done.returncode == 1
    assert len(done.stderr.strip().splitlines()) == 1
    assert done.stderr.startswith("error:")


def test_run_demo_text_prints_input_and_restored(
    hmm_json: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_demo.main(["--model-path", str(hmm_json), "--text", "hello world"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Input:    hello world" in out
    assert "Restored: hello world" in out


def test_run_demo_tolerates_a_stdout_without_reconfigure(
    hmm_json: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = io.StringIO()
    monkeypatch.setattr(sys, "stdout", fake)

    assert run_demo.main(["--model-path", str(hmm_json), "--text", "hello world"]) == 0
    assert "Restored: hello world" in fake.getvalue()


def test_run_demo_missing_model_prints_one_error_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_demo.main(["--model-path", str(tmp_path / "missing.json"), "--text", "hi"])

    err = capsys.readouterr().err
    assert code == 1
    assert len(err.strip().splitlines()) == 1
    assert err.startswith("error:")


def test_run_demo_interactive_loop_survives_bad_line(
    hmm_json: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = iter(["hello world", "...", "", "quit", "never read"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))

    code = run_demo.main(["--model-path", str(hmm_json)])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.count("Restored:") == 1
    assert captured.err.strip() == "error: empty input"


def test_run_demo_interactive_loop_ends_on_eof(
    hmm_json: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_eof(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    assert run_demo.main(["--model-path", str(hmm_json)]) == 0


def test_demo_text_survives_cp1252_stdout(hmm_json: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"PYTHONIOENCODING": "cp1252"}

    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_demo.py"),
            "--model-path",
            str(hmm_json),
            "--text",
            "hello Ωmega world",
        ],
        capture_output=True,
        cwd=ROOT,
        env=env,
        check=False,
    )

    out = done.stdout.decode("cp1252")
    assert done.returncode == 0
    assert done.stderr == b""
    assert "Restored:" in out
    assert "\\u03c9mega" in out


def test_demo_interactive_survives_cp1252_stdout(
    hmm_json: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(buffer, encoding="cp1252", errors="strict"))
    lines = iter(["hello Ωmega world", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))

    code = run_demo.main(["--model-path", str(hmm_json)])

    sys.stdout.flush()
    out = buffer.getvalue().decode("cp1252")
    assert code == 0
    assert "Restored:" in out
    assert "\\u03c9mega" in out


def test_label_sets_path_is_the_labeling_default() -> None:
    assert pipeline.LABEL_SETS_PATH is labeling.LABEL_SETS_PATH
    assert Path(pipeline.LABEL_SETS_PATH).samefile(ROOT / "configs" / "label_sets.yaml")
