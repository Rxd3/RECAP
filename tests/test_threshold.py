"""Tests for threshold decoding over CRF marginals and scripts/tune_threshold.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from punct_restore.evaluation.inputs import Vocab, load_bundle
from punct_restore.evaluation.manifest import check_manifest
from punct_restore.models.crf import CRFModel
from punct_restore.models.threshold import check_tau, decode_with_threshold, is_o_label
from scripts import tune_threshold

# ---------------------------------------------------------------- decode_with_threshold


def test_tau_one_returns_only_o_labels() -> None:
    marginals = [
        {"O": 0.01, "COMMA": 0.49, "PERIOD": 0.49, "QUESTION": 0.01},
        {"O": 0.0, "COMMA": 0.0, "PERIOD": 1.0, "QUESTION": 0.0},
        {"O": 0.9, "COMMA": 0.1, "PERIOD": 0.0, "QUESTION": 0.0},
    ]
    assert decode_with_threshold(marginals, 1) == ["O", "O", "O"]


def test_tau_zero_marks_any_token_with_nonzero_non_o_mass() -> None:
    assert decode_with_threshold([{"O": 0.99, "COMMA": 0.01}], 0) == ["COMMA"]


def test_mode_c_sums_all_o_labels() -> None:
    token = {"O|LOWER": 0.3, "O|FIRST_UPPER": 0.3, "COMMA|LOWER": 0.4}
    assert decode_with_threshold([token], 0.5)[0] in {"O|LOWER", "O|FIRST_UPPER"}
    assert decode_with_threshold([token], 0.3) == ["COMMA|LOWER"]


def test_all_mass_on_o_returns_best_o_label() -> None:
    token = {"O|LOWER": 0.2, "O|FIRST_UPPER": 0.8, "COMMA|LOWER": 0.0}
    assert decode_with_threshold([token], 0) == ["O|FIRST_UPPER"]


@pytest.mark.parametrize("tau", [-0.1, 1.1, math.nan, math.inf, True, "0.5", None, 10**400])
def test_rejects_tau_out_of_range(tau: Any) -> None:
    with pytest.raises(ValueError, match="tau"):
        decode_with_threshold([{"O": 1.0}], tau)


@pytest.mark.parametrize("tau", [0, 0.0, 0.5, 1, 1.0])
def test_check_tau_accepts_numbers_in_unit_interval(tau: float) -> None:
    assert check_tau(tau) is None


@pytest.mark.parametrize("tau", [-0.1, 1.1, math.nan, math.inf, True, "0.5", None])
def test_check_tau_rejects_everything_else(tau: Any) -> None:
    with pytest.raises(ValueError, match="tau must be a number between 0 and 1"):
        check_tau(tau)


def test_rejects_empty_or_o_free_token() -> None:
    with pytest.raises(ValueError, match="token 0"):
        decode_with_threshold([{}], 0.5)
    with pytest.raises(ValueError, match="token 1"):
        decode_with_threshold([{"O": 1.0}, {"COMMA": 1.0}], 0.5)


def test_threshold_is_strict_and_first_label_wins_ties() -> None:
    assert decode_with_threshold([{"O": 0.5, "COMMA": 0.5}], 0.5) == ["O"]
    assert decode_with_threshold([{"O": 0.5, "COMMA": 0.5}], 0.49) == ["COMMA"]
    assert decode_with_threshold([{"O": 0.5, "COMMA": 0.25, "PERIOD": 0.25}], 0) == ["COMMA"]


def test_decode_returns_new_list_and_leaves_input_untouched() -> None:
    marginals = [{"O": 0.4, "COMMA": 0.6}]
    snapshot = [dict(token) for token in marginals]
    result = decode_with_threshold(marginals, 0.5)
    assert result == ["COMMA"]
    assert marginals == snapshot
    assert decode_with_threshold([], 0.5) == []


def test_is_o_label() -> None:
    assert is_o_label("O") and is_o_label("O|LOWER")
    assert not is_o_label("COMMA") and not is_o_label("COMMA|FIRST_UPPER") and not is_o_label("OX")


# ---------------------------------------------------------------- pick_tau and parse_grid


def test_pick_tau_prefers_highest_f1_then_nearest_half_then_lower() -> None:
    assert tune_threshold.pick_tau([(0.2, 0.5), (0.4, 0.7), (0.6, 0.6)]) == 0.4
    assert tune_threshold.pick_tau([(0.1, 0.7), (0.4, 0.7), (0.9, 0.7)]) == 0.4
    assert tune_threshold.pick_tau([(0.3, 0.7), (0.7, 0.7)]) == 0.3
    assert tune_threshold.pick_tau([(0.7, 0.7), (0.3, 0.7)]) == 0.3  # lower tau, whatever the order
    with pytest.raises(ValueError, match="empty"):
        tune_threshold.pick_tau([])


def test_parse_grid_default_and_inclusive_stop() -> None:
    default = tune_threshold.parse_grid("0.05:0.95:0.05")
    assert len(default) == 19 and default[0] == 0.05 and default[-1] == 0.95
    assert tune_threshold.parse_grid("0.1:0.9:0.2") == (0.1, 0.3, 0.5, 0.7, 0.9)
    assert tune_threshold.parse_grid("0:1:1") == (0.0, 1.0)
    assert tune_threshold.parse_grid("0.5:0.5:0.1") == (0.5,)


@pytest.mark.parametrize(
    "text",
    ["", "0.1:0.9", "0.1:0.9:0.1:0.2", "a:b:c", "0.5:0.1:0.1", "-0.1:0.5:0.1",
     "0.1:1.5:0.1", "0.1:0.9:0", "0.1:0.9:-0.1", "nan:0.9:0.1", "0.1:inf:0.1"],
)
def test_parse_grid_rejects_bad_text(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        tune_threshold.parse_grid(text)


def test_bad_grid_is_a_usage_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        tune_threshold.main(["tune", "--model", "m", "--gold", "g", "--ids", "i",
                             "--grid", "0.9:0.1:0.1", "--out", str(tmp_path / "t.json")])
    assert exit_info.value.code == 2
    assert "grid" in capsys.readouterr().err


def test_parse_grid_accepts_exactly_1000_values() -> None:
    grid = tune_threshold.parse_grid("0:0.999:0.001")
    assert len(grid) == 1000 and grid[0] == 0.0 and grid[-1] == 0.999


@pytest.mark.parametrize("text", ["0:1:0.001", "0:1:0.0005", "0:1:1e-12", "0:1:5e-324"])
def test_parse_grid_rejects_more_than_1000_values(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="1000"):
        tune_threshold.parse_grid(text)


@pytest.mark.parametrize("grid", ["0:1:0.0005", "0:1:1e-12", "0:1:5e-324"])
def test_huge_grid_is_a_quick_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], grid: str
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        tune_threshold.main(["tune", "--model", "m", "--gold", "g", "--ids", "i",
                             "--grid", grid, "--out", str(tmp_path / "t.json")])
    assert exit_info.value.code == 2
    assert "grid" in capsys.readouterr().err


# ---------------------------------------------------------------- CLI fixtures

_PUNCT_ROWS = [
    (["well", "i", "think", "so"], ["COMMA", "O", "O", "PERIOD"]),
    (["how", "are", "you"], ["O", "O", "QUESTION"]),
    (["this", "is", "a", "test"], ["O", "O", "O", "PERIOD"]),
    (["hello", "world", "again"], ["COMMA", "O", "PERIOD"]),
    (["what", "is", "this"], ["O", "O", "QUESTION"]),
    (["thank", "you", "very", "much"], ["O", "COMMA", "O", "PERIOD"]),
]


@dataclass(frozen=True)
class Fixture:
    model: Path
    gold: Path
    ids: Path
    masks: Path | None


def _rows(mode: str, prefix: str) -> list[dict[str, Any]]:
    return [
        {"sample_id": f"{prefix}{index}", "label_mode": mode, "tokens": tokens,
         "punct_labels": punct, "case_labels": ["FIRST_UPPER"] + ["LOWER"] * (len(tokens) - 1)}
        for index, (tokens, punct) in enumerate(_PUNCT_ROWS)
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _build_fixture(root: Path, mode: str) -> Fixture:
    model = CRFModel(label_mode=mode, max_iterations=50)
    model.fit(_rows(mode, "train"))
    model_path = root / "m.joblib"
    model.save_model(model_path)

    gold = _rows(mode, "gold")
    gold_path, ids_path = root / "gold.jsonl", root / "gold.ids"
    _write_jsonl(gold_path, gold)
    ids_path.write_text("".join(row["sample_id"] + "\n" for row in gold), encoding="utf-8")
    masks_path = None
    if mode == "C":
        masks_path = root / "masks.jsonl"
        _write_jsonl(masks_path, [{"sample_id": row["sample_id"], "excluded_positions": [],
                                   "reason": None} for row in gold])
    return Fixture(model_path, gold_path, ids_path, masks_path)


@pytest.fixture(scope="module")
def mode_a(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return _build_fixture(tmp_path_factory.mktemp("mode_a"), "A")


@pytest.fixture(scope="module")
def mode_c(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return _build_fixture(tmp_path_factory.mktemp("mode_c"), "C")


@pytest.fixture(scope="module")
def case_fixture(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    """A saved casing-helper (label mode CASE) model: it loads, but it is not a scoring mode."""
    return _build_fixture(tmp_path_factory.mktemp("case"), "CASE")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tune_argv(fx: Fixture, out: Path, grid: str = "0.1:0.9:0.2") -> list[str]:
    argv = ["tune", "--model", str(fx.model), "--gold", str(fx.gold), "--ids", str(fx.ids),
            "--grid", grid, "--out", str(out)]
    return argv + (["--masks", str(fx.masks)] if fx.masks is not None else [])


def _expected_tau(grid: list[dict[str, float]]) -> float:
    best = max(point["macro_f1"] for point in grid)
    tied = [point["tau"] for point in grid if point["macro_f1"] == best]
    return min(tied, key=lambda tau: (round(abs(tau - 0.5), 9), tau))


# ---------------------------------------------------------------- tune


def test_tune_picks_best_tau_on_fixture(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run" / "tau.json"
    assert tune_threshold.main(_tune_argv(mode_a, out)) == 0

    written = json.loads(out.read_text(encoding="utf-8"))
    assert set(written) == {"tau", "gamma", "grid", "model_sha256", "split", "inputs"}
    assert written["gamma"] == 0 and all(point["gamma"] == 0 for point in written["grid"])
    assert [point["tau"] for point in written["grid"]] == [0.1, 0.3, 0.5, 0.7, 0.9]
    assert all(0.0 <= point["macro_f1"] <= 1.0 for point in written["grid"])
    assert len({point["macro_f1"] for point in written["grid"]}) > 1  # a scorer ignoring tau fails
    assert written["tau"] == _expected_tau(written["grid"])
    assert written["model_sha256"] == _sha256(mode_a.model)
    assert [path.name for path in out.parent.iterdir()] == ["tau.json"]  # temp dir cleaned up

    line = capsys.readouterr().out.strip()
    assert re.fullmatch(r"tau=\S+ gamma=0 macro_f1=\d\.\d{6} \(5 values scored\)", line)


def test_tune_mode_c_with_masks(mode_c: Fixture, tmp_path: Path) -> None:
    out = tmp_path / "tau.json"
    assert tune_threshold.main(_tune_argv(mode_c, out, "0.2:0.8:0.3")) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert [point["tau"] for point in written["grid"]] == [0.2, 0.5, 0.8]
    assert written["tau"] == _expected_tau(written["grid"])


def test_tune_refuses_existing_out(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "tau.json"
    out.write_text("keep me", encoding="utf-8")
    assert tune_threshold.main(_tune_argv(mode_a, out)) == 1
    assert out.read_text(encoding="utf-8") == "keep me"
    err = capsys.readouterr().err
    assert err.startswith("error: ") and len(err.splitlines()) == 1


def test_tune_rejects_gold_mode_different_from_model_mode(
    mode_a: Fixture, mode_c: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "tau.json"
    argv = ["tune", "--model", str(mode_a.model), "--gold", str(mode_c.gold),
            "--ids", str(mode_c.ids), "--out", str(out)]
    assert tune_threshold.main(argv) == 1
    assert not out.exists()
    err = capsys.readouterr().err
    assert err.startswith("error: ") and len(err.splitlines()) == 1 and "mode" in err


def test_tune_rejects_a_casing_helper_model(
    mode_a: Fixture, case_fixture: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "tau.json"
    assert tune_threshold.main(_tune_argv(replace(mode_a, model=case_fixture.model), out)) == 1
    assert not out.exists()
    err = capsys.readouterr().err
    assert err.startswith("error: ") and len(err.splitlines()) == 1 and "CASE" in err and "scoring mode" in err


def test_tune_mode_c_without_masks_is_a_friendly_error(
    mode_c: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "tau.json"
    argv = _tune_argv(mode_c, out)[:-2]  # drop --masks
    assert tune_threshold.main(argv) == 1
    assert not out.exists()
    assert "masks" in capsys.readouterr().err


def test_tune_missing_model_is_a_friendly_error(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "tau.json"
    argv = _tune_argv(mode_a, out)
    argv[argv.index("--model") + 1] = str(tmp_path / "ghost.joblib")
    assert tune_threshold.main(argv) == 1
    assert not out.exists()
    err = capsys.readouterr().err
    assert err.startswith("error: ") and len(err.splitlines()) == 1


# ---------------------------------------------------------------- predict


def _write_tau_file(path: Path, tau: Any, sha: Any) -> Path:
    path.write_text(json.dumps({"tau": tau, "grid": [], "model_sha256": sha}), encoding="utf-8")
    return path


def _predict_argv(fx: Fixture, tau_file: Path, out: Path, man: Path) -> list[str]:
    return ["predict", "--model", str(fx.model), "--input", str(fx.gold), "--ids", str(fx.ids),
            "--tau-file", str(tau_file), "--out", str(out), "--manifest-out", str(man)]


def test_predict_end_to_end_is_thresholded_and_manifest_valid(
    mode_a: Fixture, vocab: Vocab, tmp_path: Path
) -> None:
    tau_file = _write_tau_file(tmp_path / "tau.json", 0.35, _sha256(mode_a.model))
    out, man = tmp_path / "preds.jsonl", tmp_path / "preds.man.json"
    assert tune_threshold.main(_predict_argv(mode_a, tau_file, out, man)) == 0

    predictions = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    model = CRFModel.load_model(mode_a.model)
    assert all(p["model_name"] == f"{model.model_name}_tau0.35" for p in predictions)
    assert all(p["model_name"].endswith("_tau0.35") and p["label_mode"] == "A" for p in predictions)

    records = [{"sample_id": row["sample_id"], "tokens": row["tokens"]}
               for row in _rows("A", "gold")]
    expected = [decode_with_threshold(m, 0.35) for m in model.predict_marginals(records)]
    assert [p["predicted_labels"] for p in predictions] == expected

    check_manifest(man, load_bundle(mode_a.gold, mode_a.ids, out, None, vocab), vocab)


def test_predict_rejects_a_casing_helper_model(
    case_fixture: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tau_file = _write_tau_file(tmp_path / "tau.json", 0.35, _sha256(case_fixture.model))
    out, man = tmp_path / "preds.jsonl", tmp_path / "preds.man.json"
    assert tune_threshold.main(_predict_argv(case_fixture, tau_file, out, man)) == 1
    assert not out.exists() and not man.exists()
    err = capsys.readouterr().err
    assert err.startswith("error: ") and len(err.splitlines()) == 1 and "CASE" in err and "scoring mode" in err


def test_predict_tau_of_negative_zero_is_named_tau0(mode_a: Fixture, tmp_path: Path) -> None:
    tau_file = _write_tau_file(tmp_path / "tau.json", -0.0, _sha256(mode_a.model))
    out, man = tmp_path / "preds.jsonl", tmp_path / "preds.man.json"
    assert tune_threshold.main(_predict_argv(mode_a, tau_file, out, man)) == 0

    names = {json.loads(line)["model_name"] for line in out.read_text(encoding="utf-8").splitlines()}
    assert len(names) == 1 and names.pop().endswith("_tau0")
    assert json.loads(man.read_text(encoding="utf-8"))["model_name"].endswith("_tau0")


def test_predict_rejects_tau_tuned_for_another_model(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tau_file = _write_tau_file(tmp_path / "tau.json", 0.35, "0" * 64)
    out, man = tmp_path / "preds.jsonl", tmp_path / "preds.man.json"
    assert tune_threshold.main(_predict_argv(mode_a, tau_file, out, man)) == 1
    assert not out.exists() and not man.exists()
    err = capsys.readouterr().err
    assert len(err.splitlines()) == 1 and "different model" in err


@pytest.mark.parametrize("which", ["out", "manifest"])
def test_predict_refuses_existing_outputs(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str], which: str
) -> None:
    tau_file = _write_tau_file(tmp_path / "tau.json", 0.35, _sha256(mode_a.model))
    out, man = tmp_path / "preds.jsonl", tmp_path / "preds.man.json"
    existing = out if which == "out" else man
    existing.write_text("keep me", encoding="utf-8")
    assert tune_threshold.main(_predict_argv(mode_a, tau_file, out, man)) == 1
    assert existing.read_text(encoding="utf-8") == "keep me"
    assert len(list(tmp_path.glob("preds*"))) == 1  # the other output was not created
    assert capsys.readouterr().err.startswith("error: ")


@pytest.mark.parametrize(
    "content",
    ["{not json", "[1, 2]", json.dumps({"tau": 0.5}), json.dumps({"tau": "0.5", "model_sha256": "x"}),
     json.dumps({"tau": True, "model_sha256": "x"}), json.dumps({"tau": 1.5, "model_sha256": "x"}),
     json.dumps({"tau": 0.5, "model_sha256": 7})],
)
def test_predict_rejects_malformed_tau_file(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str
) -> None:
    tau_file = tmp_path / "tau.json"
    tau_file.write_text(content, encoding="utf-8")
    out, man = tmp_path / "preds.jsonl", tmp_path / "preds.man.json"
    assert tune_threshold.main(_predict_argv(mode_a, tau_file, out, man)) == 1
    assert not out.exists() and not man.exists()
    err = capsys.readouterr().err
    assert err.startswith("error: ") and len(err.splitlines()) == 1 and "tau file" in err


def test_predict_missing_tau_file_is_a_friendly_error(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _predict_argv(mode_a, tmp_path / "ghost.json", tmp_path / "p.jsonl", tmp_path / "p.man")
    assert tune_threshold.main(argv) == 1
    assert len(capsys.readouterr().err.splitlines()) == 1


def test_missing_subcommand_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        tune_threshold.main([])
    assert exit_info.value.code == 2
