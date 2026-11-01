"""Tests for prior-adjusted per-class decoding (gamma) in threshold.py and tune_threshold.py."""

from __future__ import annotations

import itertools
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import pytest
from test_threshold import (
    _PUNCT_ROWS,
    Fixture,
    _build_fixture,
    _predict_argv,
    _rows,
    _sha256,
    _tune_argv,
    _write_jsonl,
)

from punct_restore.cli import tune_threshold
from punct_restore.evaluation.inputs import Vocab, load_bundle
from punct_restore.evaluation.manifest import check_manifest
from punct_restore.models import threshold
from punct_restore.models.crf import CRFModel, get_mode_vocabulary
from punct_restore.models.threshold import decode_with_threshold, is_o_label

MODE_A = ["O", "COMMA", "PERIOD", "QUESTION"]
MODE_C = get_mode_vocabulary("C")
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)

# ---------------------------------------------------------------- gamma = 0 changes nothing


def _random_marginals(labels: list[str], seed: int, count: int = 200) -> list[dict[str, float]]:
    rng = random.Random(seed)
    tokens = []
    for _ in range(count):
        weights = [rng.random() ** 3 for _ in labels]
        total = sum(weights)
        tokens.append({label: weight / total for label, weight in zip(labels, weights, strict=True)})
    return tokens


def _todays_decode(marginals: list[dict[str, float]], tau: float) -> list[str]:
    """Frozen copy of decode_with_threshold as it was before gamma existed."""
    labels = []
    for probs in marginals:
        o_labels = [label for label in probs if is_o_label(label)]
        punct = [label for label in probs if not is_o_label(label)]
        if punct and 1 - sum(probs[label] for label in o_labels) > tau:
            labels.append(max(punct, key=probs.__getitem__))
        else:
            labels.append(max(o_labels, key=probs.__getitem__))
    return labels


@pytest.mark.parametrize("labels", [MODE_A, MODE_C], ids=["A", "C"])
@pytest.mark.parametrize("tau", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_gamma_zero_reproduces_todays_output(labels: list[str], tau: float) -> None:
    marginals = _random_marginals(labels, seed=7)
    expected = _todays_decode(marginals, tau)
    if tau == 0.5:  # not vacuous: both branches are exercised
        assert any(is_o_label(label) for label in expected)
        assert any(not is_o_label(label) for label in expected)
    priors = {label: 0.05 for label in labels if not is_o_label(label)}
    assert decode_with_threshold(marginals, tau) == expected
    for gamma in (0, 0.0, -0.0):
        assert decode_with_threshold(marginals, tau, gamma=gamma) == expected
        assert decode_with_threshold(marginals, tau, priors=priors, gamma=gamma) == expected
    assert decode_with_threshold(marginals, tau, priors="ignored at gamma 0", gamma=0) == expected


def test_gamma_zero_frozen_expected_labels() -> None:
    marginals = [
        {"O": 0.6, "COMMA": 0.3, "PERIOD": 0.1, "QUESTION": 0.0},
        {"O": 0.2, "COMMA": 0.2, "PERIOD": 0.5, "QUESTION": 0.1},
        {"O": 0.5, "COMMA": 0.25, "PERIOD": 0.25, "QUESTION": 0.0},
        {"O": 0.9, "COMMA": 0.04, "PERIOD": 0.03, "QUESTION": 0.03},
    ]
    expected = ["COMMA", "PERIOD", "COMMA", "O"]
    assert decode_with_threshold(marginals, 0.3) == expected
    assert decode_with_threshold(marginals, 0.3, gamma=0) == expected


# ---------------------------------------------------------------- gamma > 0

# Hand arithmetic. Token O=0.2, COMMA=0.5, QUESTION=0.3; priors COMMA=0.40, QUESTION=0.05.
#   gate (tau 0.5): 1 - 0.2 = 0.8 > 0.5 fires; (tau 0.85): 0.8 is not > 0.85, the token stays O.
#   gamma 0: COMMA 0.5 beats QUESTION 0.3.
#   gamma 1: COMMA 0.5 / 0.40 = 1.25, QUESTION 0.3 / 0.05 = 6.0, so the rare class wins.
RARE = {"O": 0.2, "COMMA": 0.5, "QUESTION": 0.3}
RARE_PRIORS = {"COMMA": 0.40, "QUESTION": 0.05}


def test_rare_class_wins_with_gamma_one_but_not_zero() -> None:
    assert decode_with_threshold([RARE], 0.5, priors=RARE_PRIORS, gamma=0) == ["COMMA"]
    assert decode_with_threshold([RARE], 0.5, priors=RARE_PRIORS, gamma=1) == ["QUESTION"]
    assert decode_with_threshold([RARE], 0.5, priors=RARE_PRIORS, gamma=1.0) == ["QUESTION"]


def test_gate_is_unchanged_by_gamma() -> None:
    assert decode_with_threshold([RARE], 0.85, priors=RARE_PRIORS, gamma=1.0) == ["O"]
    assert decode_with_threshold([RARE], 0.79, priors=RARE_PRIORS, gamma=1.0) == ["QUESTION"]


def test_mode_c_reweights_only_the_non_o_joint_labels() -> None:
    # 1 - P(O) = 1 - (0.3 + 0.1) = 0.6 > 0.5. gamma 1: COMMA|LOWER 0.35 / 0.5 = 0.7,
    # QUESTION|LOWER 0.25 / 0.02 = 12.5. No O|* prior is needed.
    token = {"O|LOWER": 0.3, "O|FIRST_UPPER": 0.1, "COMMA|LOWER": 0.35, "QUESTION|LOWER": 0.25}
    priors = {"COMMA|LOWER": 0.5, "QUESTION|LOWER": 0.02}
    assert decode_with_threshold([token], 0.5, priors=priors, gamma=0) == ["COMMA|LOWER"]
    assert decode_with_threshold([token], 0.5, priors=priors, gamma=1) == ["QUESTION|LOWER"]


def test_gamma_ties_go_to_the_first_label_in_mapping_order() -> None:
    priors = {"COMMA": 0.3, "PERIOD": 0.3}
    assert decode_with_threshold([{"O": 0.2, "COMMA": 0.4, "PERIOD": 0.4}], 0.5,
                                 priors=priors, gamma=1) == ["COMMA"]
    assert decode_with_threshold([{"O": 0.2, "PERIOD": 0.4, "COMMA": 0.4}], 0.5,
                                 priors=priors, gamma=1) == ["PERIOD"]
    assert decode_with_threshold([{"O": 0.5, "COMMA": 0.0, "PERIOD": 0.0}], 0.4,
                                 priors=priors, gamma=1) == ["COMMA"]


def test_huge_gamma_cannot_overflow() -> None:
    token = {"O": 0.2, "COMMA": 0.4, "PERIOD": 0.4}
    priors = {"COMMA": 1e-300, "PERIOD": 0.5}
    assert decode_with_threshold([token], 0.5, priors=priors, gamma=1e308) == ["COMMA"]


def test_unused_priors_need_not_be_usable() -> None:
    token = {"O": 0.2, "COMMA": 0.8}
    assert decode_with_threshold([token], 0.5, priors={"COMMA": 0.4, "PERIOD": 0.0}, gamma=1) == ["COMMA"]


def test_decode_with_gamma_leaves_inputs_untouched() -> None:
    marginals, priors = [dict(RARE)], dict(RARE_PRIORS)
    decode_with_threshold(marginals, 0.5, priors=priors, gamma=1)
    assert marginals == [RARE] and priors == RARE_PRIORS


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "gamma", [-0.1, -math.inf, math.nan, math.inf, True, False, "1", None, [1], 10**400]
)
def test_rejects_bad_gamma(gamma: Any) -> None:
    with pytest.raises(ValueError, match="gamma must be a finite number >= 0") as info:
        decode_with_threshold([RARE], 0.5, priors=RARE_PRIORS, gamma=gamma)
    assert "\n" not in str(info.value)


@pytest.mark.parametrize("gamma", [0, 0.0, 0.25, 1, 3.5])
def test_check_gamma_accepts_finite_non_negative_numbers(gamma: float) -> None:
    assert threshold.check_gamma(gamma) is None


@pytest.mark.parametrize("priors", [None, [0.4], "x", 3])
def test_gamma_needs_a_priors_mapping(priors: Any) -> None:
    with pytest.raises(ValueError, match="priors"):
        decode_with_threshold([RARE], 0.5, priors=priors, gamma=1.0)


@pytest.mark.parametrize("bad", [0, 0.0, -0.2, math.nan, math.inf, True, "0.4", None, 10**400])
def test_gamma_rejects_an_unusable_prior(bad: Any) -> None:
    with pytest.raises(ValueError, match="prior > 0 for label 'QUESTION'") as info:
        decode_with_threshold([RARE], 0.9, priors={"COMMA": 0.4, "QUESTION": bad}, gamma=1.0)
    assert "\n" not in str(info.value)


def test_gamma_rejects_a_missing_prior_even_when_the_gate_stays_closed() -> None:
    with pytest.raises(ValueError, match="'QUESTION'"):
        decode_with_threshold([RARE], 0.99, priors={"COMMA": 0.4}, gamma=1.0)


def test_check_priors_accepts_positive_finite_numbers_only() -> None:
    assert threshold.check_priors({"COMMA": 0.1, "PERIOD": 1}) is None
    for bad in ([], "x", None, {"COMMA": 0}, {"COMMA": math.nan}, {"COMMA": True}, {"COMMA": "0.1"}):
        with pytest.raises(ValueError, match="prior"):
            threshold.check_priors(bad)


# ---------------------------------------------------------------- label_priors


def test_label_priors_add_one_counts_by_hand_mode_a() -> None:
    # 6 tokens: COMMA x3, PERIOD x1, O x2, QUESTION x0; K = 4, so the denominator is 6 + 4 = 10.
    records = [{"punct_labels": ["COMMA", "COMMA", "O", "PERIOD"]},
               {"punct_labels": ["O", "COMMA"]}]
    priors = threshold.label_priors(records, MODE_A)
    assert priors == pytest.approx({"COMMA": 0.4, "PERIOD": 0.2, "QUESTION": 0.1})
    assert threshold.label_priors(iter(records), MODE_A) == priors  # a one-shot stream works too


def test_label_priors_mode_b_uses_all_eight_labels() -> None:
    labels = get_mode_vocabulary("B")  # K = 8, 2 tokens: EXCLAMATION x1, O x1 -> denominator 10
    priors = threshold.label_priors([{"punct_labels": ["EXCLAMATION", "O"]}], labels)
    assert len(priors) == 7 and priors["EXCLAMATION"] == pytest.approx(0.2)
    assert priors["DASH"] == pytest.approx(0.1)


def test_label_priors_joint_mode_c_by_hand() -> None:
    # Joint labels: COMMA|FIRST_UPPER, O|LOWER, PERIOD|LOWER, O|LOWER, O|FIRST_UPPER.
    # 5 tokens, K = 12: denominator 17; the O|* labels are not reported.
    records = [{"punct_labels": ["COMMA", "O", "PERIOD"], "case_labels": ["FIRST_UPPER", "LOWER", "LOWER"]},
               {"punct_labels": ["O", "O"], "case_labels": ["LOWER", "FIRST_UPPER"]}]
    priors = threshold.label_priors(records, MODE_C)
    assert len(priors) == 9 and not any(is_o_label(label) for label in priors)
    assert priors["COMMA|FIRST_UPPER"] == pytest.approx(2 / 17)
    assert priors["PERIOD|LOWER"] == pytest.approx(2 / 17)
    assert priors["QUESTION|ALL_UPPER"] == pytest.approx(1 / 17)


@pytest.mark.parametrize(
    ("labels", "record", "message"),
    [
        (MODE_A, {"tokens": ["a"]}, "punct_labels"),
        (MODE_A, {"punct_labels": ["O", 3]}, "punct_labels"),
        (MODE_A, {"punct_labels": ["O", "EXCLAMATION"]}, "outside"),
        (MODE_C, {"punct_labels": ["O"]}, "case_labels"),
        (MODE_C, {"punct_labels": ["O", "O"], "case_labels": ["LOWER"]}, "length"),
        (MODE_C, {"punct_labels": ["O"], "case_labels": ["WEIRD"]}, "outside"),
    ],
)
def test_label_priors_rejects_bad_records(labels: list[str], record: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message) as info:
        threshold.label_priors([record], labels)
    assert "\n" not in str(info.value)


# ---------------------------------------------------------------- pick_tau_gamma and the grid


def test_gammas_are_the_documented_grid() -> None:
    assert tune_threshold.GAMMAS == GAMMAS


def test_pick_tau_gamma_prefers_f1_then_smallest_gamma_then_tau_rule() -> None:
    pick = tune_threshold.pick_tau_gamma
    assert pick([(0.5, 0.0, 0.6), (0.3, 0.5, 0.8), (0.7, 1.0, 0.7)]) == (0.3, 0.5)
    # F1 ties across gammas: the smallest gamma wins even though its tau is far from 0.5.
    assert pick([(0.5, 0.5, 0.7), (0.1, 0.0, 0.7), (0.5, 1.0, 0.7)]) == (0.1, 0.0)
    # Same gamma: nearest 0.5, then the lower tau, whatever the order.
    assert pick([(0.1, 0.25, 0.7), (0.4, 0.25, 0.7), (0.9, 0.25, 0.7)]) == (0.4, 0.25)
    assert pick([(0.7, 0.25, 0.7), (0.3, 0.25, 0.7)]) == (0.3, 0.25)
    with pytest.raises(ValueError, match="empty"):
        pick([])


def test_score_grid_decodes_every_tau_gamma_pair_with_the_priors(tmp_path: Path) -> None:
    seen: list[tuple[str, list[str]]] = []

    def score_file(path: Path) -> float:
        record = json.loads(path.read_text(encoding="utf-8"))
        seen.append((record["model_name"], record["predicted_labels"]))
        return 1.0 if record["predicted_labels"] == ["QUESTION"] else 0.0

    scores = threshold.score_grid(
        threshold.Marginals("m", "A", ["s"], [[RARE]], RARE_PRIORS), (0.5, 0.85), score_file, tmp_path,
        gammas=(0.0, 1.0))
    assert scores == [(0.5, 0.0, 0.0), (0.85, 0.0, 0.0), (0.5, 1.0, 1.0), (0.85, 1.0, 0.0)]
    assert {name for name, _ in seen} == {"m_tau0.5", "m_tau0.85", "m_tau0.5_gamma1", "m_tau0.85_gamma1"}


# ---------------------------------------------------------------- CLI tune

TRAIN_IDS = "".join(f"train{i}\n" for i in range(len(_PUNCT_ROWS)))


@pytest.fixture(scope="module")
def mode_a(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return _build_fixture(tmp_path_factory.mktemp("prior_mode_a"), "A")


@pytest.fixture(scope="module")
def mode_c(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return _build_fixture(tmp_path_factory.mktemp("prior_mode_c"), "C")


def _write_train(root: Path, mode: str) -> tuple[Path, Path]:
    """The six fixture training rows plus an unlisted decoy that would change every prior."""
    decoy = {"sample_id": "decoy", "label_mode": mode, "tokens": ["x", "y"],
             "punct_labels": ["PERIOD", "PERIOD"], "case_labels": ["LOWER", "LOWER"]}
    train, ids = root / "train.jsonl", root / "train.ids"
    _write_jsonl(train, [*_rows(mode, "train"), decoy])
    ids.write_text(TRAIN_IDS, encoding="utf-8")
    return train, ids


def _train_argv(fx: Fixture, out: Path, train: Path, ids: Path, grid: str = "0.1:0.9:0.2") -> list[str]:
    return [*_tune_argv(fx, out, grid), "--train", str(train), "--train-ids", str(ids)]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_tune_with_train_searches_tau_and_gamma(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    train, ids = _write_train(tmp_path, "A")
    out = tmp_path / "run" / "tau.json"
    assert tune_threshold.main(_train_argv(mode_a, out, train, ids)) == 0
    written = _read(out)

    assert set(written) == {"tau", "gamma", "priors", "grid", "model_sha256", "split", "inputs"}
    # Hand priors from the six listed rows: 21 tokens, K = 4, COMMA 3, PERIOD 4, QUESTION 2.
    assert written["priors"] == pytest.approx({"COMMA": 4 / 25, "PERIOD": 5 / 25, "QUESTION": 3 / 25})
    grid = written["grid"]
    assert {(p["tau"], p["gamma"]) for p in grid} == set(
        itertools.product([0.1, 0.3, 0.5, 0.7, 0.9], GAMMAS))
    assert len(grid) == 25 and all(0.0 <= p["macro_f1"] <= 1.0 for p in grid)
    best = max(p["macro_f1"] for p in grid)
    chosen = min((p for p in grid if p["macro_f1"] == best),
                 key=lambda p: (p["gamma"], round(abs(p["tau"] - 0.5), 9), p["tau"]))
    assert (written["tau"], written["gamma"]) == (chosen["tau"], chosen["gamma"])
    assert written["model_sha256"] == _sha256(mode_a.model)
    assert [path.name for path in out.parent.iterdir()] == ["tau.json"]  # temp dir cleaned up

    line = capsys.readouterr().out.strip()
    assert re.fullmatch(r"tau=\S+ gamma=\S+ macro_f1=\d\.\d{6} \(25 values scored\)", line)

    plain = tmp_path / "plain.json"
    assert tune_threshold.main(_tune_argv(mode_a, plain)) == 0
    assert [(p["tau"], p["macro_f1"]) for p in grid if p["gamma"] == 0] == [
        (p["tau"], p["macro_f1"]) for p in _read(plain)["grid"]]  # gamma 0 rows are today's rows


def test_tune_without_train_keeps_gamma_zero_and_has_no_priors(mode_a: Fixture, tmp_path: Path) -> None:
    out = tmp_path / "tau.json"
    assert tune_threshold.main(_tune_argv(mode_a, out)) == 0
    written = _read(out)
    assert set(written) == {"tau", "gamma", "grid", "model_sha256", "split", "inputs"} and written["gamma"] == 0
    assert len(written["grid"]) == 5 and all(p["gamma"] == 0 for p in written["grid"])


def test_tune_mode_c_with_train_uses_joint_priors(mode_c: Fixture, tmp_path: Path) -> None:
    train, ids = _write_train(tmp_path, "C")
    out = tmp_path / "tau.json"
    assert tune_threshold.main(_train_argv(mode_c, out, train, ids, "0.2:0.8:0.3")) == 0
    written = _read(out)
    assert len(written["grid"]) == 15
    # 21 tokens, K = 12: denominator 33. Counts: COMMA|FIRST_UPPER 2, COMMA|LOWER 1,
    # PERIOD|LOWER 4, QUESTION|LOWER 2, every other non-O joint label 0.
    expected = {label: 1 / 33 for label in MODE_C if not is_o_label(label)}
    expected.update({"COMMA|FIRST_UPPER": 3 / 33, "COMMA|LOWER": 2 / 33,
                     "PERIOD|LOWER": 5 / 33, "QUESTION|LOWER": 3 / 33})
    assert written["priors"] == pytest.approx(expected)


@pytest.mark.parametrize("extra", [["--train", "t.jsonl"], ["--train-ids", "t.ids"]])
def test_tune_train_options_must_come_together(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    out = tmp_path / "t.json"
    with pytest.raises(SystemExit) as exit_info:
        tune_threshold.main(["tune", "--model", "m", "--gold", "g", "--ids", "i", *extra,
                             "--out", str(out)])
    assert exit_info.value.code == 2
    assert "--train and --train-ids must be given together" in capsys.readouterr().err
    assert not out.exists()


def _tune_with_train_text(
    fx: Fixture, tmp_path: Path, text: str | bytes, capsys: pytest.CaptureFixture[str]
) -> str:
    """Run tune against a custom train file; expect a one-line error and no tau file."""
    train, ids = tmp_path / "train.jsonl", tmp_path / "train.ids"
    train.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    ids.write_text(TRAIN_IDS, encoding="utf-8")
    out = tmp_path / "tau.json"
    assert tune_threshold.main(_train_argv(fx, out, train, ids)) == 1
    assert not out.exists()
    err = capsys.readouterr().err
    assert err.startswith("error: ") and len(err.splitlines()) == 1
    return err


def _lines(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


@pytest.mark.parametrize("variant", ["bad_json", "not_object", "no_id", "missing", "duplicate",
                                     "wrong_mode", "no_labels", "outside_vocab", "not_utf8"])
def test_tune_train_file_errors_are_one_friendly_line(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str], variant: str
) -> None:
    rows = _rows("A", "train")
    cases: dict[str, tuple[str | bytes, str]] = {
        "bad_json": ("\n{oops\n", "train.jsonl line 2"),
        "not_object": ("[1]\n", "train.jsonl line 1"),
        "no_id": (_lines([{"tokens": ["a"]}]), "train.jsonl line 1"),
        "missing": (_lines(rows[:5]), "1 listed ID(s) are missing from train.jsonl"),
        "duplicate": (_lines([*rows, rows[0]]), "train.jsonl line 7"),
        "wrong_mode": (_lines([*rows[:5], {**rows[5], "label_mode": "C"}]), "train.jsonl line 6"),
        "no_labels": (_lines([*rows[:5], {"sample_id": "train5", "label_mode": "A"}]), "punct_labels"),
        "outside_vocab": (_lines([*rows[:5], {**rows[5], "punct_labels": ["DASH"] * 4}]), "outside"),
        "not_utf8": (b"\xff\xfe\n", "UTF-8"),
    }
    text, needle = cases[variant]
    assert needle in _tune_with_train_text(mode_a, tmp_path, text, capsys)


def test_tune_train_file_missing_is_a_friendly_error(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, ids = _write_train(tmp_path, "A")
    out = tmp_path / "tau.json"
    assert tune_threshold.main(_train_argv(mode_a, out, tmp_path / "ghost.jsonl", ids)) == 1
    err = capsys.readouterr().err
    assert len(err.splitlines()) == 1 and "ghost.jsonl" in err and not out.exists()


def test_tune_never_overwrites_the_train_file(mode_a: Fixture, tmp_path: Path) -> None:
    train, ids = _write_train(tmp_path, "A")
    before = train.read_bytes()
    assert tune_threshold.main(_train_argv(mode_a, train, train, ids)) == 1
    assert train.read_bytes() == before


# ---------------------------------------------------------------- CLI predict

EXTREME_PRIORS = {"COMMA": 1e-6, "PERIOD": 0.9, "QUESTION": 0.9}
_ABSENT = object()


def _gamma_tau_file(path: Path, fx: Fixture, tau: Any, gamma: Any, priors: Any = _ABSENT) -> Path:
    data: dict[str, Any] = {"tau": tau, "grid": [], "model_sha256": _sha256(fx.model)}
    if gamma is not _ABSENT:
        data["gamma"] = gamma
    if priors is not _ABSENT:
        data["priors"] = priors
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _predict(fx: Fixture, tau_file: Path, tmp_path: Path, tag: str = "p") -> tuple[int, Path, Path]:
    out, man = tmp_path / f"{tag}.jsonl", tmp_path / f"{tag}.man.json"
    return tune_threshold.main(_predict_argv(fx, tau_file, out, man)), out, man


def test_predict_with_gamma_names_the_model_decodes_with_priors_and_passes_the_manifest(
    mode_a: Fixture, vocab: Vocab, tmp_path: Path
) -> None:
    tau_file = _gamma_tau_file(tmp_path / "tau.json", mode_a, 0.2, 1.0, EXTREME_PRIORS)
    code, out, man = _predict(mode_a, tau_file, tmp_path)
    assert code == 0

    predictions = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    model = CRFModel.load_model(mode_a.model)
    name = f"{model.model_name}_tau0.2_gamma1"
    assert all(p["model_name"] == name for p in predictions)
    assert _read(man)["model_name"] == name

    records = [{"sample_id": row["sample_id"], "tokens": row["tokens"]} for row in _rows("A", "gold")]
    marginals = model.predict_marginals(records)
    expected = [decode_with_threshold(m, 0.2, priors=EXTREME_PRIORS, gamma=1.0) for m in marginals]
    assert [p["predicted_labels"] for p in predictions] == expected
    assert expected != [decode_with_threshold(m, 0.2) for m in marginals]  # gamma really applied
    check_manifest(man, load_bundle(mode_a.gold, mode_a.ids, out, None, vocab), vocab)


def test_predict_names_a_fractional_gamma(mode_a: Fixture, tmp_path: Path) -> None:
    tau_file = _gamma_tau_file(tmp_path / "tau.json", mode_a, 0.35, 0.25, EXTREME_PRIORS)
    code, out, _ = _predict(mode_a, tau_file, tmp_path)
    assert code == 0
    names = {json.loads(line)["model_name"] for line in out.read_text(encoding="utf-8").splitlines()}
    assert len(names) == 1 and names.pop().endswith("_tau0.35_gamma0.25")


@pytest.mark.parametrize("gamma", [_ABSENT, 0, 0.0, -0.0])
def test_predict_without_gamma_is_identical_to_an_old_tau_file(
    mode_a: Fixture, tmp_path: Path, gamma: Any
) -> None:
    old = _gamma_tau_file(tmp_path / "old.json", mode_a, 0.35, _ABSENT)
    code, old_out, old_man = _predict(mode_a, old, tmp_path, "old")
    assert code == 0
    new = _gamma_tau_file(tmp_path / "new.json", mode_a, 0.35, gamma, {"COMMA": 0.0})  # unused at 0
    code, new_out, new_man = _predict(mode_a, new, tmp_path, "new")
    assert code == 0
    assert new_out.read_bytes() == old_out.read_bytes()
    assert new_man.read_bytes() == old_man.read_bytes()
    assert "gamma" not in new_out.read_text(encoding="utf-8")


@pytest.mark.parametrize("gamma", [-0.5, True, "0.5", None, math.nan, math.inf])
def test_predict_rejects_a_bad_gamma_in_the_tau_file(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str], gamma: Any
) -> None:
    tau_file = _gamma_tau_file(tmp_path / "tau.json", mode_a, 0.35, gamma, EXTREME_PRIORS)
    code, out, man = _predict(mode_a, tau_file, tmp_path)
    assert code == 1 and not out.exists() and not man.exists()
    err = capsys.readouterr().err
    assert err.startswith("error: ") and len(err.splitlines()) == 1 and "tau file" in err and "gamma" in err


@pytest.mark.parametrize(
    "priors", [_ABSENT, None, [], {"COMMA": 0.0}, {"COMMA": True}, {"COMMA": 0.4}],
    ids=["absent", "null", "list", "zero", "bool", "labels_missing"],
)
def test_predict_gamma_without_usable_priors_is_a_one_line_error(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str], priors: Any
) -> None:
    tau_file = _gamma_tau_file(tmp_path / "tau.json", mode_a, 0.35, 0.5, priors)
    code, out, man = _predict(mode_a, tau_file, tmp_path)
    assert code == 1 and not out.exists() and not man.exists()
    err = capsys.readouterr().err
    assert err.startswith("error: ") and len(err.splitlines()) == 1 and "prior" in err


def test_predict_still_checks_the_model_hash_when_gamma_is_set(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tau_file = _gamma_tau_file(tmp_path / "tau.json", mode_a, 0.35, 1.0, EXTREME_PRIORS)
    data = _read(tau_file)
    data["model_sha256"] = "0" * 64
    tau_file.write_text(json.dumps(data), encoding="utf-8")
    code, out, _ = _predict(mode_a, tau_file, tmp_path)
    assert code == 1 and not out.exists() and "different model" in capsys.readouterr().err
