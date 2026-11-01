"""Threshold decoding over CRF marginals (team extension; tau and gamma are tuned on validation only).

Plain Viterbi decoding picks the most probable label sequence. Here every token is decoded
on its own from its posterior marginals: a token gets punctuation only when the total
probability of "some punctuation" (1 - P(O)) is strictly greater than ``tau``. A small tau
favours recall, a large tau favours precision.

Once the gate fires, the label is the best non-O one. With ``gamma > 0`` each non-O label is
first divided by ``prior ** gamma`` (its training frequency), so rare classes are not drowned
out by common ones; ``gamma == 0`` is the plain argmax.

The tuning that `recap tune-threshold` runs is here too (the command keeps the arguments, the files it opens and
the printing): the grid of (tau, gamma) scored on decoded predictions, the label priors streamed from the training
file, and the tau file that ``predict`` reads back. Nothing here prints.
"""

from __future__ import annotations

import math
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

from punct_restore.file_io import iter_jsonl, read_object, require_file
from punct_restore.hashing import sha256_file
from punct_restore.prediction_io import parse_sample_record, read_ids, write_predictions_jsonl

NO_PUNCT = "O"


def is_o_label(label: str) -> bool:
    """True for the no-punctuation label of every mode: ``O`` or a joint ``O|<CASE>`` (never for a non-string)."""
    return isinstance(label, str) and (label == NO_PUNCT or label.startswith(f"{NO_PUNCT}|"))


def _is_finite_number(value: object) -> TypeGuard[float]:
    """True for a finite int or float (not a bool); an int too big for a float is not finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def check_tau(tau: object) -> None:
    """Raise ValueError unless ``tau`` is a finite number (not a bool) in [0, 1]."""
    if not (_is_finite_number(tau) and 0 <= tau <= 1):  # _is_finite_number also refuses ints too big for a float
        raise ValueError(f"tau must be a number between 0 and 1, got {tau!r}")


def check_gamma(gamma: object) -> None:
    """Raise ValueError unless ``gamma`` is a finite number (not a bool) that is >= 0."""
    if not (_is_finite_number(gamma) and gamma >= 0):
        raise ValueError(f"gamma must be a finite number >= 0, got {gamma!r}")


def _as_priors(priors: object) -> Mapping[str, float]:
    if isinstance(priors, Mapping):
        return priors
    raise ValueError("priors must be a mapping of label to a finite number > 0")


def _valid_prior(priors: Mapping[str, float], label: str) -> float:
    prior = priors.get(label)
    if not (_is_finite_number(prior) and prior > 0):
        raise ValueError(f"gamma > 0 needs a finite prior > 0 for label {label!r}")
    return prior


def check_priors(priors: object) -> None:
    """Raise ValueError unless ``priors`` maps every label to a finite number > 0."""
    mapping = _as_priors(priors)
    for label in mapping:
        _valid_prior(mapping, label)


def _label_shifts(
    marginals: Sequence[Mapping[str, float]], priors: object, gamma: float
) -> dict[str, float]:
    """``gamma * log(prior)`` of every non-O label that appears in ``marginals``.

    Raises:
        ValueError: ``priors`` is not a mapping, or lacks a finite prior > 0 for an appearing label.
    """
    mapping = _as_priors(priors)
    labels = {label for probs in marginals for label in probs if not is_o_label(label)}
    return {label: gamma * math.log(_valid_prior(mapping, label)) for label in labels}


def _adjusted(prob: float, shift: float) -> float:
    """``log(P / prior ** gamma)``; log space, so a huge gamma cannot overflow or divide by zero."""
    return -math.inf if prob <= 0 else math.log(prob) - shift


def _decode_token(
    probs: Mapping[str, float], index: int, tau: float, shifts: Mapping[str, float] | None
) -> str:
    if not probs:
        raise ValueError(f"token {index} has no label probabilities")
    for label in probs:
        if not isinstance(label, str):
            raise ValueError(f"token {index} has a label that is not text: {label!r}")
    o_labels = [label for label in probs if is_o_label(label)]
    if not o_labels:
        raise ValueError(f"token {index} has no O label to compare against")
    punct_labels = [label for label in probs if not is_o_label(label)]
    p_o = sum(probs[label] for label in o_labels)
    if punct_labels and 1 - p_o > tau:
        if shifts is None:
            return max(punct_labels, key=probs.__getitem__)
        return max(punct_labels, key=lambda label: _adjusted(probs[label], shifts[label]))
    return max(o_labels, key=probs.__getitem__)


def decode_with_threshold(
    marginals: Sequence[Mapping[str, float]],
    tau: float,
    *,
    priors: Mapping[str, float] | None = None,
    gamma: float = 0.0,
) -> list[str]:
    """One label per token: the best punctuation label if ``1 - P(O) > tau``, else the best O label.

    ``P(O)`` sums every O label (``O`` or ``O|*``), so mode C casing never splits the O mass.
    With ``gamma > 0`` the punctuation label is ``argmax P(c|x) / priors[c] ** gamma`` over the
    token's non-O labels; the gate is unchanged. With ``gamma == 0`` (the default) ``priors`` is
    ignored and the output is the plain argmax. Ties go to the first label in the mapping's
    order. Returns a new list.

    Raises:
        ValueError: ``tau`` is not a finite number in [0, 1], ``gamma`` is not a finite number
            >= 0, ``gamma > 0`` without a priors mapping that has a finite prior > 0 for every
            non-O label that appears, or a token has no probabilities or no O label.
    """
    check_tau(tau)
    check_gamma(gamma)
    shifts = _label_shifts(marginals, priors, gamma) if gamma > 0 else None
    return [_decode_token(probs, index, tau, shifts) for index, probs in enumerate(marginals)]


def _label_list(record: Mapping[str, Any], key: str, where: str = "a record") -> list[str]:
    labels = record.get(key)
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError(f"{where}: {key} must be a list of strings")
    return labels


def _gold_labels(record: Mapping[str, Any], joint: bool) -> list[str]:
    punct = _label_list(record, "punct_labels")
    if not joint:
        return punct
    case = _label_list(record, "case_labels")
    if len(case) != len(punct):
        raise ValueError("a record's case_labels and punct_labels differ in length")
    return [f"{p}|{c}" for p, c in zip(punct, case, strict=True)]


def _count_labels(records: Iterable[Mapping[str, Any]], joint: bool) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(_gold_labels(record, joint))
    return counts


def _priors_of(counts: Counter[str], labels: Sequence[str], source: str = "") -> dict[str, float]:
    """The add-one priors of the non-O ``labels``; ``source`` (a file name) starts the message of a bad count."""
    outside = sorted(counts.keys() - set(labels))
    if outside:
        raise ValueError(f"{source + ': ' if source else ''}gold label {outside[0]!r} is outside the mode's label set")
    denominator = sum(counts.values()) + len(labels)
    return {label: (counts[label] + 1) / denominator for label in labels if not is_o_label(label)}


def label_priors(records: Iterable[Mapping[str, Any]], labels: Sequence[str]) -> dict[str, float]:
    """Add-one-smoothed gold frequency of every non-O label: ``(count + 1) / (tokens + K)``.

    ``K = len(labels)`` is the mode's label count, so no prior is 0. Counts come from the
    records' ``punct_labels``; when ``labels`` are joint (``PUNCT|CASE``, mode C) they come from
    the ``f"{punct}|{case}"`` pairs instead. ``records`` may be a one-shot iterable.

    Raises:
        ValueError: a record lacks its label lists, they differ in length, or a gold label is
            outside ``labels``.
    """
    return _priors_of(_count_labels(records, any("|" in label for label in labels)), labels)


# --- the (tau, gamma) grid -------------------------------------------------------------------------------------------

GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)  # searched only when the training file gives the priors


def pick_tau_gamma(grid_scores: Sequence[tuple[float, float, float]]) -> tuple[float, float]:
    """``(tau, gamma)`` with the highest macro F1 among ``(tau, gamma, f1)`` rows.

    Ties go to the smallest gamma, then to the tau nearest 0.5, then to the lower tau.
    """
    if not grid_scores:
        raise ValueError("cannot pick a tau from an empty grid")
    best = max(score for _, _, score in grid_scores)
    tied = [(tau, gamma) for tau, gamma, score in grid_scores if score == best]
    return min(tied, key=lambda pair: (pair[1], round(abs(pair[0] - 0.5), 9), pair[0]))


def tau_model_name(base_name: str, tau: float, gamma: float = 0.0) -> str:
    """The model name of predictions decoded at ``tau``; ``gamma`` is named only when above 0."""
    name = f"{base_name}_tau{tau:g}"
    return f"{name}_gamma{gamma:g}" if gamma > 0 else name


@dataclass(frozen=True)
class Marginals:
    """A model's per-token marginals for some sequences, the name and mode of its predictions, the label priors."""

    base_name: str
    mode: str
    sample_ids: Sequence[str]
    marginals: Sequence[Sequence[Mapping[str, float]]]
    priors: Mapping[str, float] | None = None


def decode_predictions(source: Marginals, tau: float, *, gamma: float = 0.0) -> list[dict[str, Any]]:
    """Prediction records (model contract) for every sequence, decoded at ``tau`` and ``gamma``."""
    name = tau_model_name(source.base_name, tau, gamma)
    return [
        {"sample_id": sample_id, "model_name": name, "label_mode": source.mode,
         "predicted_labels": decode_with_threshold(sequence, tau, priors=source.priors, gamma=gamma)}
        for sample_id, sequence in zip(source.sample_ids, source.marginals, strict=True)
    ]


def score_grid(
    source: Marginals,
    grid: Sequence[float],
    score_file: Callable[[Path], float],
    work_dir: Path,
    *,
    gammas: Sequence[float] = (0.0,),
) -> list[tuple[float, float, float]]:
    """``(tau, gamma, score_file(predictions))`` of the decoded predictions for every gamma and tau.

    The predictions of each pair are written to a file under ``work_dir`` (made if missing) for ``score_file``;
    the file and its temporary folder are gone when this returns.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    scores: list[tuple[float, float, float]] = []
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        predictions_path = Path(tmp) / "predictions.jsonl"
        for gamma in gammas:
            for tau in grid:
                write_predictions_jsonl(decode_predictions(source, tau, gamma=gamma), predictions_path)
                scores.append((tau, gamma, score_file(predictions_path)))
    return scores


def tau_result(
    best: tuple[float, float],
    scores: Sequence[tuple[float, float, float]],
    priors: Mapping[str, float] | None,
    model_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """The content of a tau file: the pick, the priors (when searched), the whole grid, the model, the provenance."""
    result: dict[str, Any] = {"tau": best[0], "gamma": best[1]}
    if priors is not None:
        result["priors"] = dict(priors)
    result["grid"] = [{"tau": tau, "gamma": gamma, "macro_f1": f1} for tau, gamma, f1 in scores]
    result["model_sha256"] = model_sha256
    return {**result, **provenance}


# --- the label priors of the training file ---------------------------------------------------------------------------


def _check_train_labels(record: Mapping[str, Any], mode: str, where: str) -> None:
    """Raise ValueError (leading with ``where``, ``<file> line N``) unless the labels the priors read fit the tokens."""
    tokens = record.get("tokens")
    for key in ("punct_labels", "case_labels") if mode == "C" else ("punct_labels",):
        labels = _label_list(record, key, where)
        if not isinstance(tokens, list) or len(labels) != len(tokens):
            raise ValueError(f"{where}: {key} must hold one label per token")


def stream_train_labels(train_path: Path, ids: Sequence[str], mode: str) -> Iterator[dict[str, Any]]:
    """Yield the gold label lists of the listed records, one line at a time; all IDs must be found."""
    wanted, found = set(ids), set()
    for number, record in iter_jsonl(train_path, parse_sample_record):
        if record["sample_id"] not in wanted:
            continue
        where = f"{train_path.name} line {number}"
        if record["sample_id"] in found:
            raise ValueError(f"{where}: a listed sample_id appears twice")
        if record.get("label_mode") != mode:
            raise ValueError(f"{where}: a listed record is not label mode {mode}")
        found.add(record["sample_id"])
        _check_train_labels(record, mode, where)
        yield {key: record.get(key) for key in ("punct_labels", "case_labels")}
    if len(found) < len(wanted):
        raise ValueError(f"{train_path.name}: {len(wanted) - len(found)} listed ID(s) are missing")


def train_priors(train: Path, train_ids: Path, mode: str, vocabulary: Sequence[str]) -> dict[str, float]:
    """Add-one label priors of ``mode`` (over ``vocabulary``) from the training records that ``train_ids`` lists."""
    require_file(train, "training file")
    ids = read_ids(train_ids)
    counts = _count_labels(stream_train_labels(train, ids, mode), joint=any("|" in label for label in vocabulary))
    return _priors_of(counts, vocabulary, train.name)


# --- the tau file that predict reads ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class TauSettings:
    """What ``predict`` needs from a tau file; ``priors`` is None when ``gamma`` is 0."""

    tau: float
    gamma: float
    priors: dict[str, float] | None


def _gamma_settings(data: Mapping[str, Any]) -> tuple[float, dict[str, float] | None]:
    """Gamma (0 when the file predates it) and, when gamma > 0, the validated priors."""
    gamma = data.get("gamma", 0.0)
    check_gamma(gamma)
    gamma = float(gamma) + 0.0  # + 0.0: a gamma of -0.0 is a plain gamma of 0
    if gamma == 0:
        return 0.0, None
    priors = data.get("priors")
    check_priors(priors)
    return gamma, dict(_as_priors(priors))


def read_tau_file(path: Path, model_path: Path) -> TauSettings:
    """Tau, gamma and priors from a ``tune`` result, provided it was tuned for exactly this model file.

    Raises:
        OSError: ``<name>: tau file not found`` (``FileNotFoundError``), a folder, or a file that cannot be read.
        ValueError: One line that starts with the file name: not UTF-8, not JSON, not a JSON object, a missing or
            wrong ``tau``, ``gamma``, ``priors`` or ``model_sha256``, or ``tuned for a different model``.
    """
    require_file(path, "tau file")
    data = read_object(path, {"tau": object, "model_sha256": str})
    try:
        check_tau(data["tau"])
        gamma, priors = _gamma_settings(data)
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from exc
    if data["model_sha256"] != sha256_file(model_path):
        raise ValueError(f"{path.name}: tuned for a different model")
    # + 0.0: a tau of -0.0 must not name the model "_tau-0"
    return TauSettings(float(data["tau"]) + 0.0, gamma, priors)
