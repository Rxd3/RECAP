"""verify() audits a finished HMM run on its own: it recounts scores, refits the winners and reproduces predictions.

The happy-path tests read the session's finished study; each refusal test works on a private copy and tampers with
exactly one thing. Every test names the wrong implementation it would catch.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from hmm_study_fixtures import (
    MODES,
    PRIMARY_IDS,
    PROCESSED,
    SPLITS_DIR,
    Study,
    copy_study,
    finished_study,
    patch_study,
    read_json,
    rewrite_json,
    sha256,
    write_text,
)

from punct_restore.data import load_label_sets
from punct_restore.models.hmm import HMMModel
from punct_restore.models.hmm_experiment_verify import verify

TEST_SPLITS = ("test", "test_amb")
FINAL_SPLITS = ("validation", *TEST_SPLITS)
SAMPLE_KEYS = {"sample_id", "position", "gold", "predicted", "context_tokens", "selection_rule"}
REPORT_KEYS = {"schema", "at_utc", "plan", "selection", "python", "modes", "runtime_checked", "result"}

EXISTS = "verification output already exists; use a new verification directory"
INCOMPLETE = "finalization is incomplete"
UNBOUND = "final results are not bound to frozen selection"
SIDECAR = "prediction sidecar provenance/hash mismatch"
REFIT_PREDICTIONS = "independent refit changed full-split predictions"
COUNTS = "scored-set counts disagree"
ID_ORDER = "predictions do not preserve exact required IDs/order/coverage"
SUPPORT = "learned state counts do not match the approved training subset"
SEQUENCES = "model trained on the wrong sequence count"
BYTES = "independent refit does not reproduce the frozen artifact bytes"
PRIVILEGED = "predictions depend on privileged fields"
RUNTIME_INCOMPLETE = "runtime measurements are incomplete"
RUNTIME_INVALID = "invalid runtime repetitions"
PLAN_MOVED = "selection does not match its frozen plan"
INPUT_HASH = "shared report input hash mismatch"


@pytest.fixture
def finished(tmp_path_factory: pytest.TempPathFactory) -> Study:
    """The session's finished study; tests only read it."""
    return finished_study(tmp_path_factory)


@pytest.fixture
def study(tmp_path_factory: pytest.TempPathFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Study:
    """A private copy of the finished study with the study code patched to it."""
    copy = copy_study(finished_study(tmp_path_factory), tmp_path)
    patch_study(monkeypatch, copy)
    return copy


@pytest.fixture
def out(study: Study) -> Path:
    """A new audit folder inside the run (verify refuses an existing one); it does not exist yet."""
    return study.run / "audit-2"


# ---- helpers ----


def _exact(message: str) -> str:
    return f"^{re.escape(message)}$"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_text(path, "".join(json.dumps(row) + "\n" for row in rows))


def _final(study: Study, mode: str, split: str) -> Path:
    return study.run / f"final/mode_{mode}/{split}"


def _refused(study: Study, out: Path, pattern: str, *, before_output: bool = False) -> None:
    """verify() raises a one-line ValueError matching `pattern` and writes no verification.json."""
    with pytest.raises(ValueError, match=pattern):
        verify(study.run, study.evaluator, out)
    assert not (out / "verification.json").exists()
    if before_output:
        assert not out.exists()


def _resync_sidecar(folder: Path) -> None:
    """Make the prediction sidecar cite the current predictions file, as a forger who also edited it would."""
    current = sha256(folder / "predictions.jsonl")
    rewrite_json(folder / "predictions.manifest.json", lambda m: {**m, "predictions_sha256": current})


def _first_label_changed(folder: Path) -> None:
    """Change the first predicted label of the first row to another valid mode A label."""
    rows = _rows(folder / "predictions.jsonl")
    labels = rows[0]["predicted_labels"]
    other = next(label for label in load_label_sets()["A"] if label != labels[0])
    rows[0] = {**rows[0], "predicted_labels": [other, *labels[1:]]}
    _write_rows(folder / "predictions.jsonl", rows)


def _rebind_release(study: Study, section: str, name: str, path: Path) -> None:
    """Make the release manifest cite a release file's current bytes, as a forger who also edited it would."""
    entry = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = study.root / PROCESSED / "release_manifest.json"
    rewrite_json(manifest, lambda m: {**m, section: {**m[section], name: entry}})


def _primary_rows(study: Study, mode: str, split: str) -> list[dict[str, Any]]:
    ids = set((study.root / SPLITS_DIR / f"primary_eval/{split}_mode_{mode}.ids").read_text("utf-8").split())
    rows = _rows(study.root / PROCESSED / f"mode_{mode}/{split}.jsonl")
    return [row for row in rows if row["sample_id"] in ids]


def _gold_labels(row: dict[str, Any], mode: str) -> list[str]:
    if mode == "C":
        return [f"{p}|{c}" for p, c in zip(row["punct_labels"], row["case_labels"], strict=True)]
    return row["punct_labels"]


def _expected_samples(study: Study, mode: str, split: str) -> list[dict[str, Any]]:
    """The first two errors per ordered (gold, predicted) pair of the primary block, counted from the raw files."""
    predicted = {r["sample_id"]: r["predicted_labels"] for r in _rows(_final(study, mode, split) / "predictions.jsonl")}
    masks: dict[str, set[int]] = {}
    if mode == "C":  # the primary block of mode C, joint/unmasked, skips the masked positions
        masked = _rows(study.root / PROCESSED / f"masks/case_joint/{split}.jsonl")
        masks = {row["sample_id"]: set(row["excluded_positions"]) for row in masked}
    samples, seen = [], Counter[tuple[str, str]]()
    for row in _primary_rows(study, mode, split):
        sid = row["sample_id"]
        for i, (gold, guess) in enumerate(zip(_gold_labels(row, mode), predicted[sid], strict=True)):
            if gold == guess or i in masks.get(sid, set()) or seen[gold, guess] >= 2:
                continue
            seen[gold, guess] += 1
            samples.append(
                {
                    "sample_id": sid, "position": i, "gold": gold, "predicted": guess,
                    "context_tokens": row["tokens"][max(0, i - 3) : i + 4],
                    "selection_rule": "first two occurrences per ordered confusion pair",
                }
            )
    return samples


# ---- A. the happy path ----


def test_report_names_its_inputs_and_passes(finished: Study) -> None:
    """A verifier that forgets to bind the report to the exact plan and selection files, or omits a field, fails."""
    report = read_json(finished.run / "verification/verification.json")
    assert set(report) == REPORT_KEYS
    assert report["schema"] == "recap-hmm-verification-v1"
    assert report["result"] == "PASS"
    assert report["runtime_checked"] is True
    assert report["python"] == sys.version.split()[0]
    assert datetime.fromisoformat(report["at_utc"]).utcoffset() == timedelta(0)
    assert report["plan"] == {"path": "artifacts/hmm/unit/plan.json", "sha256": sha256(finished.run / "plan.json")}
    assert report["selection"] == {
        "path": "artifacts/hmm/unit/selection.json",
        "sha256": sha256(finished.run / "selection.json"),
    }
    assert set(report["modes"]) == set(MODES)


@pytest.mark.parametrize("mode", MODES)
def test_every_candidate_is_checked_and_the_refit_matches_the_winner(finished: Study, mode: str) -> None:
    """Checking only the winner, or trusting the stored artifact without a refit, fails the 4 and the winner's hash."""
    report = read_json(finished.run / "verification/verification.json")["modes"][mode]
    winner = read_json(finished.run / "selection.json")["modes"][mode]
    assert report["candidates_independently_checked"] == 4
    assert report["independent_refit"]["result"] == "PASS"
    assert report["independent_refit"]["model_sha256"] == winner["model_sha256"]
    assert report["independent_refit"]["model_sha256"] == sha256(finished.run / winner["model"])


@pytest.mark.parametrize("mode", MODES)
def test_final_reports_cover_validation_and_both_test_sets_with_their_shortcut(finished: Study, mode: str) -> None:
    """A verifier that skips a split, the shortcut baseline or the primary-ID restriction (8 of 10) fails."""
    reports = read_json(finished.run / "verification/verification.json")["modes"][mode]["final_reports"]
    assert set(reports) == set(FINAL_SPLITS)
    for split, report in reports.items():
        rows = _primary_rows(finished, mode, split)
        assert (report["sequences"], report["tokens"]) == (PRIMARY_IDS, sum(len(r["tokens"]) for r in rows))
        assert report["shortcut"]["sequences"] == PRIMARY_IDS
        assert report["shortcut"]["tokens"] == report["tokens"]
        assert report["shortcut"]["sample_errors"] == []


@pytest.mark.parametrize("mode", MODES)
def test_validation_has_no_error_examples_and_test_has_the_primary_block_ones(finished: Study, mode: str) -> None:
    """Examples taken from another block (mode C casing, say), beyond two per pair, or from the 2 ignored IDs, fail."""
    reports = read_json(finished.run / "verification/verification.json")["modes"][mode]["final_reports"]
    assert reports["validation"]["sample_errors"] == []
    for split in TEST_SPLITS:
        samples = reports[split]["sample_errors"]
        assert samples, "the fixture's model makes errors on the test sets"
        assert all(set(sample) == SAMPLE_KEYS for sample in samples)
        assert samples == _expected_samples(finished, mode, split)


@pytest.mark.parametrize("mode", MODES)
def test_reproduced_models_are_the_winners_bytes(finished: Study, mode: str) -> None:
    """A refit that is only 'close' to the frozen artifact, or never written, fails the byte comparison."""
    winner = read_json(finished.run / "selection.json")["modes"][mode]
    reproduced = finished.run / f"verification/mode_{mode}.reproduced.json"
    assert reproduced.read_bytes() == (finished.run / winner["model"]).read_bytes()


def test_a_fresh_audit_passes_logs_one_line_per_mode_and_leaves_the_run_untouched(study: Study, out: Path) -> None:
    """A verifier that logs nothing or per candidate, writes verification.json elsewhere, or changes the run fails."""
    before = {p: sha256(p) for p in study.run.rglob("*") if p.is_file()}
    lines: list[str] = []
    verify(study.run, study.evaluator, out, log=lines.append)
    assert lines == [
        "Verified mode A: search, counts, probabilities, full refit/predictions, scores.",
        "Verified mode C: search, counts, probabilities, full refit/predictions, scores.",
        f"Complete HMM verification: {out / 'verification.json'}",
    ]
    report, earlier = read_json(out / "verification.json"), read_json(study.run / "verification/verification.json")
    assert report["result"] == "PASS"
    assert {**report, "at_utc": None} == {**earlier, "at_utc": None}  # the audit is deterministic
    assert {p.name for p in out.iterdir()} == {"verification.json", "mode_A.reproduced.json", "mode_C.reproduced.json"}
    assert {p: sha256(p) for p in before} == before
    assert {p for p in study.run.rglob("*") if p.is_file()} - set(before) == set(out.iterdir())


# ---- B. refusals before the audit writes anything ----


def test_an_existing_output_folder_is_refused_and_kept_as_it_is(study: Study) -> None:
    """A verifier that overwrites an earlier audit (or merges into the folder) fails."""
    existing = study.run / "verification"
    before = {p.name: sha256(p) for p in existing.iterdir()}
    with pytest.raises(ValueError, match=_exact(EXISTS)):
        verify(study.run, study.evaluator, existing)
    assert {p.name: sha256(p) for p in existing.iterdir()} == before


def test_a_run_without_results_is_refused_before_any_output(study: Study, out: Path) -> None:
    """A verifier that audits a run whose finalization never finished fails."""
    (study.run / "results.json").unlink()
    _refused(study, out, _exact(INCOMPLETE), before_output=True)


def test_results_bound_to_another_selection_are_refused_before_any_output(study: Study, out: Path) -> None:
    """A verifier that does not tie results.json to the frozen selection fails."""
    rewrite_json(study.run / "results.json", lambda r: {**r, "selection_sha256": "0" * 64})
    _refused(study, out, _exact(UNBOUND), before_output=True)


def test_a_moved_plan_is_refused_by_the_gate_before_any_output(study: Study, out: Path) -> None:
    """The frozen-selection gate runs first: a plan edited after selection stops verify before it creates anything."""
    plan = study.run / "plan.json"
    plan.write_text(plan.read_text(encoding="utf-8") + " ", encoding="utf-8")
    _refused(study, out, _exact(PLAN_MOVED), before_output=True)


@pytest.mark.parametrize(
    ("relative", "name"),
    [
        pytest.param("mode_A/train.jsonl", "mode_A/train.jsonl", id="train-file-read-by-the-tune-phase"),
        pytest.param("mode_C/test_amb.jsonl", "mode_C/test_amb.jsonl", id="test-file-read-by-the-finalize-phase"),
    ],
)
def test_a_changed_release_file_is_refused_before_any_output(
    study: Study, out: Path, relative: str, name: str
) -> None:
    """A verifier that authenticates only one phase's release files (train or held-out) fails one of the two cases."""
    path = study.root / PROCESSED / relative
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _refused(study, out, _exact(f"{name}: release checksum mismatch"), before_output=True)


# ---- B. refusals during the audit: sidecar, predictions, scores ----


def _sidecar_model(manifest: dict[str, Any]) -> dict[str, Any]:
    return {**manifest, "model_sha256": "0" * 64}


def _sidecar_selection(manifest: dict[str, Any]) -> dict[str, Any]:
    record = {**manifest["inputs"]["selection_record"], "sha256": "0" * 64}
    return {**manifest, "inputs": {**manifest["inputs"], "selection_record": record}}


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(None, id="predictions-file-edited"),
        pytest.param(_sidecar_model, id="sidecar-cites-another-model"),
        pytest.param(_sidecar_selection, id="sidecar-cites-another-selection-record"),
    ],
)
def test_a_sidecar_that_does_not_match_its_files_is_refused(
    study: Study, out: Path, change: Callable[[dict[str, Any]], dict[str, Any]] | None
) -> None:
    """A verifier that checks only one of the sidecar's three hashes fails two of these cases."""
    folder = _final(study, "A", "test")
    if change is None:
        _first_label_changed(folder)
    else:
        rewrite_json(folder / "predictions.manifest.json", change)
    _refused(study, out, _exact(SIDECAR))


def test_edited_predictions_with_a_forged_sidecar_hash_are_caught_by_the_refit(study: Study, out: Path) -> None:
    """A verifier that trusts predictions whose sidecar hash matches, without reproducing them, fails."""
    folder = _final(study, "A", "test")
    _first_label_changed(folder)
    _resync_sidecar(folder)
    _refused(study, out, _exact(REFIT_PREDICTIONS))


def test_reordered_predictions_with_a_forged_sidecar_hash_are_caught_by_the_refit(study: Study, out: Path) -> None:
    """The refit comparison runs before the report's ID/order check, so it is the one that fires for final files."""
    folder = _final(study, "C", "test")
    rows = _rows(folder / "predictions.jsonl")
    _write_rows(folder / "predictions.jsonl", [rows[1], rows[0], *rows[2:]])
    _resync_sidecar(folder)
    _refused(study, out, _exact(REFIT_PREDICTIONS))


def test_reordered_candidate_predictions_are_refused_by_the_id_order_check(study: Study, out: Path) -> None:
    """The search evidence is recounted too: a verifier that only audits the final splits misses this."""
    winner = read_json(study.run / "selection.json")["modes"]["C"]
    predictions = (study.run / winner["model"]).parent / "predictions.jsonl"
    rows = _rows(predictions)
    _write_rows(predictions, [rows[1], rows[0], *rows[2:]])
    _refused(study, out, _exact(ID_ORDER))


def test_a_raised_metric_is_caught_by_the_independent_recount(study: Study, out: Path) -> None:
    """A verifier that trusts the scorer's numbers instead of recounting them fails."""
    path = _final(study, "A", "validation") / "evaluation.json"
    before = read_json(path)["blocks"]["punctuation/all"]["macro_f1"]

    def raise_macro_f1(report: dict[str, Any]) -> dict[str, Any]:
        block = {**report["blocks"]["punctuation/all"], "macro_f1": before + 0.25}
        return {**report, "blocks": {**report["blocks"], "punctuation/all": block}}

    rewrite_json(path, raise_macro_f1)
    with pytest.raises(ValueError, match=r"^independent metric mismatch: ") as caught:
        verify(study.run, study.evaluator, out)
    tampered, recount = (float(x) for x in str(caught.value).split(": ", 1)[1].split(" != "))
    assert tampered == before + 0.25
    assert abs(recount - before) <= 5e-7  # the recount agrees with the stored value to the scorer's six decimals
    assert not (out / "verification.json").exists()


def test_a_wrong_scored_sequence_count_is_refused(study: Study, out: Path) -> None:
    """A verifier that recounts the metrics but not the report's scored-set counts fails."""
    path = _final(study, "A", "test") / "evaluation.json"
    rewrite_json(path, lambda r: {**r, "counts": {**r["counts"], "sequences": r["counts"]["sequences"] + 1}})
    _refused(study, out, _exact(COUNTS))


def test_a_shortcut_baseline_that_differs_from_its_report_is_refused(study: Study, out: Path) -> None:
    """The shortcut baseline is audited too: a verifier that skips its input hashes fails."""
    folder = _final(study, "A", "test")
    rows = _rows(folder / "shortcut.jsonl")
    flipped = {**rows[0], "predicted_labels": ["QUESTION"] * len(rows[0]["predicted_labels"])}
    _write_rows(folder / "shortcut.jsonl", [flipped, *rows[1:]])
    _refused(study, out, _exact(INPUT_HASH))


# ---- B. refusals during the audit: refit, release and runtime ----


def test_state_counts_that_disagree_with_the_release_stats_are_refused(study: Study, out: Path) -> None:
    """A verifier that never compares the learned counts with the approved training subset fails."""

    def one_more_o(stats: dict[str, Any]) -> dict[str, Any]:
        sets = stats["labels_default_and_primary"]
        mode_a = sets["train"]["A"]
        counts = {**mode_a["counts"], "O": mode_a["counts"]["O"] + 1}
        return {**stats, "labels_default_and_primary": {"train": {**sets["train"], "A": {"counts": counts}}}}

    rewrite_json(study.root / PROCESSED / "stats.json", one_more_o)
    _refused(study, out, _exact(SUPPORT))


def test_a_training_list_of_another_length_is_refused_as_the_wrong_sequence_count(study: Study, out: Path) -> None:
    """Without the count check the shorter list reaches the refit and fails later with the byte message instead."""
    ids = study.root / SPLITS_DIR / "default_train/mode_A.ids"
    ids.write_text("".join(f"{i}\n" for i in ids.read_text(encoding="utf-8").split()[:-1]), encoding="utf-8")
    _rebind_release(study, "split_files", "default_train/mode_A.ids", ids)
    _refused(study, out, _exact(SEQUENCES))


def test_a_refit_that_differs_from_the_frozen_artifact_is_refused(study: Study, out: Path) -> None:
    """The refit is compared byte for byte: a verifier that only checks it loads and sums to one fails."""
    train = study.root / PROCESSED / "mode_A/train.jsonl"
    rows = _rows(train)
    rows[0] = {**rows[0], "tokens": ["zzzzz", *rows[0]["tokens"][1:]]}
    _write_rows(train, rows)
    _rebind_release(study, "files", "mode_A/train.jsonl", train)
    _refused(study, out, _exact(BYTES))


def test_a_model_that_reads_gold_fields_is_refused(study: Study, out: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A verifier that never re-predicts with poisoned gold labels and metadata cannot see the leak."""
    honest = HMMModel.predict

    def leaky(self: HMMModel, records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        rows = list(records)
        predicted = honest(self, rows)
        if not any(row.get("punct_labels") == ["INVALID"] for row in rows):
            return predicted
        return [{**p, "predicted_labels": ["LEAK", *p["predicted_labels"][1:]]} for p in predicted]

    monkeypatch.setattr(HMMModel, "predict", leaky)
    _refused(study, out, _exact(PRIVILEGED))


def _without_mode_c(runtime: dict[str, Any]) -> dict[str, Any]:
    return {**runtime, "modes": {m: v for m, v in runtime["modes"].items() if m != "C"}}


def _with_seconds(mode: str, seconds: list[float]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def change(runtime: dict[str, Any]) -> dict[str, Any]:
        modes = {**runtime["modes"], mode: {**runtime["modes"][mode], "prediction_seconds": seconds}}
        return {**runtime, "modes": modes}

    return change


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(_without_mode_c, RUNTIME_INCOMPLETE, id="mode-C-missing"),
        pytest.param(_with_seconds("A", [0.001]), RUNTIME_INVALID, id="A-one-repetition"),
        pytest.param(_with_seconds("C", [0.001, 0.0]), RUNTIME_INVALID, id="C-zero-time"),
        pytest.param(_with_seconds("A", [-0.5, 0.001]), RUNTIME_INVALID, id="A-negative-time"),
    ],
)
def test_runtime_measurements_that_are_incomplete_or_invalid_are_refused(
    study: Study, out: Path, change: Callable[[dict[str, Any]], dict[str, Any]], message: str
) -> None:
    """The runtime check covers every configured mode (not just the first) and no verification.json is written."""
    rewrite_json(study.run / "runtime.json", change)
    _refused(study, out, _exact(message))
