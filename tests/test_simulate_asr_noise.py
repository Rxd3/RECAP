"""Tests for punct_restore.cli.simulate_asr_noise (the CLI) on a tiny synthetic data root;
the edit model is in test_asr_noise.py."""

from __future__ import annotations

import argparse
import errno
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from punct_restore.data.asr_noise import Edits
from punct_restore.evaluation import load_scored_gold, load_vocab

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "simulate_asr_noise.py"  # the stub: subprocess runs
CLI_MODULE = ROOT / "src" / "punct_restore" / "cli" / "simulate_asr_noise.py"  # the code: in-process loads
MODULE = ROOT / "src" / "punct_restore" / "data" / "asr_noise.py"
RELEASE = "sponspeech-m1-v2"
MODES = ("A", "B", "C")
WORDS = ["the", "cat", "is", "on", "a", "mat", "their", "dog", "and", "you're",
         "too", "fast", "then", "it", "ran", "to", "its", "home", "today", "now"]
SAMPLES = 30
LISTED_C = 28  # mode C lists fewer samples than A and B, as in the release
DATA = {m: Path("processed", RELEASE, f"mode_{m}", "validation.jsonl") for m in MODES}
MASKS = Path("processed", RELEASE, "masks", "case_joint", "validation.jsonl")
IDS = {m: Path("splits", RELEASE, "primary_eval", f"validation_mode_{m}.ids") for m in MODES}


@pytest.fixture(scope="module")
def sim() -> Iterator[ModuleType]:
    name = "simulate_asr_noise_under_test"
    spec = importlib.util.spec_from_file_location(name, CLI_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses looks its module up here
    spec.loader.exec_module(module)
    yield module
    del sys.modules[name]


def make_sample(index: int, words: list[str] | None = None) -> dict[str, Any]:
    tokens = words or WORDS[index % len(WORDS):] + WORDS[:index % len(WORDS)]
    n = len(tokens)
    punct_a = ["O"] * n
    if n > 12:
        punct_a[4] = punct_a[12] = "COMMA"
    punct_a[-1] = "QUESTION" if index % 5 == 0 else "PERIOD"
    punct_b = list(punct_a)
    if index % 3 == 0 and n > 12:
        punct_b[12] = "DASH"
    if index % 7 == 0:
        punct_b[-1] = "EXCLAMATION"
    case = ["LOWER"] * n
    case[0], case[5 % n] = "FIRST_UPPER", "ALL_UPPER"
    return {"id": f"s{index:02d}", "tokens": tokens, "A": punct_a, "B": punct_b, "case": case}


def row(sample: dict[str, Any], mode: str) -> dict[str, Any]:
    n = len(sample["tokens"])
    return {"sample_id": sample["id"], "label_mode": mode, "schema_version": 1, "source": "fx",
            "split": "validation", "tokens": sample["tokens"],
            "punct_labels": sample["B" if mode == "B" else "A"], "case_labels": sample["case"],
            "metadata": {"group_id": "g", "policy_version": "m1-v2", "condition": "clean",
                         "original_text": "kept as is", "offsets": [[k * 10, k * 10 + 5] for k in range(n)]}}


def write_lines(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows).encode())


def write_root(root: Path, samples: list[dict[str, Any]], listed_c: int) -> Path:
    """Data root with all three modes, ID lists (descending, unlike the file order) and masks.

    The files hold two extra records (x00, x01) that no list names; masks cover every mode C record.
    """
    extras = [make_sample(i + 100) | {"id": f"x{i:02d}"} for i in range(2)]
    everything = extras + samples
    for mode in MODES:
        write_lines(root / DATA[mode], [row(s, mode) for s in everything])
        names = [s["id"] for s in (samples[:listed_c] if mode == "C" else samples)]
        (root / IDS[mode]).parent.mkdir(parents=True, exist_ok=True)
        (root / IDS[mode]).write_bytes(("\n".join(sorted(names, reverse=True)) + "\n").encode())
    masks = [{"sample_id": s["id"], "excluded_positions": [3, 7, 15] if k % 2 == 0 else [],
              "reason": "case_mixed_approximation" if k % 2 == 0 else None}
             for k, s in enumerate(everything)]
    write_lines(root / MASKS, masks)
    return root


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return write_root(tmp_path / "data", [make_sample(i) for i in range(SAMPLES)], LISTED_C)


def argv_for(root: Path, out: Path, wer: float = 0.10, *extra: str, seed: int = 2026) -> list[str]:
    return ["--data-root", str(root), "--split", "validation", "--wer", str(wer),
            "--seed", str(seed), "--out", str(out), *extra]


def run(sim: ModuleType, capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str, str]:
    code = sim.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def assert_one_error_line(err: str) -> None:
    assert err.startswith("error: ") and err.endswith("\n") and err.count("\n") == 1, repr(err)
    assert "Traceback" not in err and "\r" not in err, repr(err)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def tree(path: Path) -> dict[str, bytes]:
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


def produce(sim: ModuleType, capsys: pytest.CaptureFixture[str], root: Path, out: Path, wer: float = 0.10,
            seed: int = 2026) -> dict[str, Any]:
    code, _, err = run(sim, capsys, argv_for(root, out, wer, seed=seed))
    assert code == 0, err
    return json.loads((out / "summary.json").read_text(encoding="utf-8"))


def listed_input(root: Path, mode: str) -> list[dict[str, Any]]:
    listed = set((root / IDS[mode]).read_text().split())
    return [r for r in read_rows(root / DATA[mode]) if r["sample_id"] in listed]


def distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


@pytest.mark.parametrize("mode", MODES)
def test_outputs_validate_with_the_evaluator(sim, capsys, root, tmp_path, mode):
    out = tmp_path / "out"
    produce(sim, capsys, root, out)
    gold = out / f"mode_{mode}" / "validation.jsonl"
    ids = out / f"validation_mode_{mode}.ids"
    found_mode, records = load_scored_gold(gold, ids, load_vocab())
    assert found_mode == mode and len(records) == len(listed_input(root, mode))
    for rec in read_rows(gold):
        n = len(rec["tokens"])
        assert n >= 1 and len(rec["punct_labels"]) == n == len(rec["case_labels"])
        assert len(rec["metadata"]["offsets"]) == n


@pytest.mark.parametrize("mode", MODES)
def test_listed_records_only_in_input_order_with_same_schema(sim, capsys, root, tmp_path, mode):
    out = tmp_path / "out"
    produce(sim, capsys, root, out)
    before, after = listed_input(root, mode), read_rows(out / f"mode_{mode}" / "validation.jsonl")
    assert [r["sample_id"] for r in after] == [r["sample_id"] for r in before]
    assert all(
        list(a) == list(b) and list(a["metadata"]) == list(b["metadata"]) for a, b in zip(after, before, strict=False)
    )
    assert all(a["metadata"]["original_text"] == "kept as is" for a in after)
    assert b"\r" not in (out / f"mode_{mode}" / "validation.jsonl").read_bytes()
    assert (out / f"validation_mode_{mode}.ids").read_bytes() == (root / IDS[mode]).read_bytes()


def test_tokens_and_o_pattern_agree_across_modes(sim, capsys, root, tmp_path):
    out = tmp_path / "out"
    produce(sim, capsys, root, out)
    by_mode = {m: {r["sample_id"]: r for r in read_rows(out / f"mode_{m}" / "validation.jsonl")} for m in MODES}
    assert by_mode["A"].keys() == by_mode["B"].keys() and by_mode["C"].keys() <= by_mode["A"].keys()
    for sid, c_rec in by_mode["C"].items():
        a_rec, b_rec = by_mode["A"][sid], by_mode["B"][sid]
        assert a_rec["tokens"] == b_rec["tokens"] == c_rec["tokens"]
        assert a_rec["case_labels"] == b_rec["case_labels"] == c_rec["case_labels"]
        pattern = [p == "O" for p in a_rec["punct_labels"]]
        assert pattern == [p == "O" for p in b_rec["punct_labels"]] == [p == "O" for p in c_rec["punct_labels"]]
        assert a_rec["punct_labels"] == c_rec["punct_labels"]


@pytest.mark.parametrize("wer", [0.05, 0.10, 0.15])
def test_achieved_cost_equals_the_rounded_budget(sim, capsys, root, tmp_path, wer):
    out = tmp_path / "out"
    summary = produce(sim, capsys, root, out, wer)
    total = SAMPLES * len(WORDS)
    edits = summary["edits"]
    assert summary["tokens_before"] == total and summary["budget"] == round(wer * total)
    assert edits["deletion"] + edits["substitution"] + 2 * edits["swap"] == summary["budget"]
    assert summary["edit_cost"] == summary["budget"] and summary["achieved_wer"] == summary["budget"] / total
    assert summary["wer"] == wer and summary["seed"] == 2026
    assert summary["tokens_after"] == total - edits["deletion"]
    before, after = listed_input(root, "A"), read_rows(out / "mode_A" / "validation.jsonl")
    assert sum(len(r["tokens"]) for r in after) == summary["tokens_after"]
    assert sum(distance(b["tokens"], a["tokens"]) for a, b in zip(after, before, strict=False)) <= summary["edit_cost"]


def test_not_enough_candidates_exits_one_with_one_line(sim, capsys, tmp_path):
    root = write_root(tmp_path / "data", [make_sample(i, ["cat", "dog"]) for i in range(10)], 10)
    out = tmp_path / "out"
    code, stdout, err = run(sim, capsys, argv_for(root, out, 0.10))
    assert code == 1 and stdout == ""
    assert_one_error_line(err)
    assert "not enough editable tokens for WER 0.1" in err
    assert not out.exists() and [p.name for p in tmp_path.iterdir()] == ["data"]


def test_same_seed_is_byte_identical_and_other_seed_differs(sim, capsys, root, tmp_path):
    produce(sim, capsys, root, tmp_path / "one")
    produce(sim, capsys, root, tmp_path / "two")
    produce(sim, capsys, root, tmp_path / "other", seed=7)
    assert tree(tmp_path / "one") == tree(tmp_path / "two")
    assert tree(tmp_path / "one")["mode_A/validation.jsonl"] != tree(tmp_path / "other")["mode_A/validation.jsonl"]


def test_summary_reports_the_cost_share_of_each_edit_type(sim, capsys, root, tmp_path):
    summary = produce(sim, capsys, root, tmp_path / "out")  # budget 60 = 15 rounds
    assert summary["edits"] == {"deletion": 15, "substitution": 15, "swap": 15}
    assert summary["cost_share"] == {"deletion": 0.25, "substitution": 0.25, "swap": 0.5}


def test_a_budget_that_rounds_to_zero_gives_zero_shares(sim, capsys, root, tmp_path):
    summary = produce(sim, capsys, root, tmp_path / "out", wer=0.0001)  # round(0.0001 * 600) = 0
    assert summary["edit_cost"] == 0 and summary["edits"] == {"deletion": 0, "substitution": 0, "swap": 0}
    assert summary["cost_share"] == {"deletion": 0.0, "substitution": 0.0, "swap": 0.0}


def test_written_masks_follow_the_tokens(sim, capsys, root, tmp_path):
    out = tmp_path / "out"
    produce(sim, capsys, root, out)
    before = {r["sample_id"]: r for r in listed_input(root, "C")}
    after = {r["sample_id"]: r for r in read_rows(out / "mode_C" / "validation.jsonl")}
    masks = read_rows(out / "masks" / "validation.jsonl")
    assert [m["sample_id"] for m in masks] == list(after)
    for m in masks:
        sid = m["sample_id"]
        old, new = before[sid]["metadata"]["offsets"], after[sid]["metadata"]["offsets"]
        kept = [old[p] for p in ([3, 7, 15] if int(sid[1:]) % 2 == 0 else []) if old[p] in new]
        assert [new[p] for p in m["excluded_positions"]] == sorted(kept, key=new.index)
        assert m["excluded_positions"] == sorted(m["excluded_positions"])


def test_existing_out_is_refused_and_left_alone(sim, capsys, root, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    code, _, err = run(sim, capsys, argv_for(root, out))
    assert code == 1 and "already exists" in err
    assert_one_error_line(err)
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_a_write_failure_leaves_no_out_and_no_temp_folder(sim, capsys, root, tmp_path, monkeypatch):
    def fail(path: Path, rows: Any) -> None:
        raise OSError(errno.ENOSPC, "No space left on device", str(path))
    monkeypatch.setattr(sim, "write_jsonl", fail)
    out = tmp_path / "out"
    code, _, err = run(sim, capsys, argv_for(root, out))
    assert code == 1 and "No space left on device" in err and str(tmp_path) not in err
    assert_one_error_line(err)
    assert [p.name for p in tmp_path.iterdir()] == ["data"]


def test_masks_override_and_missing_default(sim, capsys, root, tmp_path):
    moved = tmp_path / "elsewhere" / "m.jsonl"
    moved.parent.mkdir()
    (root / MASKS).rename(moved)
    code, _, err = run(sim, capsys, argv_for(root, tmp_path / "fail"))
    assert code == 1 and "validation.jsonl" in err and not (tmp_path / "fail").exists()
    assert_one_error_line(err)
    code, _, err = run(sim, capsys, argv_for(root, tmp_path / "ok", 0.10, "--masks", str(moved)))
    assert code == 0, err
    assert (tmp_path / "ok" / "masks" / "validation.jsonl").is_file()


def test_wer_one_half_is_accepted(sim, capsys, root, tmp_path):
    summary = produce(sim, capsys, root, tmp_path / "out", wer=0.5)  # budget 300 of 600 tokens
    assert summary["wer"] == 0.5 and summary["budget"] == summary["edit_cost"] == 300


def test_out_in_a_missing_folder_is_refused(sim, capsys, root, tmp_path):
    code, stdout, err = run(sim, capsys, argv_for(root, tmp_path / "nope" / "out"))
    assert code == 1 and stdout == ""
    assert_one_error_line(err)
    assert "out: the folder to create it in does not exist" in err and str(tmp_path) not in err
    assert [p.name for p in tmp_path.iterdir()] == ["data"]


def test_masks_given_as_a_folder_says_so_whatever_the_platform_errno(sim, capsys, root, tmp_path):
    folder = tmp_path / "maskdir"  # Windows reports EACCES for this, POSIX EISDIR
    folder.mkdir()
    code, stdout, err = run(sim, capsys, argv_for(root, tmp_path / "out", 0.10, "--masks", str(folder)))
    assert code == 1 and stdout == ""
    assert_one_error_line(err)
    assert err == f"error: {tmp_path.name}/maskdir: is a folder, not a file\n"
    assert [p.name for p in tmp_path.iterdir()] == ["data", "maskdir"]


def test_display_name_is_the_parent_and_file_without_an_empty_parent(sim):
    assert sim._display_name(Path("rev-b2", "masks.jsonl")) == "rev-b2/masks.jsonl"
    assert sim._display_name(Path("/rev-b2")) == "rev-b2"  # the parent is the root: no leading slash
    assert sim._display_name(Path("masks.jsonl")) == "masks.jsonl"


def test_masks_follow_the_mode_c_input_whatever_the_order_of_the_inputs(sim, capsys, root, tmp_path, monkeypatch):
    load_inputs = sim.load_inputs
    monkeypatch.setattr(sim, "load_inputs", lambda args: load_inputs(args)[::-1])  # C, B, A
    out = tmp_path / "out"
    produce(sim, capsys, root, out)
    masks = read_rows(out / "masks" / "validation.jsonl")
    assert len(masks) == LISTED_C
    assert [m["sample_id"] for m in masks] == [r["sample_id"] for r in read_rows(out / "mode_C" / "validation.jsonl")]


def test_a_data_file_that_starts_with_a_byte_order_mark_reads_like_one_without(sim, capsys, root, tmp_path):
    produce(sim, capsys, root, tmp_path / "plain")
    path = root / DATA["B"]
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    produce(sim, capsys, root, tmp_path / "bom")
    assert tree(tmp_path / "bom") == tree(tmp_path / "plain")


def test_blank_lines_before_and_inside_a_data_file_are_counted_in_the_line_number(sim, capsys, root, tmp_path):
    path = root / DATA["A"]
    path.write_bytes(b"\n\n" + path.read_bytes() + b"\n{not json\n")
    code, _, err = run(sim, capsys, argv_for(root, tmp_path / "out"))
    rows = len(path.read_bytes().splitlines())
    assert code == 1 and err == f"error: mode_A/validation.jsonl line {rows}: not valid JSON\n"


def test_every_data_and_mask_file_is_read_through_the_shared_jsonl_reader(sim, capsys, root, tmp_path, monkeypatch):
    read: list[str] = []
    shared_reader = sim.iter_jsonl

    def recording_reader(path, parse):
        read.append(Path(path).name)
        return shared_reader(path, parse)

    monkeypatch.setattr(sim, "iter_jsonl", recording_reader)
    produce(sim, capsys, root, tmp_path / "out")
    assert read and set(read) == {"validation.jsonl"}  # the three data files and the masks all carry that name


def test_summary_cost_is_computed_from_the_cost_table(sim, monkeypatch):
    monkeypatch.setattr(sim, "COST", {**sim.COST, "swap": 3})
    args = argparse.Namespace(release="r", split="validation", wer=0.5, seed=1, edit_model="v1")
    plan = {"s": Edits(swapped=frozenset({0}))}
    summary = sim.summarize(args, [], {"s": ["a", "b", "c", "d"]}, plan)
    assert summary["edit_cost"] == 3 and summary["cost_share"]["swap"] == 1.0


def test_summary_of_edit_model_v1_has_neither_the_model_nor_a_vocabulary(sim):
    args = argparse.Namespace(release="r", split="validation", wer=0.5, seed=1, edit_model="v1")
    plan = {"s": Edits(deleted=frozenset({0}))}
    summary = sim.summarize(args, [], {"s": ["a", "b", "c", "d"]}, plan)
    assert "edit_model" not in summary and "vocabulary_source" not in summary
    assert summary["edits"]["deletion"] == 1


def test_summary_of_edit_model_v2_names_the_model_and_its_vocabulary(sim):
    args = argparse.Namespace(release="r", split="validation", wer=0.5, seed=1, edit_model="v2")
    plan = {"s": Edits(deleted=frozenset({0}))}
    summary = sim.summarize(args, [], {"s": ["a", "b", "c", "d"]}, plan)
    assert summary["edit_model"] == "v2" and summary["vocabulary_source"] == "default_train"


def rewrite(path: Path, change: Callable[[list[dict[str, Any]]], Any]) -> None:
    rows = read_rows(path)
    change(rows)
    write_lines(path, rows)


def append(path: Path, data: bytes) -> None:
    path.write_bytes(path.read_bytes() + data)


def set_sample(path: Path, **changes: Any) -> None:
    """Change fields of row 2 (sample s00, listed in every mode) of a data file."""
    rewrite(path, lambda rows: rows[2].update(changes))


def with_token(rows_path: Path, index: int, value: Any) -> None:
    rewrite(rows_path, lambda rows: rows[2]["tokens"].__setitem__(index, value))


def with_offsets_cut(rows_path: Path) -> None:
    rewrite(rows_path, lambda rows: rows[2]["metadata"].update(offsets=rows[2]["metadata"]["offsets"][1:]))


def with_offset(rows_path: Path, index: int, value: Any) -> None:
    rewrite(rows_path, lambda rows: rows[2]["metadata"]["offsets"].__setitem__(index, value))


BAD_SPAN = "metadata.offsets[3] must be [start, end] with 0 <= start <= end, both integers"
LINE_OF_ROW_2 = "mode_A/validation.jsonl line 3"  # row 2 is the third line of the file


DAMAGE = [  # (what to break under the data root, file the error must name, what the error must say)
    (lambda r: with_token(r / DATA["B"], 0, "zzz"), "mode_B/validation.jsonl", "tokens differ from another mode"),
    (lambda r: set_sample(r / DATA["C"], punct_labels=["O"]), "mode_C/validation.jsonl",
     "punct_labels must hold one string per token"),
    (lambda r: set_sample(r / DATA["A"], label_mode="B"), "mode_A/validation.jsonl", "label_mode must be A"),
    (lambda r: append(r / DATA["B"], b"{not json\n"), "mode_B/validation.jsonl", "not valid JSON"),
    (lambda r: append(r / DATA["B"], b"[" * 50_000 + b"\n"), "mode_B/validation.jsonl", "not valid JSON"),
    (lambda r: append(r / DATA["B"], b"[1,2]\n"), "mode_B/validation.jsonl", "not a JSON object"),
    (lambda r: rewrite(r / DATA["B"], lambda x: x.append(dict(x[2]))), "mode_B/validation.jsonl",
     "duplicate sample_id"),
    (lambda r: with_token(r / DATA["A"], 1, ""), "mode_A/validation.jsonl", "nonempty list of nonempty strings"),
    (lambda r: with_token(r / DATA["A"], 1, 5), "mode_A/validation.jsonl", "nonempty list of nonempty strings"),
    (lambda r: with_offsets_cut(r / DATA["A"]), "mode_A/validation.jsonl",
     "metadata.offsets must hold one entry per token"),
    (lambda r: with_offset(r / DATA["A"], 3, 5), LINE_OF_ROW_2, BAD_SPAN),  # not a list
    (lambda r: with_offset(r / DATA["A"], 3, [0, 1, 2]), LINE_OF_ROW_2, BAD_SPAN),  # three elements
    (lambda r: with_offset(r / DATA["A"], 3, ["0", 5]), LINE_OF_ROW_2, BAD_SPAN),  # string start
    (lambda r: with_offset(r / DATA["A"], 3, [True, 5]), LINE_OF_ROW_2, BAD_SPAN),  # bool is not an int here
    (lambda r: with_offset(r / DATA["A"], 3, [9, 3]), LINE_OF_ROW_2, BAD_SPAN),  # start after end
    (lambda r: with_offset(r / DATA["A"], 3, [-1, 5]), LINE_OF_ROW_2, BAD_SPAN),  # negative start
    (lambda r: (r / DATA["B"]).write_bytes(b"\xff\xfe\n"), "mode_B/validation.jsonl", "not UTF-8 text"),
    (lambda r: (r / IDS["B"]).unlink(), "validation_mode_B.ids", "No such file"),
    (lambda r: append(r / IDS["A"], b"nope\n"), "mode_A/validation.jsonl", "1 listed ID(s) not found"),
    (lambda r: append(r / IDS["A"], b"s00\n"), "validation_mode_A.ids", "Duplicate ID 's00' found at line"),
    (lambda r: (r / IDS["A"]).write_bytes(b"\n\n"), "validation_mode_A.ids", "ID list is empty"),
    (lambda r: rewrite(r / MASKS, lambda x: x.pop(2)), "case_joint/validation.jsonl", "1 listed ID(s) not found"),
    (lambda r: rewrite(r / MASKS, lambda x: x[2].update(excluded_positions=[999])), "case_joint/validation.jsonl",
     "excluded_positions must be unique integers"),
]


@pytest.mark.parametrize(("damage", "fragment", "reason"), DAMAGE)
def test_bad_inputs_exit_one_naming_the_file(sim, capsys, root, tmp_path, damage, fragment, reason):
    damage(root)
    code, stdout, err = run(sim, capsys, argv_for(root, tmp_path / "out"))
    assert code == 1 and stdout == ""
    assert_one_error_line(err)
    assert fragment in err and reason in err and str(tmp_path) not in err
    assert [p.name for p in tmp_path.iterdir()] == ["data"]


@pytest.mark.parametrize("wer", ["0", "0.51", "-0.1", "nan", "abc"])
def test_wer_must_be_above_zero_and_at_most_half(sim, root, tmp_path, wer):
    with pytest.raises(SystemExit, match="2"):
        sim.main(argv_for(root, tmp_path / "out", wer))  # type: ignore[arg-type]


def run_script(root: Path, out: Path, cwd: Path, **env: str) -> subprocess.CompletedProcess[str]:
    """The stub as a user runs it: a fresh interpreter, no PYTHONPATH
    (the installed package is found in site-packages)."""
    environ = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | env
    return subprocess.run([sys.executable, str(SCRIPT), *argv_for(root, out)], cwd=cwd, env=environ,
                          capture_output=True, text=True, check=False)


def test_runs_as_a_subprocess_without_pythonpath(root, tmp_path):
    out = tmp_path / "out"
    done = run_script(root, out, tmp_path)
    assert done.returncode == 0 and done.stderr == "", done.stderr
    assert (out / "summary.json").is_file()
    done = run_script(root, out, tmp_path)  # --out now exists
    assert done.returncode == 1 and done.stderr.startswith("error: ") and done.stderr.count("\n") == 1


def test_output_is_byte_identical_under_different_hash_seeds(root, tmp_path):
    trees = []
    for hash_seed in ("1", "2"):  # set iteration order changes with PYTHONHASHSEED
        out = tmp_path / f"out{hash_seed}"
        done = run_script(root, out, tmp_path, PYTHONHASHSEED=hash_seed)
        assert done.returncode == 0 and done.stderr == "", done.stderr
        trees.append(tree(out))
    assert trees[0] and trees[0] == trees[1]


def test_script_and_tests_have_no_todo_marker_or_bom():
    marker = "TO" + "DO"  # split so this file does not match itself
    for path in (SCRIPT, CLI_MODULE, MODULE, Path(__file__), Path(__file__).with_name("test_asr_noise.py")):
        data = path.read_bytes()
        assert not data.startswith(b"\xef\xbb\xbf") and marker.encode() not in data


GOLDEN_V1_HASHES = {
    "masks/validation.jsonl": "1eaca502fcc1f6089e4acfe8a5d7d4d9be5e960f334784f923c7fe200628a405",
    "mode_A/validation.jsonl": "5da685ef7e8dd9e683059c717d497d4135ccaa6e44b3153f0c933a13ade2fad7",
    "mode_B/validation.jsonl": "564e04ba24c4cd5950fd7d5d76e566f413df32277de7d05ef6e31f813f13db56",
    "mode_C/validation.jsonl": "5a13ed89835194a64633f645cb1c63164de6a3240da4cf94dc7e3af58f51b9b0",
    "summary.json": "6e44ed5a6c4be08990d92d5a40954ca23c232af2922363a26f9772afa3ae97ad",
    "validation_mode_A.ids": "07f8aa6ba363eae6af8494b023f85388297518292353449e905f1d81ef8dec45",
    "validation_mode_B.ids": "07f8aa6ba363eae6af8494b023f85388297518292353449e905f1d81ef8dec45",
    "validation_mode_C.ids": "33db7b073c4090c44f87a0c71a149b2070c3b12d52a20c44ba8349d549c2620f",
}


def test_v1_golden_output_is_byte_identical_by_default_and_explicit_flag(sim, capsys, root, tmp_path):
    import hashlib
    out_default = tmp_path / "out_default"
    out_explicit = tmp_path / "out_explicit"
    produce(sim, capsys, root, out_default, 0.10, seed=2026)
    code, _, err = run(sim, capsys, argv_for(root, out_explicit, 0.10, "--edit-model", "v1", seed=2026))
    assert code == 0, err
    hashes_default = {k: hashlib.sha256(v).hexdigest() for k, v in tree(out_default).items()}
    hashes_explicit = {k: hashlib.sha256(v).hexdigest() for k, v in tree(out_explicit).items()}
    assert hashes_default == GOLDEN_V1_HASHES
    assert hashes_explicit == GOLDEN_V1_HASHES


def write_train(root: Path, words_with_counts: dict[str, int], unlisted_words: list[str] | None = None) -> None:
    """Write mode_A/train.jsonl and default_train/mode_A.ids under root."""
    train_path = root / "processed" / RELEASE / "mode_A" / "train.jsonl"
    ids_path = root / "splits" / RELEASE / "default_train" / "mode_A.ids"
    rows = []
    listed_ids = []
    idx = 0
    for word, count in sorted(words_with_counts.items()):
        for _ in range(count):
            sid = f"tr{idx:03d}"
            listed_ids.append(sid)
            rows.append(row({"id": sid, "tokens": [word], "A": ["PERIOD"], "B": ["PERIOD"], "case": ["LOWER"]}, "A"))
            idx += 1
    if unlisted_words:
        rows.append(row({"id": "unlisted", "tokens": unlisted_words, "A": ["PERIOD"] * len(unlisted_words),
                         "B": ["PERIOD"] * len(unlisted_words), "case": ["LOWER"] * len(unlisted_words)}, "A"))
    write_lines(train_path, rows)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_bytes(("\n".join(listed_ids) + "\n").encode())


def replacement_words(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> set[str]:
    """Words that substitutions put in: output tokens that keep an input token's offsets but change its text."""
    found: set[str] = set()
    for old, new in zip(before, after, strict=True):
        source = {tuple(span): token for span, token in zip(old["metadata"]["offsets"], old["tokens"], strict=True)}
        found |= {token for span, token in zip(new["metadata"]["offsets"], new["tokens"], strict=True)
                  if span[0] != span[1] and token != source[tuple(span)]}  # an empty span is an inserted token
    return found


def test_v2_replacements_come_only_from_default_train_with_min_count_five_never_the_evaluation_records(
        sim, capsys, tmp_path):
    # Every utterance is "mark" + 19 x "marx" (validation only, 570 times, so it would qualify if the evaluation
    # tokens were counted; "mark" would too, 30 times). Train: "mars" 5 times (default_train, qualifies), "mare" 4
    # times (below 5) and "mart" 10 times in a train record that default_train does not list.
    samples = [make_sample(i, ["mark"] + ["marx"] * 19) for i in range(SAMPLES)]
    data_root = write_root(tmp_path / "data", samples, LISTED_C)
    write_train(data_root, {"mars": 5, "mare": 4}, unlisted_words=["mart"] * 10)
    out = tmp_path / "out_v2"
    code, _, err = run(sim, capsys, argv_for(data_root, out, 0.10, "--edit-model", "v2", seed=2026))
    assert code == 0, err
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["edit_model"] == "v2"
    assert summary["vocabulary_source"] == "default_train"
    assert summary["edits"]["substitution"] > 0
    after = read_rows(out / "mode_A" / "validation.jsonl")
    assert replacement_words(listed_input(data_root, "A"), after) == {"mars"}


def test_v2_outputs_validate_and_summary_records_shares_and_masks_insertions(sim, capsys, root, tmp_path):
    write_train(root, {"cap": 5, "map": 5, "hat": 5, "ran": 5, "tan": 5, "fan": 5, "now": 5, "new": 5})
    out = tmp_path / "out_v2"
    code, _, err = run(sim, capsys, argv_for(root, out, 0.10, "--edit-model", "v2", seed=2026))
    assert code == 0, err
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["edit_model"] == "v2"
    assert summary["vocabulary_source"] == "default_train"
    assert summary["cost_share"] == {"deletion": 0.3, "insertion": 0.15, "substitution": 0.45, "swap": 0.1}
    assert summary["tokens_after"] == (
        summary["tokens_before"] - summary["edits"]["deletion"] + summary["edits"]["insertion"]
    )
    for mode in MODES:
        gold = out / f"mode_{mode}" / "validation.jsonl"
        ids = out / f"validation_mode_{mode}.ids"
        found_mode, records = load_scored_gold(gold, ids, load_vocab())
        assert found_mode == mode and len(records) == len(listed_input(root, mode))
    after_c = {r["sample_id"]: r for r in read_rows(out / "mode_C" / "validation.jsonl")}
    masks = {m["sample_id"]: m["excluded_positions"] for m in read_rows(out / "masks" / "validation.jsonl")}
    inserted_found = False
    for sid, rec in after_c.items():
        for idx, span in enumerate(rec["metadata"]["offsets"]):
            if span[0] == span[1]:  # empty span = inserted token
                inserted_found = True
                assert idx in masks[sid]
                assert rec["punct_labels"][idx] == "O" and rec["case_labels"][idx] == "LOWER"
    assert inserted_found


def test_v2_missing_train_file_exits_one_naming_the_file(sim, capsys, root, tmp_path):
    code, stdout, err = run(sim, capsys, argv_for(root, tmp_path / "out_v2", 0.10, "--edit-model", "v2"))
    assert code == 1 and stdout == ""
    assert_one_error_line(err)
    assert "mode_A.ids" in err and str(tmp_path) not in err


def test_v2_bad_offsets_exit_one_instead_of_crashing_on_an_insertion(sim, capsys, root, tmp_path):
    write_train(root, {"cap": 5, "map": 5, "hat": 5, "ran": 5, "tan": 5, "fan": 5, "now": 5, "new": 5})
    rewrite(root / DATA["A"], lambda rows: [r["metadata"].update(offsets=[5] * len(r["tokens"])) for r in rows])
    out = tmp_path / "out_v2"
    code, stdout, err = run(sim, capsys, argv_for(root, out, 0.10, "--edit-model", "v2"))
    assert code == 1 and stdout == ""
    assert_one_error_line(err)
    assert "mode_A/validation.jsonl line" in err and "metadata.offsets[0] must be" in err and str(tmp_path) not in err
    assert not out.exists() and [p.name for p in tmp_path.iterdir()] == ["data"]


def test_invalid_edit_model_exits_two(sim, root, tmp_path):
    with pytest.raises(SystemExit, match="2"):
        sim.main(argv_for(root, tmp_path / "out", 0.10, "--edit-model", "v3"))


