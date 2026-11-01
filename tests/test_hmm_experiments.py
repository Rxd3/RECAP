"""Independent workflow gates and inference/metric checks on tiny fixtures."""

import ast
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml
from test_hmm_cli import ROOT, jsonl, setup_run
from test_hmm_workflow import run

from punct_restore.hashing import sha256_file
from punct_restore.models import hmm_experiment_audit as audit
from punct_restore.models import hmm_experiment_finalize, hmm_experiment_identity
from punct_restore.models.hmm import HMMModel
from punct_restore.models.hmm_experiment_commands import HmmCommandFailed, execute
from punct_restore.models.hmm_experiment_finalize import finalize
from punct_restore.models.hmm_experiment_identity import SOURCE_FILES, check_evaluator, implementation
from punct_restore.models.hmm_experiment_protocol import choose, validate_config
from punct_restore.models.hmm_workflow import PredictionOptions, predict_run, predict_run_with

HMM_COMMANDS = ("train_hmm", "predict_hmm", "run_hmm_experiments", "verify_hmm_experiments")


def config():
    return yaml.safe_load((ROOT / "configs/hmm_sponspeech_v2.yaml").read_text())


def test_source_files_list_the_hmm_modules_then_the_shared_code_then_each_command_as_module_then_stub():
    """The hashed code names the study's own modules, the shared code they load (sorted), then each command and stub."""
    models = ("hmm", "hmm_distributions", "hmm_validation", "hmm_records", "hmm_workflow", "hmm_release",
              "hmm_experiment_protocol",
              "hmm_experiment_commands", "hmm_experiment_runtime", "hmm_experiment_audit",
              "hmm_experiment_identity", "hmm_experiment_tune", "hmm_experiment_frozen",
              "hmm_experiment_finalize", "hmm_experiment_verify")
    own = [f"src/punct_restore/models/{name}.py" for name in models]
    commands = []
    for command in HMM_COMMANDS:
        commands += [f"src/punct_restore/cli/{command}.py", f"scripts/{command}.py"]
    names = list(SOURCE_FILES)
    shared = names[len(own):-len(commands)]
    assert names[:len(own)] == own and names[-len(commands):] == commands
    assert shared == sorted(shared) and all(name.startswith("src/punct_restore/") for name in shared)
    assert [name for name in names if not (ROOT / name).is_file()] == []
    assert list(implementation()) == names


CLOSURE_SCRIPT = """
import sys
from pathlib import Path

import punct_restore.cli.predict_hmm, punct_restore.cli.run_hmm_experiments
import punct_restore.cli.train_hmm, punct_restore.cli.verify_hmm_experiments

root = Path(sys.argv[1])
files = (getattr(m, "__file__", None) for name, m in sys.modules.items() if name.split(".")[0] == "punct_restore")
print("\\n".join(sorted(Path(file).resolve().relative_to(root).as_posix() for file in files if file)))
"""


def test_source_files_are_exactly_the_modules_the_four_hmm_commands_load():
    """A file the study loads decides its results, so a change to it must change the record: package inits included."""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}  # the worktree's code, not an editable install elsewhere
    done = subprocess.run([sys.executable, "-c", CLOSURE_SCRIPT, str(ROOT.resolve())],
                          env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    loaded = set(done.stdout.split())
    listed = {name for name in SOURCE_FILES if name.startswith("src/")}
    assert "src/punct_restore/data/__init__.py" in loaded  # the scan sees the package inits the commands run first
    assert sorted(loaded - listed) == [], "loaded by the study but not hashed"
    assert sorted(listed - loaded) == [], "hashed but not loaded by the study"


FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _imports_punct_restore(node: ast.AST) -> bool:
    """Whether a statement imports from the package, by its name (absolute) or by a dot (relative)."""
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] == "punct_restore" for alias in node.names)
    return isinstance(node, ast.ImportFrom) and (node.level > 0 or (node.module or "").split(".")[0] == "punct_restore")


def lazy_punct_restore_imports(source: str) -> list[int]:
    """The line numbers of the imports of punct_restore, absolute or relative, that sit inside a function."""
    return sorted({
        node.lineno
        for function in ast.walk(ast.parse(source)) if isinstance(function, FUNCTIONS)
        for node in ast.walk(function) if _imports_punct_restore(node)
    })


LAZY_SOURCE = """
import json
from punct_restore.top import fine
from . import also_fine


def plain():
    import os
    from collections import Counter


def inside(path):
    import punct_restore.models.hmm
    from punct_restore.data import labeling
    from punct_restore import paths
    from .sibling import thing
    from . import other
    from .. import parent
    return None


class Holder:
    def method(self):
        import punct_restore.file_io as io

        def nested():
            from punct_restore.hashing import sha256_file
"""


def test_the_lazy_import_checker_finds_every_form_of_import_inside_a_function_and_nothing_else():
    """Self-test on inline source: a checker that misses a form (relative, `from x import y`, a method, a nested
    function) or flags a module-level or standard-library import would make the closure test below worthless."""
    found = lazy_punct_restore_imports(LAZY_SOURCE)
    lines = LAZY_SOURCE.splitlines()
    assert [lines[number - 1].strip() for number in found] == [
        "import punct_restore.models.hmm",
        "from punct_restore.data import labeling",
        "from punct_restore import paths",
        "from .sibling import thing",
        "from . import other",
        "from .. import parent",
        "import punct_restore.file_io as io",
        "from punct_restore.hashing import sha256_file",
    ]
    assert lazy_punct_restore_imports("import punct_restore\nfrom punct_restore.x import y\nfrom . import z\n") == []


def test_no_hashed_source_file_imports_punct_restore_inside_a_function():
    """The runtime-closure test only sees an import that has run: a lazy one would load a module the record misses."""
    offenders = {
        name: lines
        for name in SOURCE_FILES
        if name.startswith("src/") and (lines := lazy_punct_restore_imports((ROOT / name).read_text(encoding="utf-8")))
    }
    assert offenders == {}


def test_the_module_that_computes_the_implementation_record_is_itself_hashed():
    assert "src/punct_restore/models/hmm_experiment_identity.py" in SOURCE_FILES


def test_check_evaluator_still_hashes_scripts_evaluate_models_in_the_pinned_checkout(
    tmp_path, monkeypatch
):
    """The pinned evaluator commit keeps its scorer at scripts/evaluate_models.py, so that path stays."""
    names = [
        "src/punct_restore/evaluation/report.py",
        "scripts/evaluate_models.py",
        "configs/label_sets.yaml",
    ]
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(name, encoding="utf-8")
    pinned = {"commit": "pinned", "dirty": False}
    monkeypatch.setattr(hmm_experiment_identity, "code_identity", lambda _root: pinned)
    files = check_evaluator(tmp_path, "pinned")["files"]
    assert sorted(files) == sorted(names)
    scorer = tmp_path / "scripts/evaluate_models.py"
    assert files["scripts/evaluate_models.py"] == sha256_file(scorer)


def test_check_evaluator_rejects_a_dirty_or_unpinned_checkout(tmp_path, monkeypatch):
    """The patched identity must be the one check_evaluator reads: a different commit is refused."""
    monkeypatch.setattr(hmm_experiment_identity, "code_identity", lambda _root: {"commit": "other", "dirty": False})
    with pytest.raises(ValueError, match="clean at the pinned"):
        check_evaluator(tmp_path, "pinned")


def test_selection_uses_only_validation_score_and_predeclared_tie_break():
    records = [
        {"validation_primary": 0.2, "min_token_count": 1, "smoothing": 0.01, "test_score": 1.0},
        {"validation_primary": 0.3, "min_token_count": 5, "smoothing": 0.01, "test_score": 1.0},
        {"validation_primary": 0.3, "min_token_count": 1, "smoothing": 0.1, "test_score": 0.0},
        {"validation_primary": 0.3, "min_token_count": 1, "smoothing": 0.01, "test_score": 0.0},
    ]
    assert choose(records) is records[3]
    for record in records:
        record["test_score"] = 1 - record["test_score"]
    assert choose(records) is records[3]


@pytest.mark.parametrize(
    "bad",
    [
        "test-selection",
        "retrain-validation",
        "C-wrong-metric",
        "negative-alpha",
        "fractional-cutoff",
        "repeat-grid",
        "missing-test-amb",
        "wrong-tie",
    ],
)
def test_experiment_rejects_invalid_or_test_driven_protocols(bad):
    c = config()
    if bad == "test-selection":
        c["search"]["selection_split"] = "test"
    elif bad == "retrain-validation":
        c["search"]["retrain_on_validation"] = True
    elif bad == "C-wrong-metric":
        c["search"]["selection_blocks"]["C"] = "punctuation/all"
    elif bad == "negative-alpha":
        c["search"]["smoothing"] = [-0.1]
    elif bad == "fractional-cutoff":
        c["search"]["min_token_count"] = [1.1]
    elif bad == "repeat-grid":
        c["search"]["smoothing"] = [0.1, 0.1]
    elif bad == "missing-test-amb":
        c["evaluation"]["test_sets"] = ["test"]
    else:
        c["search"]["tie_break"] = ["test_score"]
    with pytest.raises(ValueError):
        validate_config(c)


def test_finalize_cannot_touch_test_data_before_global_selection_exists(tmp_path, monkeypatch):
    (tmp_path / "plan.json").write_text(json.dumps({"config": config()}))
    called = []
    monkeypatch.setattr(hmm_experiment_finalize, "authenticate", lambda *args: called.append(args))
    with pytest.raises(FileNotFoundError):
        finalize(tmp_path, tmp_path / "evaluator")
    assert not called and not (tmp_path / "final").exists()


def test_a_failing_command_raises_hmm_command_failed_naming_only_its_log(tmp_path):
    log = tmp_path / "deep" / "c1.training.log"
    failing = [sys.executable, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(3)"]
    with pytest.raises(HmmCommandFailed) as caught:
        execute(failing, tmp_path, log)
    assert str(caught.value) == "command failed; see c1.training.log"
    assert isinstance(caught.value, RuntimeError)
    assert "boom" in log.read_text(encoding="utf-8")  # the detail the message leaves out is in the log
    assert json.loads((tmp_path / "commands.jsonl").read_text(encoding="utf-8"))["returncode"] == 3


def test_a_succeeding_command_is_logged_and_returned(tmp_path):
    record = execute([sys.executable, "-c", "print('fine')"], tmp_path, tmp_path / "ok.log")
    assert record["returncode"] == 0 and record["log"] == "ok.log"
    assert (tmp_path / "ok.log").read_text(encoding="utf-8").strip() == "fine"


def test_independent_f1_counts_gold_O_errors_and_absent_classes():
    result = audit.independent_scores(
        [("O", "COMMA"), ("COMMA", "O"), ("PERIOD", "PERIOD")],
        ["O", "COMMA", "PERIOD", "QUESTION"],
        ["COMMA", "PERIOD", "QUESTION"],
    )
    assert result["confusion"] == [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
    assert result["per_class"]["COMMA"]["predicted"] == 1
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["micro"]["f1"] == pytest.approx(0.5)
    assert result["per_class"]["QUESTION"]["f1"] == 0


@pytest.mark.parametrize("mode", ["A", "B", "C"])
def test_saved_model_inference_is_reproducible_and_does_not_read_gold(tmp_path, mode):
    args, _, prediction, trained = setup_run(tmp_path, mode)
    assert run(args).returncode == 0
    rows = [
        {"schema_version": 1, "sample_id": "p1", "tokens": ["hello", "unseen"]},
        {"schema_version": 1, "sample_id": "p2", "tokens": ["nasa"]},
    ]
    jsonl(prediction, rows)
    first = tmp_path / "pred1"
    before = predict_run(trained / "model.json", prediction, first)
    poisoned = [
        {
            **row,
            "punct_labels": ["INVALID"],
            "case_labels": [],
            "metadata": {"original_text": "A GOLD LEAK"},
        }
        for row in rows
    ]
    jsonl(prediction, poisoned)
    second = tmp_path / "pred2"
    after = predict_run(trained / "model.json", prediction, second)
    assert before["predictions_sha256"] == after["predictions_sha256"]
    assert before["num_sequences"] == 2 and before["num_tokens"] == 3
    assert before["inputs"]["input"]["sha256"] != after["inputs"]["input"]["sha256"]
    with pytest.raises(ValueError, match="already exists"):
        predict_run(trained / "model.json", prediction, first)


def test_artifact_cannot_be_substituted_for_another_frozen_winner(tmp_path):
    args, _, prediction, trained = setup_run(tmp_path)
    assert run(args).returncode == 0
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "used_test_labels": False,
                "training_includes_validation": False,
                "modes": {"A": {"model_sha256": "incorrect"}},
            }
        )
    )
    out = tmp_path / "prediction-output"
    with pytest.raises(ValueError, match="frozen selection"):
        predict_run_with(trained / "model.json", prediction, out, PredictionOptions(selection_record=selection))
    assert not out.exists()


def _fitted_model():
    labels = ["O", "COMMA", "PERIOD", "QUESTION"]
    train = {"schema_version": 1, "sample_id": "t", "source": "unit", "split": "train", "label_mode": "A",
             "tokens": ["a", "b"], "punct_labels": ["O", "PERIOD"], "case_labels": ["LOWER", "LOWER"]}
    return HMMModel(label_mode="A", labels=labels, min_token_count=1).fit([train])


def test_the_probability_check_reads_only_the_public_distributions_of_the_model():
    """A stand-in that offers `labels` and `distributions` alone is enough: no private field is read."""
    model = _fitted_model()
    audit.probability_checks(SimpleNamespace(labels=model.labels, distributions=model.distributions))


def test_the_probability_check_rejects_a_distribution_that_does_not_sum_to_one():
    model = _fitted_model()
    broken = replace(model.distributions, log_start=(0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="metric mismatch"):
        audit.probability_checks(SimpleNamespace(labels=model.labels, distributions=broken))


def test_metric_checker_rejects_a_green_looking_incorrect_report():
    expected = audit.independent_scores([("O", "COMMA")], ["O", "COMMA"], ["COMMA"])
    wrong = deepcopy(expected)
    wrong["macro_f1"] = 1.0
    with pytest.raises(ValueError, match="metric mismatch"):
        audit.assert_block(wrong, expected)
