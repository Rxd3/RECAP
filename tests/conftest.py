"""Shared fixtures for the Member 5 evaluation tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from cli_analysis_cases import build_case
from eval_fixtures import MODULE_PATH, Case

from punct_restore.evaluation import load_vocab
from punct_restore.evaluation.inputs import Vocab


@pytest.fixture(scope="session")
def vocab() -> Vocab:
    """The real label vocabularies from configs/label_sets.yaml."""
    return load_vocab()


@pytest.fixture(scope="session")
def cli() -> ModuleType:
    """punct_restore.cli.evaluate_models loaded fresh from its file, so tests can call main(argv) in-process."""
    spec = importlib.util.spec_from_file_location("evaluate_models_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def grouped_case(tmp_path: Path) -> Case:
    """Mode A command line case: 8 records over 4 groups, preds.jsonl (model_a) and preds_b.jsonl (model_b)."""
    return build_case(tmp_path / "case", "A")


@pytest.fixture
def mode_c_case(tmp_path: Path) -> Case:
    """The same case in mode C, with a masks file."""
    return build_case(tmp_path / "case_c", "C")
