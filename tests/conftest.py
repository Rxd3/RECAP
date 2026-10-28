"""Shared fixtures for the Member 5 evaluation tests."""

from __future__ import annotations

import importlib.util
from types import ModuleType

import pytest
from eval_fixtures import SCRIPT_PATH

from punct_restore.evaluation import load_vocab
from punct_restore.evaluation.inputs import Vocab


@pytest.fixture(scope="session")
def vocab() -> Vocab:
    """The real label vocabularies from configs/label_sets.yaml."""
    return load_vocab()


@pytest.fixture(scope="session")
def cli() -> ModuleType:
    """scripts/evaluate_models.py imported as a module, so tests can call main(argv) in-process."""
    spec = importlib.util.spec_from_file_location("evaluate_models_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
