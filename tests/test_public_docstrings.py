"""The public functions the clean-code review listed without a docstring now have one that says what they do."""

from __future__ import annotations

import importlib

import pytest

LISTED = [
    ("punct_restore.cli.package_hmm_handoff", "package"),
    ("punct_restore.models.hmm_release", "checked_release_file"),
    ("punct_restore.models.hmm_workflow", "portable_path"),
    ("punct_restore.models.hmm_workflow", "code_identity"),
    ("punct_restore.models.hmm_workflow", "write_json"),
    ("punct_restore.cli.train_hmm", "nonempty"),
    ("punct_restore.cli.simulate_asr_noise", "write_jsonl"),
    ("punct_restore.cli.simulate_asr_noise", "summarize"),
]


@pytest.mark.parametrize(("module", "name"), LISTED, ids=[f"{m.rsplit('.', 1)[1]}.{n}" for m, n in LISTED])
def test_a_listed_public_function_has_a_docstring(module: str, name: str) -> None:
    doc = getattr(importlib.import_module(module), name).__doc__
    assert doc is not None and len(doc.split()) >= 4, f"{module}.{name} needs a docstring that says what it does"
