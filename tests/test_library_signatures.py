"""Spec section 2: every public library callable of main 72ebfd1 is still called the way main's callers called it.

`fixtures/sigs_main.json` is the signature snapshot taken on main (`api_sigs.py dump`): for each of 264 callables
the (name, kind, has a default) of every parameter, in order. The rule is the checker's (`api_sigs.py check`): a
callable passes when its parameter list on this branch is equal to main's. A removed, renamed or reordered parameter,
a changed kind (positional or keyword only) and a dropped default are changes; so is a new parameter, because the
snapshot is the whole call form. A callable that `inspect.signature` cannot read (a builtin) is not compared.

`tests/test_library_names.py` checks that the names still import; this file checks that the old calls still work.
"""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sigs_main.json"
SNAPSHOT_CALLABLES = 264
SNAPSHOT_UNREADABLE = 8  # builtins and C functions of main that `inspect.signature` could not read

Parameters = list[tuple[str, str, bool]]


def parameters_of(obj: Callable[..., Any]) -> Parameters | None:
    """(name, kind, has a default) of every parameter of ``obj``; None when the signature cannot be read."""
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    return [(p.name, p.kind.name, p.default is not inspect.Parameter.empty) for p in signature.parameters.values()]


def _spelled(parameters: Parameters) -> str:
    return ", ".join(f"{name} ({kind.lower()}{', default' if default else ''})" for name, kind, default in parameters)


def signature_problem(key: str, main: Parameters | None, head: Parameters | None) -> str | None:
    """One text naming both parameter lists when they differ; None when equal or when one cannot be read."""
    if main is None or head is None or main == head:
        return None
    return f"{key}\n    main: {_spelled(main)}\n    head: {_spelled(head)}"


def _snapshot() -> dict[str, Parameters | None]:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {key: None if rows is None else [(name, kind, bool(has_default)) for name, kind, has_default in rows]
            for key, rows in raw.items()}


def _head_parameters(key: str) -> Parameters | None:
    module, name = key.rsplit(".", 1)
    return parameters_of(getattr(importlib.import_module(module), name))


def test_the_snapshot_holds_every_callable_of_main() -> None:
    """A truncated or emptied fixture would make the comparison pass vacuously."""
    snapshot = _snapshot()
    assert len(snapshot) == SNAPSHOT_CALLABLES
    assert sum(rows is None for rows in snapshot.values()) == SNAPSHOT_UNREADABLE


def test_every_callable_of_main_still_has_the_parameters_of_main() -> None:
    problems = [text for key, main in _snapshot().items()
                if (text := signature_problem(key, main, _head_parameters(key)))]
    assert problems == [], "\n".join(problems)


# --- the comparison itself: each change a caller of main would feel is caught -----------------------------------


def _main_form(path: str, ids: str, *, strict: bool = False, log: object = None) -> None: ...


def _problem(head: Callable[..., Any]) -> str | None:
    return signature_problem("m.f", parameters_of(_main_form), parameters_of(head))


def test_the_comparison_accepts_the_same_parameters_under_other_annotations_and_defaults() -> None:
    def head(path: bytes, ids: bytes, *, strict: bool = True, log: int = 3) -> int: ...

    assert _problem(head) is None


def test_the_comparison_catches_a_removed_parameter() -> None:
    def head(path: str, ids: str, *, strict: bool = False) -> None: ...

    assert _problem(head) is not None


def test_the_comparison_catches_a_dropped_default() -> None:
    def head(path: str, ids: str, *, strict: bool, log: object = None) -> None: ...

    assert _problem(head) is not None


def test_the_comparison_catches_a_renamed_parameter() -> None:
    def head(path: str, ids_path: str, *, strict: bool = False, log: object = None) -> None: ...

    assert _problem(head) is not None


def test_the_comparison_catches_a_keyword_only_parameter_that_became_positional() -> None:
    def head(path: str, ids: str, strict: bool = False, *, log: object = None) -> None: ...

    assert _problem(head) is not None


def test_the_comparison_catches_a_new_parameter_because_the_snapshot_is_the_whole_call_form() -> None:
    def head(path: str, ids: str, *, strict: bool = False, log: object = None, extra: int = 0) -> None: ...

    assert _problem(head) is not None


def test_the_comparison_names_the_callable_and_both_parameter_lists() -> None:
    def head(sources: object, *, log: object = None) -> None: ...

    problem = _problem(head)
    assert problem is not None
    assert problem.startswith("m.f\n")
    assert "main: path (positional_or_keyword), ids (positional_or_keyword)" in problem
    assert "head: sources (positional_or_keyword), log (keyword_only, default)" in problem


class _Unreadable:
    __signature__ = "not a signature"  # inspect.signature raises TypeError for it

    def __call__(self) -> None: ...


def test_the_comparison_skips_a_callable_it_cannot_read() -> None:
    assert parameters_of(_Unreadable()) is None
    assert signature_problem("m.f", None, parameters_of(_main_form)) is None
    assert signature_problem("m.f", parameters_of(_main_form), None) is None
