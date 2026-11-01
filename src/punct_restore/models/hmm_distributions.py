"""A read-only copy of the log-probability tables of a fitted HMM, for code that checks them (no model state here)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class HMMDistributions:
    """The tables a fitted `HMMModel` decodes with, one entry per state in label order.

    Every field is immutable and detached from the model: changing the model later does not change a copy taken
    before, and nothing here can change the model.
    """

    log_start: tuple[float, ...]
    log_transition: tuple[tuple[float, ...], ...]
    log_end: tuple[float, ...]
    vocabulary: frozenset[str]
    log_emissions: tuple[Mapping[str, float], ...]
    log_default: tuple[float, ...]
    log_unknown: tuple[float, ...]

    @classmethod
    def copy_of(
        cls,
        log_start: Sequence[float],
        log_transition: Sequence[Sequence[float]],
        log_end: Sequence[float],
        vocabulary: frozenset[str],
        log_emissions: Sequence[Mapping[str, float]],
        log_default: Sequence[float],
        log_unknown: Sequence[float],
    ) -> HMMDistributions:
        """Freeze copies of the given tables."""
        return cls(
            log_start=tuple(log_start),
            log_transition=tuple(tuple(row) for row in log_transition),
            log_end=tuple(log_end),
            vocabulary=frozenset(vocabulary),
            log_emissions=tuple(MappingProxyType(dict(table)) for table in log_emissions),
            log_default=tuple(log_default),
            log_unknown=tuple(log_unknown),
        )
