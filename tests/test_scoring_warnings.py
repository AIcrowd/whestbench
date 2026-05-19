from __future__ import annotations

import whestbench
from whestbench.scoring import (
    BudgetExhaustionWarning,
    ScoringExhaustionWarning,
    TimeExhaustionWarning,
)


def test_warning_class_hierarchy() -> None:
    """ScoringExhaustionWarning is a UserWarning; budget/time inherit from it."""
    assert issubclass(ScoringExhaustionWarning, UserWarning)
    assert issubclass(BudgetExhaustionWarning, ScoringExhaustionWarning)
    assert issubclass(TimeExhaustionWarning, ScoringExhaustionWarning)


def test_warning_classes_reexported_from_package() -> None:
    """Library users can import the classes from the top-level package."""
    assert whestbench.ScoringExhaustionWarning is ScoringExhaustionWarning
    assert whestbench.BudgetExhaustionWarning is BudgetExhaustionWarning
    assert whestbench.TimeExhaustionWarning is TimeExhaustionWarning
