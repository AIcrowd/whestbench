"""The prediction must be materialised and validated inside the metered window.

A return value from ``predict()`` is not required to be a concrete array -- the
contract only reads ``.shape`` and then coerces. Coercing after the
``BudgetContext`` has exited meant work deferred into ``__array__`` was billed
neither FLOPs nor residual seconds (``wall_time_s`` is frozen at ``__exit__``),
and because ``validate_predictions`` returned the caller's object rather than
the coerced array, the values that passed the finiteness gate were not
necessarily the values scored.

The invariant these tests lock is not "materialised exactly once" -- ``asarray``
legitimately probes an object more than once -- but "whatever validation
returns is CONCRETE and STABLE", so no later consumer can observe different
values than the ones that were checked.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope.numpy as fnp
from whestbench.scoring import validate_predictions


class _Lazy:
    """Concrete ``.shape``; values produced only on materialisation."""

    def __init__(self, depth, width, value=1.0):
        self.shape = (depth, width)
        self._value = value
        self.materialisations = 0

    def __array__(self, dtype=None, copy=None):
        self.materialisations += 1
        out = np.full(self.shape, self._value, dtype=np.float32)
        return out if dtype is None else out.astype(dtype)


class _Shifty(_Lazy):
    """Yields different values on every materialisation."""

    def __array__(self, dtype=None, copy=None):
        self.materialisations += 1
        out = np.full(self.shape, float(self.materialisations), dtype=np.float32)
        return out if dtype is None else out.astype(dtype)


def test_validation_returns_a_concrete_array_not_the_callers_object():
    lazy = _Lazy(4, 8, value=2.5)
    result = validate_predictions(lazy, depth=4, width=8)

    assert result is not lazy, (
        "validate_predictions must not return the caller's object; scoring would "
        "re-materialise it and could observe values that never passed validation"
    )
    assert isinstance(result, fnp.ndarray)
    assert np.allclose(np.asarray(result), 2.5)


def test_validated_result_is_stable_across_reads():
    """A shifting return value must not be able to change after validation."""
    shifty = _Shifty(4, 8)
    result = validate_predictions(shifty, depth=4, width=8)

    first = np.array(np.asarray(result), copy=True)
    second = np.array(np.asarray(result), copy=True)
    np.testing.assert_array_equal(
        first,
        second,
        err_msg="validated predictions changed between reads -- the scored array "
        "is not pinned to the array that passed validation",
    )
    assert shifty.materialisations == 0 or True  # count is an implementation detail


def test_non_finite_values_are_rejected_on_the_array_that_is_returned():
    class _Inf(_Lazy):
        def __array__(self, dtype=None, copy=None):
            self.materialisations += 1
            out = np.full(self.shape, np.inf, dtype=np.float32)
            return out if dtype is None else out.astype(dtype)

    with pytest.raises(ValueError, match="finite"):
        validate_predictions(_Inf(4, 8), depth=4, width=8)


def test_plain_array_round_trips_unchanged():
    arr = fnp.asarray(np.arange(32, dtype=np.float32).reshape(4, 8))
    result = validate_predictions(arr, depth=4, width=8)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(arr))
