"""The metered window must bound participant work exactly -- no more, no less.

Two failures are possible here and they pull in opposite directions:

* Work deferred out of the window (into ``__array__``) is billed to nobody, so a
  submission gets free compute.
* Harness checks pulled into the window are billed to the participant, so an
  honest submission pays for the grader's own bookkeeping.

Both are tested, because a fix for either one is easy to write in a way that
causes the other.
"""

from __future__ import annotations

import gc

import flopscope
import flopscope.numpy as fnp
import numpy as np
import pytest

from whestbench import subprocess_worker
from whestbench.scoring import (
    ContestData,
    ContestSpec,
    evaluate_estimator,
    make_contest,
    materialise_predictions,
    validate_predictions,
)
from whestbench.sdk import BaseEstimator

_W, _D = 32, 4
_DEFERRED_MATMULS = 200
_DEFERRED_FLOPS = _DEFERRED_MATMULS * 2 * 64**3


def _burn() -> None:
    """Real, countable flopscope work."""
    a = fnp.asarray(np.random.rand(64, 64).astype(np.float32))
    acc = a
    for _ in range(_DEFERRED_MATMULS):
        acc = fnp.matmul(acc, a)


class _Deferred:
    """Concrete ``.shape``; the work happens only on materialisation."""

    def __init__(self, depth: int, width: int) -> None:
        self.shape = (depth, width)

    def __array__(self, dtype=None, copy=None):
        _burn()
        out = np.zeros(self.shape, dtype=np.float32)
        return out if dtype is None else out.astype(dtype)


class _Shifty:
    """Yields different values on every materialisation."""

    def __init__(self, depth: int, width: int) -> None:
        self.shape = (depth, width)
        self.n = 0

    def __array__(self, dtype=None, copy=None):
        self.n += 1
        out = np.full(self.shape, float(self.n), dtype=np.float32)
        return out if dtype is None else out.astype(dtype)


class _DeferredEstimator(BaseEstimator):
    def setup(self, context):
        pass

    def predict(self, mlp, budget):  # pyright: ignore[reportIncompatibleMethodOverride]
        return _Deferred(mlp.depth, mlp.width)


class _InlineEstimator(BaseEstimator):
    """Does the SAME work, inline, where it is unambiguously billable."""

    def setup(self, context):
        pass

    def predict(self, mlp, budget):
        _burn()
        return fnp.asarray(np.zeros((mlp.depth, mlp.width), dtype=np.float32))


class _ZeroFlopEstimator(BaseEstimator):
    def setup(self, context):
        pass

    def predict(self, mlp, budget):
        return fnp.asarray(np.zeros((mlp.depth, mlp.width), dtype=np.float32))


def _contest(flop_budget: int) -> ContestData:
    spec = ContestSpec(
        width=_W, depth=_D, n_mlps=1, flop_budget=flop_budget, ground_truth_samples=64, seed=3
    )
    return make_contest(spec)


# --- work must not escape the window ----------------------------------------


def test_deferred_work_is_billed_like_inline_work():
    """The headline invariant.

    Deferring identical computation into ``__array__`` must not make it free.
    Both estimators do exactly ``_DEFERRED_FLOPS`` of work; a budget below that
    must stop both.
    """
    budget = _DEFERRED_FLOPS // 4
    data = _contest(budget)

    inline = evaluate_estimator(_InlineEstimator(), data)["per_mlp"][0]
    deferred = evaluate_estimator(_DeferredEstimator(), data)["per_mlp"][0]

    assert inline["flops_used"] > 0
    assert deferred["flops_used"] == inline["flops_used"], (
        "work deferred into __array__ was billed differently from the same work "
        "done inline -- the metered window does not bound participant compute"
    )


def test_deferred_work_can_exhaust_the_budget():
    data = _contest(_DEFERRED_FLOPS // 4)
    with pytest.warns(Warning):
        row = evaluate_estimator(_DeferredEstimator(), data)["per_mlp"][0]
    assert row.get("budget_exhausted") is True


def test_worker_bills_deferred_work(monkeypatch):
    sent: dict = {}
    monkeypatch.setattr(subprocess_worker, "_write_response", lambda p: sent.update(p))
    mlp = _contest(10**12).mlps[0]
    request = {
        "mlp": {
            "width": _W,
            "depth": _D,
            "seed": 0,
            "weights": [np.asarray(w).tolist() for w in mlp.weights],
        },
        "budget": 10**12,
    }
    subprocess_worker._handle_predict(_DeferredEstimator(), request)
    assert sent["status"] == "ok"
    assert sent["flops_used"] >= _DEFERRED_FLOPS * 0.9


class _Finaliser:
    """Does its work in ``__del__`` rather than ``__array__``."""

    def __init__(self, depth: int, width: int) -> None:
        self.shape = (depth, width)

    def __array__(self, dtype=None, copy=None):
        out = np.zeros(self.shape, dtype=np.float32)
        return out if dtype is None else out.astype(dtype)

    def __del__(self):
        try:
            _burn()
        except Exception:  # interpreter teardown
            pass


class _FinaliserEstimator(BaseEstimator):
    def setup(self, context):
        pass

    def predict(self, mlp, budget):  # pyright: ignore[reportIncompatibleMethodOverride]
        return _Finaliser(mlp.depth, mlp.width)


@pytest.mark.parametrize("path", ["local", "worker"])
def test_finaliser_work_is_billed_on_both_paths(path, monkeypatch):
    """``__del__`` is the hook that survives materialising inside the window.

    Materialisation alone does not bound it: the harness's own reference keeps
    the object alive, so the finaliser fires whenever that reference is dropped.
    On the worker path that used to be at function-scope exit -- after the
    response carrying ``flops_used`` had already been written, so the number
    reported could never account for it. Both paths must now release the
    reference before the window closes.
    """
    gc.collect()
    if path == "local":
        row = evaluate_estimator(_FinaliserEstimator(), _contest(10**12))["per_mlp"][0]
        billed = row["flops_used"]
    else:
        sent: dict = {}
        monkeypatch.setattr(subprocess_worker, "_write_response", lambda p: sent.update(p))
        mlp = _contest(10**12).mlps[0]
        subprocess_worker._handle_predict(
            _FinaliserEstimator(),
            {
                "mlp": {
                    "width": _W,
                    "depth": _D,
                    "seed": 0,
                    "weights": [np.asarray(w).tolist() for w in mlp.weights],
                },
                "budget": 10**12,
            },
        )
        gc.collect()
        billed = sent["flops_used"]

    assert billed >= _DEFERRED_FLOPS * 0.9, (
        f"{path}: finaliser work was not billed ({billed:,} FLOPs); the object "
        f"outlived the metered window"
    )


# --- the harness must not bill its own checks --------------------------------


def test_zero_flop_estimator_is_billed_zero_on_the_local_path():
    """An honest estimator pays nothing for the grader's bookkeeping.

    ``validate_predictions`` scans for non-finite values using counted ops, so
    pulling it inside the window would bill ``2 * numel - 1`` -- enough to tip a
    budget-tight submission into ``budget_exhausted`` with an error naming an op
    it never called.
    """
    row = evaluate_estimator(_ZeroFlopEstimator(), _contest(10_000))["per_mlp"][0]
    assert row["flops_used"] == 0
    assert not row.get("budget_exhausted")


def test_zero_flop_estimator_is_billed_zero_in_the_worker(monkeypatch):
    sent: dict = {}
    monkeypatch.setattr(subprocess_worker, "_write_response", lambda p: sent.update(p))
    mlp = _contest(10_000).mlps[0]
    request = {
        "mlp": {
            "width": _W,
            "depth": _D,
            "seed": 0,
            "weights": [np.asarray(w).tolist() for w in mlp.weights],
        },
        "budget": 10_000,
    }
    subprocess_worker._handle_predict(_ZeroFlopEstimator(), request)
    assert sent["status"] == "ok", sent.get("error_message")
    assert sent["flops_used"] == 0


def test_materialise_is_free():
    # This is what lets materialisation happen inside the window at all.
    arr = fnp.asarray(np.zeros((_D, _W), dtype=np.float32))
    ctx = flopscope.BudgetContext(10**12, quiet=True)
    with ctx:
        materialise_predictions(arr, depth=_D, width=_W)
    assert ctx.flops_used == 0


def test_validation_is_metered_and_therefore_belongs_outside():
    # Documents WHY validation must stay out of the window; if this ever becomes
    # free, the split is no longer load-bearing and can be simplified.
    arr = fnp.asarray(np.zeros((_D, _W), dtype=np.float32))
    ctx = flopscope.BudgetContext(10**12, quiet=True)
    with ctx:
        validate_predictions(arr, depth=_D, width=_W)
    assert ctx.flops_used == 2 * _D * _W - 1


# --- what is validated must be what is scored --------------------------------


def test_validation_returns_the_coerced_array_not_the_callers_object():
    lazy = _Deferred(_D, _W)
    result = validate_predictions(lazy, depth=_D, width=_W)
    assert result is not lazy, (
        "returning the caller's object lets a later consumer re-materialise it "
        "and observe values that never passed the finiteness gate"
    )
    assert isinstance(result, fnp.ndarray)


def test_a_value_that_changes_per_materialisation_cannot_change_after_validation():
    shifty = _Shifty(_D, _W)
    validated = validate_predictions(shifty, depth=_D, width=_W)
    first = np.asarray(validated).copy()
    np.asarray(validated)  # any later consumer
    assert np.array_equal(np.asarray(validated), first)


# --- error quality is preserved by the split ---------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param({"means": [1.0, 2.0]}, id="dict"),
        pytest.param([[1.0, 2.0], [3.0]], id="ragged"),
    ],
)
@pytest.mark.parametrize("fn", [materialise_predictions, validate_predictions])
def test_non_coercible_returns_keep_their_structured_error(fn, bad):
    """Coercing before the shape check would surface a bare TypeError instead.

    The shape check runs first precisely so the participant still gets
    ``details``/``cause_hints`` rather than a numpy coercion message.
    """
    with pytest.raises(ValueError) as excinfo:
        fn(bad, depth=_D, width=_W)
    assert isinstance(getattr(excinfo.value, "details", None), dict)


def test_materialise_does_not_run_the_finiteness_check():
    # Non-finite values are validation's job, not materialisation's -- otherwise
    # the metered scan sneaks back inside the window.
    nan = fnp.asarray(np.full((_D, _W), np.nan, dtype=np.float32))
    materialise_predictions(nan, depth=_D, width=_W)
    with pytest.raises(ValueError, match="finite"):
        validate_predictions(nan, depth=_D, width=_W)
