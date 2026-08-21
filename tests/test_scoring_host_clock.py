"""The per-MLP wall clock must measure participant work, not harness transport.

Under an out-of-process runner the host wraps the whole IPC round-trip — request
serialization, pipe write, wait, response decode — in a BudgetContext carrying the
per-MLP wall limit, while the worker times only the participant's own predict(). The
flag was then decided on the host clock and the number reported from the worker's,
so a row could read ``time_exhausted: true`` with a ``wall_time_s`` below the limit
it was measured against (whestbench#129).
"""

from __future__ import annotations

import time

import flopscope.numpy as fnp

from whestbench.runner import LocalRunner, SubprocessRunner
from whestbench.scoring import ContestSpec, evaluate_estimator, make_contest
from whestbench.sdk import BaseEstimator

_LIMIT_S = 0.2
_OVER_LIMIT_S = 0.5


def _spec() -> ContestSpec:
    return ContestSpec(
        width=8,
        depth=2,
        n_mlps=1,
        flop_budget=100_000_000,
        ground_truth_samples=200,
        wall_time_limit_s=_LIMIT_S,
    )


class _OutOfProcessEstimator(BaseEstimator):
    """Stands in for a runner-wrapped estimator: the time the host observes is
    dominated by transport, and the worker reports its own (much smaller) number."""

    enforces_wall_time_limit = True

    def __init__(self, reported_wall_time_s: float) -> None:
        self._reported = reported_wall_time_s

    def predict(self, mlp, budget):
        # Transport cost the participant did not incur.
        time.sleep(_OVER_LIMIT_S)
        # runner.py decodes the response with a metered op inside the host context;
        # that is where flopscope's cooperative deadline check runs.
        return fnp.asarray(
            [[0.0] * mlp.width for _ in range(mlp.depth)],
            dtype=fnp.float32,
        )

    def last_predict_stats(self):
        # Billed to backend time, not residual: residual is charged at lambda into the
        # combined budget, which would trip a *different* exhaustion flag and stop
        # these tests from isolating the wall clock. Keeps the documented identity
        # wall = backend + overhead + residual true either way.
        return {
            "flops_used": 0,
            "wall_time_s": self._reported,
            "flopscope_backend_time_s": self._reported,
            "flopscope_overhead_time_s": 0.0,
            "residual_wall_time_s": 0.0,
            "budget_breakdown": None,
        }


def test_transport_time_does_not_exhaust_the_per_mlp_wall() -> None:
    # The worker says the participant used 0.01s against a 0.2s limit. Whatever the
    # host spent shuttling bytes is not the participant's to pay for.
    estimator = _OutOfProcessEstimator(reported_wall_time_s=0.01)

    result = evaluate_estimator(estimator, make_contest(_spec()))

    assert result["per_mlp"][0].get("time_exhausted") is False


def test_a_row_is_never_flagged_below_its_own_reported_wall_time() -> None:
    # The self-contradictory row from the report: flagged exhausted, yet its own
    # wall_time_s sits under the limit. Whatever the mechanism, that must not happen.
    estimator = _OutOfProcessEstimator(reported_wall_time_s=0.01)

    row = evaluate_estimator(estimator, make_contest(_spec()))["per_mlp"][0]

    assert not (row.get("time_exhausted") and row["wall_time_s"] < _LIMIT_S)


def test_worker_reported_overrun_still_exhausts_the_wall() -> None:
    # The backstop has to survive the fix: when the participant's OWN time is over
    # the limit, the MLP is still flagged.
    estimator = _OutOfProcessEstimator(reported_wall_time_s=_OVER_LIMIT_S)

    result = evaluate_estimator(estimator, make_contest(_spec()))

    assert result["per_mlp"][0].get("time_exhausted") is True


def test_in_process_estimator_is_still_held_to_the_wall() -> None:
    # No out-of-process runner, so the host context IS the participant's clock and
    # must keep enforcing the limit.
    class SlowEstimator(BaseEstimator):
        def predict(self, mlp, budget):
            time.sleep(_OVER_LIMIT_S)
            return fnp.zeros((mlp.depth, mlp.width), dtype=fnp.float32)

    result = evaluate_estimator(SlowEstimator(), make_contest(_spec()))

    assert result["per_mlp"][0].get("time_exhausted") is True


def test_subprocess_runner_declares_it_enforces_the_wall_itself() -> None:
    assert SubprocessRunner.enforces_wall_time_limit is True


def test_local_runner_leaves_wall_enforcement_to_the_host() -> None:
    assert LocalRunner.enforces_wall_time_limit is False
