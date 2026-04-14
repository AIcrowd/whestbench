import pytest
import whest as we

from whestbench.domain import MLP
from whestbench.sdk import BaseEstimator, SetupContext


def test_setup_context_fields() -> None:
    ctx = SetupContext(width=256, depth=16, flop_budget=1000, api_version="1.0")
    assert ctx.width == 256
    assert ctx.depth == 16
    assert ctx.flop_budget == 1000
    assert ctx.api_version == "1.0"
    assert ctx.scratch_dir is None


def test_base_estimator_requires_predict() -> None:
    """Subclass must implement predict."""

    class IncompleteEstimator(BaseEstimator):
        pass

    with pytest.raises(TypeError):
        IncompleteEstimator()


def test_base_estimator_default_setup_teardown() -> None:
    """setup and teardown should be callable without error."""

    class MinimalEstimator(BaseEstimator):
        def predict(self, mlp: MLP, budget: int) -> we.ndarray:
            return we.zeros((mlp.depth, mlp.width))

    est = MinimalEstimator()
    ctx = SetupContext(width=4, depth=2, flop_budget=100, api_version="1.0")
    est.setup(ctx)  # should not raise
    est.teardown()  # should not raise
