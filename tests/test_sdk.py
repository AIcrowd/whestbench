import numpy as np
import pytest

from network_estimation.domain import MLP
from network_estimation.sdk import BaseEstimator, SetupContext


def test_setup_context_fields() -> None:
    ctx = SetupContext(width=256, depth=16, estimator_budget=1000, api_version="1.0")
    assert ctx.width == 256
    assert ctx.depth == 16
    assert ctx.estimator_budget == 1000
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
        def predict(self, mlp: MLP, budget: int) -> np.ndarray:
            return np.zeros((mlp.depth, mlp.width), dtype=np.float32)

    est = MinimalEstimator()
    ctx = SetupContext(width=4, depth=2, estimator_budget=100, api_version="1.0")
    est.setup(ctx)  # should not raise
    est.teardown()  # should not raise
