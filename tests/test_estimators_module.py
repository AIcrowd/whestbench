from __future__ import annotations

import circuit_estimation.estimators as estimators_module


def test_estimators_module_documents_migration_to_examples() -> None:
    assert estimators_module.__doc__ is not None
    lowered = estimators_module.__doc__.lower()
    assert "examples/estimators" in lowered
    assert "class-based" in lowered


def test_no_function_style_estimator_entrypoints_exposed() -> None:
    for name in ("mean_propagation", "covariance_propagation", "combined_estimator"):
        assert not hasattr(estimators_module, name)
