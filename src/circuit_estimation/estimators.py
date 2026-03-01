"""Legacy estimator module retained only as migration guidance.

Participant-facing starter estimators now live under `examples/estimators/` and
must implement class-based `Estimator(BaseEstimator)` entrypoints loaded through
`load_estimator_from_path(...)`.

Function-style estimator callables are intentionally not exposed from this
module after the full migration.
"""

__all__: list[str] = []
