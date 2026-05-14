"""Memory-limit enforcement in subprocess worker."""

from __future__ import annotations

import logging
import sys
import textwrap

import pytest


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="setrlimit(RLIMIT_AS) is a no-op on macOS; enforcement is Linux-only",
)
def test_subprocess_worker_sets_address_space_limit(tmp_path):
    """Worker must setrlimit(RLIMIT_AS, ...) before loading the estimator.

    The estimator reads back its own RLIMIT_AS via resource.getrlimit() and
    encodes the soft limit into the first cell of its prediction. The host then
    decodes and verifies the limit equals memory_limit_mb * 1024 * 1024.
    """
    import flopscope.numpy as fnp

    from whestbench.domain import MLP
    from whestbench.runner import (
        EstimatorEntrypoint,
        ResourceLimits,
        SubprocessRunner,
    )
    from whestbench.sdk import SetupContext

    memory_limit_mb = 1024  # 1 GiB; large enough to not collide with worker baseline allocations
    expected_bytes = memory_limit_mb * 1024 * 1024

    estimator_py = tmp_path / "limit_reading_estimator.py"
    estimator_py.write_text(
        textwrap.dedent("""
            import resource
            import flopscope.numpy as fnp
            from whestbench.sdk import BaseEstimator

            class LimitReadingEstimator(BaseEstimator):
                def predict(self, mlp, budget):
                    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                    out = fnp.zeros((mlp.depth, mlp.width), dtype=fnp.float32)
                    # Encode the soft limit (a large int) into the first two
                    # cells as float32-safe halves to round-trip the value.
                    # float32 only has 24 bits of mantissa — split into two 24-bit halves.
                    high = (soft >> 24) & 0xFFFFFF
                    low = soft & 0xFFFFFF
                    out_np = fnp.array([[float(high), float(low)] + [0.0] * (mlp.width - 2)] +
                                       [[0.0] * mlp.width] * (mlp.depth - 1), dtype=fnp.float32)
                    return out_np
        """)
    )

    runner = SubprocessRunner()
    limits = ResourceLimits(
        setup_timeout_s=5.0,
        predict_timeout_s=30.0,
        memory_limit_mb=memory_limit_mb,
        flop_budget=100_000_000,
    )
    context = SetupContext(width=4, depth=2, flop_budget=100_000_000, api_version="1.0")

    try:
        runner.start(
            EstimatorEntrypoint(file_path=estimator_py, class_name="LimitReadingEstimator"),
            context,
            limits,
        )
        weights = [fnp.array(fnp.zeros((4, 4), dtype=fnp.float32)) for _ in range(2)]
        mlp = MLP(width=4, depth=2, weights=weights)
        result = runner.predict(mlp, 100_000_000)
        # Decode the two-half encoding
        high = int(result[0][0])
        low = int(result[0][1])
        observed_soft_limit = (high << 24) | low
        assert observed_soft_limit == expected_bytes, (
            f"Worker did not apply RLIMIT_AS: expected soft={expected_bytes}, "
            f"got soft={observed_soft_limit}"
        )
    finally:
        runner.close()


def test_local_runner_warns_when_memory_limit_unsupported(caplog):
    """LocalRunner.start should emit a warning that memory_limit_mb is advisory."""
    import tempfile
    import textwrap as _tw
    from pathlib import Path

    from whestbench.runner import (
        EstimatorEntrypoint,
        LocalRunner,
        ResourceLimits,
    )
    from whestbench.sdk import SetupContext

    # Create a temp estimator file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            _tw.dedent("""
                import flopscope.numpy as fnp
                from whestbench.sdk import BaseEstimator
                class _E(BaseEstimator):
                    def predict(self, mlp, budget):
                        return fnp.zeros((mlp.depth, mlp.width))
            """)
        )
        est_path = Path(f.name)

    runner = LocalRunner()
    limits = ResourceLimits(
        setup_timeout_s=5.0,
        predict_timeout_s=30.0,
        memory_limit_mb=512,
        flop_budget=1_000_000,
    )
    context = SetupContext(width=4, depth=2, flop_budget=1_000_000, api_version="1.0")

    with caplog.at_level(logging.WARNING, logger="whestbench.runner"):
        runner.start(EstimatorEntrypoint(file_path=est_path, class_name="_E"), context, limits)

    assert any(
        "memory_limit_mb" in record.message and "local" in record.message.lower()
        for record in caplog.records
    ), f"Expected memory-limit warning; got: {[r.message for r in caplog.records]}"

    runner.close()
    est_path.unlink(missing_ok=True)
