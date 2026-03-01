from __future__ import annotations

from typing import Any

import circuit_estimation.cli as cli


def test_run_default_score_returns_float(monkeypatch) -> None:
    # Default path should remain a simple numeric score for starter-kit users.
    def fake_score_estimator(*_args: Any, **_kwargs: Any) -> float:
        return 0.5

    monkeypatch.setattr(cli, "score_estimator", fake_score_estimator)
    score = cli.run_default_score()
    assert isinstance(score, float)


def test_run_default_score_with_profile_returns_diagnostics(monkeypatch) -> None:
    # Profile mode should surface structured diagnostics for future perf tooling.
    def fake_score_estimator(*_args: Any, **kwargs: Any) -> float:
        profiler = kwargs.get("profiler")
        if profiler is not None:
            profiler(
                {
                    "wall_time_s": 0.01,
                    "cpu_time_s": 0.01,
                    "rss_bytes": 1024,
                    "peak_rss_bytes": 2048,
                }
            )
        return 0.7

    monkeypatch.setattr(cli, "score_estimator", fake_score_estimator)
    score, profile = cli.run_default_score(profile=True)

    assert isinstance(score, float)
    assert isinstance(profile, list)
    assert profile
    assert {"wall_time_s", "cpu_time_s", "rss_bytes", "peak_rss_bytes"} <= set(profile[0].keys())
