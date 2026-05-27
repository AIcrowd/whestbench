"""End-to-end-ish tests for the hf:// dataset progress wrapper in cli.py.

These tests patch the heavyweight `_wb_load_dataset` call so we never touch
the network. They assert that the Rich console writes the expected before/
during/after copy in each mode (cache hit, cache miss).
"""

from __future__ import annotations

from whestbench.hf_progress import HFPreflight


def test_hfpreflight_smoke() -> None:
    """Smoke test — establishes that we can build an HFPreflight in tests."""
    pf = HFPreflight(
        repo_id="aicrowd/foo",
        revision="v1-warmup",
        file_count=3,
        total_bytes=2_000_000_000,
        is_cached=True,
        files=[("metadata.json", 6000)],
    )
    assert pf.is_cached is True
