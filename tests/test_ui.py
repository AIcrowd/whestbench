"""Unit tests for whestbench.ui — shared UX helpers."""

from __future__ import annotations

import pytest

from whestbench.ui import format_bytes


@pytest.mark.parametrize(
    "n_bytes,expected",
    [
        (0, "0 B"),
        (1, "1 B"),
        (1023, "1023 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1_048_576, "1.0 MB"),
        (1_572_864, "1.5 MB"),
        (1_073_741_824, "1.0 GB"),
        (2_118_949_161, "2.0 GB"),
        (1_099_511_627_776, "1.0 TB"),
    ],
)
def test_format_bytes(n_bytes: int, expected: str) -> None:
    assert format_bytes(n_bytes) == expected


def test_format_bytes_negative_raises() -> None:
    with pytest.raises(ValueError):
        format_bytes(-1)
