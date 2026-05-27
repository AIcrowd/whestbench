"""Unit tests for whestbench.ui — shared UX helpers."""

from __future__ import annotations

import pytest

from whestbench.ui import format_bytes, format_duration, format_throughput


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


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.0, "0ms"),
        (0.05, "50ms"),
        (0.999, "999ms"),
        (1.0, "1.0s"),
        (2.137, "2.1s"),
        (31.7, "31.7s"),
        (59.9, "59.9s"),
        (60.0, "1m 0s"),
        (125.0, "2m 5s"),
        (3599.0, "59m 59s"),
        (3600.0, "1h 0m 0s"),
        (7325.0, "2h 2m 5s"),
    ],
)
def test_format_duration(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected


def test_format_duration_negative_raises() -> None:
    with pytest.raises(ValueError):
        format_duration(-0.1)


@pytest.mark.parametrize(
    "n_bytes,seconds,expected",
    [
        (2_000_000_000, 30.0, "63.6 MB/s"),
        (1_048_576, 1.0, "1.0 MB/s"),
        (512, 1.0, "512 B/s"),
        (1_073_741_824, 1.0, "1.0 GB/s"),
    ],
)
def test_format_throughput(n_bytes: int, seconds: float, expected: str) -> None:
    assert format_throughput(n_bytes, seconds) == expected


def test_format_throughput_zero_seconds_returns_dash() -> None:
    assert format_throughput(1024, 0.0) == "— /s"
