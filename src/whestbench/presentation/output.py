from __future__ import annotations

import argparse
from typing import Literal

OutputFormat = Literal["rich", "plain", "json"]


def add_output_format_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("rich", "plain", "json"),
        default=None,
        help="Select output format: rich, plain, or json.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Alias for --format json.",
    )


def resolve_output_format(
    format_arg: str | None,
    json_output: bool,
    is_tty: bool,
) -> OutputFormat:
    if json_output:
        return "json"
    if format_arg == "rich" or format_arg == "plain" or format_arg == "json":
        return format_arg
    return "rich" if is_tty else "plain"
