"""Generate self-contained HTML profiling dashboard."""
import argparse
import json
import os
import sys


def load_data(path):
    """Load profiling JSON from file path."""
    with open(path) as f:
        return json.load(f)


def normalize_data(data):
    """Normalize single-config or multi-config data into multi-config format."""
    if "configs" in data:
        return data
    hostname = data.get("hardware", {}).get("hostname", "local")
    config_name = hostname if hostname != "local" else "local"
    return {
        "run_id": f"local-{config_name}",
        "git_commit": "",
        "git_dirty": False,
        "collected_at": "",
        "configs": {config_name: data},
    }


def parse_args(argv=None):
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate profiling dashboard HTML")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="Load from profiling/results/{run-id}/combined.json")
    group.add_argument("--input", help="Path to combined.json or single-config output.json")
    parser.add_argument("--output", help="Output HTML path")
    return parser.parse_args(argv)


def resolve_paths(args):
    """Resolve input/output file paths from parsed args."""
    if args.run_id:
        input_path = os.path.join("profiling", "results", args.run_id, "combined.json")
        default_output = os.path.join("profiling", "results", args.run_id, "dashboard.html")
    else:
        input_path = args.input
        default_output = "dashboard.html"
    output_path = args.output or default_output
    return input_path, output_path
