#!/usr/bin/env python3
"""Pull profiling results from S3 and aggregate into combined reports.

Usage:
    python profiling/collect_results.py --run-id 2026-03-20-143000-abc1234
    python profiling/collect_results.py --run-id 2026-03-20-v1 --format json
    python profiling/collect_results.py --run-id 2026-03-20-v1 --output ./my-results/
"""

import argparse
import csv
import io
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from profiling.instance_matrix import INSTANCE_MATRIX
from profiling.run_helpers import load_infra_config


def build_combined_json(
    run_id: str,
    git_commit: str,
    git_dirty: bool,
    config_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the combined JSON structure from per-config results."""
    return {
        "run_id": run_id,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "configs": config_results,
    }


def build_summary_csv(config_results: Dict[str, Any]) -> str:
    """Build a flattened CSV from per-config timing results.

    Columns: config_name, cpu, memory, backend, operation, width, depth,
             n_samples, median_time, speedup_vs_numpy
    """
    config_lookup = {c["name"]: c for c in INSTANCE_MATRIX}

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "config_name",
            "cpu",
            "memory",
            "backend",
            "operation",
            "width",
            "depth",
            "n_samples",
            "median_time",
            "speedup_vs_numpy",
        ],
    )
    writer.writeheader()

    for config_name, result in sorted(config_results.items()):
        config_meta = config_lookup.get(config_name, {})
        for timing in result.get("timing", []):
            writer.writerow(
                {
                    "config_name": config_name,
                    "cpu": config_meta.get("cpu", ""),
                    "memory": config_meta.get("memory", ""),
                    "backend": timing.get("backend", ""),
                    "operation": timing.get("operation", ""),
                    "width": timing.get("width", ""),
                    "depth": timing.get("depth", ""),
                    "n_samples": timing.get("n_samples", ""),
                    "median_time": timing.get("median_time", ""),
                    "speedup_vs_numpy": timing.get("speedup_vs_numpy", ""),
                }
            )

    return output.getvalue()


def s3_list_objects(bucket: str, prefix: str, region: str) -> List[str]:
    """List S3 object keys under a prefix."""
    result = subprocess.run(
        [
            "aws",
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--query",
            "Contents[].Key",
            "--output",
            "json",
            "--region",
            region,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    keys = json.loads(result.stdout) if result.stdout.strip() else []
    return keys or []


def s3_download(bucket: str, key: str, dest: str, region: str) -> bool:
    """Download a single S3 object. Returns True on success."""
    result = subprocess.run(
        ["aws", "s3", "cp", f"s3://{bucket}/{key}", dest, "--region", region],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Collect and aggregate profiling results from S3.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID to collect results for",
    )
    parser.add_argument(
        "--output",
        default="profiling/results",
        help="Output directory (default: profiling/results/)",
    )
    parser.add_argument(
        "--format",
        default="json,csv",
        help="Output formats: json, csv, or both (default: json,csv)",
    )

    args = parser.parse_args()
    formats = [f.strip() for f in args.format.split(",")]

    infra = load_infra_config()
    bucket = infra["s3_bucket"]
    region = infra["region"]
    prefix = f"{args.run_id}/"

    output_dir = Path(args.output) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing results for run: {args.run_id}")
    keys = s3_list_objects(bucket, prefix, region)

    if not keys:
        print(f"No results found in s3://{bucket}/{prefix}")
        sys.exit(1)

    config_results: Dict[str, Any] = {}
    expected_configs = {c["name"] for c in INSTANCE_MATRIX}
    found_configs = set()

    for key in keys:
        config_name = key.split("/")[-1].replace(".json", "")
        dest = str(output_dir / f"{config_name}.json")
        print(f"  Downloading: {config_name}...")
        if s3_download(bucket, key, dest, region):
            with open(dest) as f:
                config_results[config_name] = json.load(f)
            found_configs.add(config_name)
        else:
            print(f"  WARNING: Failed to download {key}")

    missing = expected_configs - found_configs
    if missing:
        print(f"\nWARNING: Missing results for: {', '.join(sorted(missing))}")

    print(f"\nCollected {len(config_results)} / {len(expected_configs)} configs")

    # Extract git info from run ID if it matches auto-generated pattern
    git_commit = ""
    git_dirty = args.run_id.endswith("-dirty")
    clean_id = args.run_id[:-6] if args.run_id.endswith("-dirty") else args.run_id
    match = re.match(r"\d{4}-\d{2}-\d{2}-\d{6}-([a-f0-9]{7})$", clean_id)
    if match:
        git_commit = match.group(1)

    if "json" in formats:
        combined = build_combined_json(
            run_id=args.run_id,
            git_commit=git_commit,
            git_dirty=git_dirty,
            config_results=config_results,
        )
        combined_path = output_dir / "combined.json"
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"Combined JSON: {combined_path}")

    if "csv" in formats:
        csv_str = build_summary_csv(config_results)
        csv_path = output_dir / "summary.csv"
        with open(csv_path, "w") as f:
            f.write(csv_str)
        print(f"Summary CSV:   {csv_path}")

    print(f"\n{'Config':<25} {'Backends':<10} {'Fastest':<12}")
    print("-" * 47)
    for config_name in sorted(config_results):
        result = config_results[config_name]
        backends = {t["backend"] for t in result.get("timing", [])}
        run_mlp_times = [t for t in result.get("timing", []) if t.get("operation") == "run_mlp"]
        if run_mlp_times:
            fastest = min(run_mlp_times, key=lambda t: t["median_time"])
            fastest_name = fastest["backend"]
        else:
            fastest_name = "—"
        print(f"  {config_name:<23} {len(backends):<10} {fastest_name}")


if __name__ == "__main__":
    main()
