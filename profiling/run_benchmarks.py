#!/usr/bin/env python3
"""Orchestrator for running profiling tasks across Fargate configs.

Launches parallel ECS Fargate tasks, one per instance matrix config,
monitors their progress, and reports status.

Usage:
    python profiling/run_benchmarks.py
    python profiling/run_benchmarks.py --preset super-quick --configs compute-small
    python profiling/run_benchmarks.py --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from profiling.instance_matrix import get_configs
from profiling.run_helpers import generate_run_id, git_metadata, load_infra_config


def aws_cli(args: List[str], region: str) -> Dict[str, Any]:
    """Run an AWS CLI command and return parsed JSON output."""
    cmd = ["aws"] + args + ["--region", region, "--output", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"AWS CLI error: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"AWS CLI failed: {' '.join(cmd)}")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def register_task_definition(
    config: Dict[str, Any],
    infra: Dict[str, Any],
    preset: str,
    run_id: str,
    backends: Optional[str],
    max_threads: Optional[int],
    verbose: bool = False,
    timeout_minutes: Optional[int] = None,
) -> str:
    """Register a Fargate task definition for a given config. Returns task def ARN."""
    family = f"whest-profiling-{config['name']}"

    env_vars = [
        {"name": "RUN_ID", "value": run_id},
        {"name": "CONFIG_NAME", "value": config["name"]},
        {"name": "S3_BUCKET", "value": infra["s3_bucket"]},
        {"name": "PRESET", "value": preset},
    ]
    if backends:
        env_vars.append({"name": "BACKENDS", "value": backends})
    # Derive thread count from Fargate CPU units (1024 = 1 vCPU) unless
    # explicitly overridden.  nproc inside Fargate containers often
    # reports 1 regardless of allocation, so we must pass this explicitly.
    effective_threads = max_threads if max_threads is not None else config["cpu"] // 1024
    env_vars.append({"name": "MAX_THREADS", "value": str(effective_threads)})
    if verbose:
        env_vars.append({"name": "VERBOSE", "value": "1"})
    if timeout_minutes is not None:
        env_vars.append({"name": "TIMEOUT_MINUTES", "value": str(timeout_minutes)})

    task_def = {
        "family": family,
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": str(config["cpu"]),
        "memory": str(config["memory"]),
        "executionRoleArn": infra["execution_role_arn"],
        "taskRoleArn": infra["task_role_arn"],
        "containerDefinitions": [
            {
                "name": "profiler",
                "image": infra["image_uri"],
                "essential": True,
                "environment": env_vars,
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": infra["log_group"],
                        "awslogs-region": infra["region"],
                        "awslogs-stream-prefix": run_id,
                    },
                },
            }
        ],
    }

    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(task_def, f)
        tmp_path = f.name

    result = aws_cli(
        ["ecs", "register-task-definition", "--cli-input-json", f"file://{tmp_path}"],
        infra["region"],
    )
    os.unlink(tmp_path)

    return result["taskDefinition"]["taskDefinitionArn"]


def launch_task(
    task_def_arn: str,
    infra: Dict[str, Any],
) -> str:
    """Launch a Fargate task. Returns task ARN."""
    subnets_result = aws_cli(
        [
            "ec2",
            "describe-subnets",
            "--filters",
            "Name=default-for-az,Values=true",
            "--query",
            "Subnets[].SubnetId",
        ],
        infra["region"],
    )
    subnets = subnets_result if isinstance(subnets_result, list) else []
    if not subnets:
        raise RuntimeError("No default subnets found. Ensure a default VPC exists.")

    network_config = {
        "awsvpcConfiguration": {
            "subnets": subnets[:3],
            "assignPublicIp": "ENABLED",
        }
    }

    result = aws_cli(
        [
            "ecs",
            "run-task",
            "--cluster",
            infra["cluster_name"],
            "--task-definition",
            task_def_arn,
            "--launch-type",
            "FARGATE",
            "--network-configuration",
            json.dumps(network_config),
        ],
        infra["region"],
    )

    tasks = result.get("tasks", [])
    if not tasks:
        failures = result.get("failures", [])
        reason = failures[0]["reason"] if failures else "unknown"
        raise RuntimeError(f"Failed to launch task: {reason}")

    return tasks[0]["taskArn"]


def monitor_tasks(
    task_arns: Dict[str, str],
    infra: Dict[str, Any],
    timeout_minutes: int,
) -> Dict[str, str]:
    """Poll task status until all stopped or timeout. Returns final statuses."""
    start = time.time()
    timeout_secs = timeout_minutes * 60
    cluster = infra["cluster_name"]
    region = infra["region"]

    task_start_times = {name: start for name in task_arns}
    final_statuses: Dict[str, str] = {}

    while True:
        elapsed = time.time() - start
        if elapsed > timeout_secs:
            print(f"\nTimeout after {timeout_minutes} minutes. Stopping remaining tasks...")
            for name, arn in task_arns.items():
                if name not in final_statuses:
                    subprocess.run(
                        [
                            "aws",
                            "ecs",
                            "stop-task",
                            "--cluster",
                            cluster,
                            "--task",
                            arn,
                            "--region",
                            region,
                        ],
                        capture_output=True,
                    )
                    final_statuses[name] = "TIMEOUT"
            break

        pending_arns = {n: a for n, a in task_arns.items() if n not in final_statuses}
        if not pending_arns:
            break

        arn_list = list(pending_arns.values())
        result = aws_cli(
            ["ecs", "describe-tasks", "--cluster", cluster, "--tasks"] + arn_list,
            region,
        )

        arn_to_name = {a: n for n, a in pending_arns.items()}

        now = time.time()
        if sys.stdout.isatty():
            print(f"\r\033[{len(task_arns) + 1}A", end="")
        print(f"{'Config':<25} {'Status':<12} {'Duration':<12}")
        for name in task_arns:
            if name in final_statuses:
                status = final_statuses[name]
                duration = ""
                marker = " ✓" if status == "STOPPED_OK" else " ✗"
            else:
                task_info = next(
                    (t for t in result.get("tasks", []) if arn_to_name.get(t["taskArn"]) == name),
                    None,
                )
                if task_info:
                    status = task_info.get("lastStatus", "UNKNOWN")
                    dur_secs = int(now - task_start_times[name])
                    minutes, secs = divmod(dur_secs, 60)
                    duration = f"{minutes}m {secs:02d}s"

                    if status == "STOPPED":
                        containers = task_info.get("containers", [])
                        exit_code = containers[0].get("exitCode", -1) if containers else -1
                        if exit_code == 0:
                            final_statuses[name] = "STOPPED_OK"
                            marker = " ✓"
                        else:
                            final_statuses[name] = f"FAILED(exit={exit_code})"
                            marker = " ✗"
                    else:
                        marker = ""
                else:
                    status = "UNKNOWN"
                    duration = ""
                    marker = ""
            print(f"  {name:<23} {status:<12} {duration}{marker}")

        time.sleep(10)

    return final_statuses


def print_dry_run(configs, preset, run_id, backends, max_threads, infra):
    """Print what would be launched without actually launching."""
    print("=== DRY RUN ===")
    print(f"Run ID:    {run_id}")
    print(f"Preset:    {preset}")
    print(f"Backends:  {backends or 'all'}")
    print(f"Threads:   {max_threads or 'unlimited'}")
    print(f"Cluster:   {infra['cluster_name']}")
    print(f"Image:     {infra.get('image_uri', 'NOT SET')}")
    print(f"S3 bucket: {infra['s3_bucket']}")
    print("")
    print(f"Would launch {len(configs)} Fargate tasks:")
    for c in configs:
        print(f"  {c['name']:<25} {c['label']}")
    print("")
    print("S3 output keys:")
    for c in configs:
        print(f"  s3://{infra['s3_bucket']}/{run_id}/{c['name']}.json")


def main():
    parser = argparse.ArgumentParser(
        description="Run whest profiler across Fargate instance configs.",
    )
    parser.add_argument(
        "--preset",
        default="exhaustive",
        choices=["super-quick", "quick", "standard", "exhaustive"],
        help="Profiler preset (default: exhaustive)",
    )
    parser.add_argument(
        "--configs",
        help="Comma-separated config names to run (default: all)",
    )
    parser.add_argument(
        "--run-id",
        help="Custom run ID (default: auto-generated from timestamp + git hash)",
    )
    parser.add_argument(
        "--backends",
        help="Comma-separated backend filter (passed to whest)",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        help="Thread cap (passed to whest --max-threads)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=480,
        help="Timeout in minutes for all tasks (default: 480 = 8 hours)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass --verbose to whest profiler (more detailed output in logs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be launched without launching",
    )

    args = parser.parse_args()

    config_names = args.configs.split(",") if args.configs else None
    configs = get_configs(names=config_names)

    run_id = generate_run_id(override=args.run_id)
    git_info = git_metadata()

    if git_info["dirty"] and not args.run_id:
        print(f"WARNING: Working tree is dirty. Run ID: {run_id}", file=sys.stderr)

    infra = load_infra_config()

    if args.dry_run:
        print_dry_run(configs, args.preset, run_id, args.backends, args.max_threads, infra)
        return

    print("=== Launching Cloud Profiling Run ===")
    print(f"Run ID:    {run_id}")
    print(f"Git:       {git_info['commit_short']}{' (dirty)' if git_info['dirty'] else ''}")
    print(f"Preset:    {args.preset}")
    print(f"Configs:   {len(configs)}")
    print(f"Timeout:   {args.timeout} minutes")
    print("")

    task_arns: Dict[str, str] = {}
    for i, config in enumerate(configs):
        print(f"Registering + launching: {config['name']} ({config['label']})...")
        task_def_arn = register_task_definition(
            config,
            infra,
            args.preset,
            run_id,
            args.backends,
            args.max_threads,
            verbose=args.verbose,
            timeout_minutes=args.timeout,
        )
        task_arn = launch_task(task_def_arn, infra)
        task_arns[config["name"]] = task_arn

        if i < len(configs) - 1:
            time.sleep(1)

    print(f"\nAll {len(task_arns)} tasks launched. Monitoring...\n")
    for _ in range(len(task_arns) + 1):
        print()

    final = monitor_tasks(task_arns, infra, args.timeout)

    print("\n=== Run Complete ===")
    ok = sum(1 for s in final.values() if s == "STOPPED_OK")
    failed = len(final) - ok
    print(f"Succeeded: {ok}/{len(final)}")
    if failed:
        print(f"Failed:    {failed}")
        for name, status in final.items():
            if status != "STOPPED_OK":
                print(f"  {name}: {status}")
        print("\n--- CloudWatch logs for failed tasks ---")
        for name, status in final.items():
            if status != "STOPPED_OK":
                print(f"\n[{name}]:")
                subprocess.run(
                    [
                        "aws",
                        "logs",
                        "tail",
                        infra["log_group"],
                        "--log-stream-name-prefix",
                        f"{run_id}",
                        "--since",
                        "2h",
                        "--region",
                        infra["region"],
                    ],
                    timeout=10,
                )
                print()

    print("\nCollect results with:")
    print(f"  python profiling/collect_results.py --run-id {run_id}")


if __name__ == "__main__":
    main()
