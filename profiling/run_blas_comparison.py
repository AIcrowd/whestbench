#!/usr/bin/env python3
"""Launch parallel A/B benchmark: PyTorch MKL vs OpenBLAS on Fargate.

Launches two identical Fargate tasks (same vCPU/memory) with different
Docker images — one using pip torch (MKL) and one using source-built
torch (OpenBLAS). Results are uploaded to S3 under separate config names
so they can be compared in the dashboard.

Usage:
    PYTHONPATH=. python profiling/run_blas_comparison.py [--vcpus 8] [--preset quick]
"""
from __future__ import annotations

import argparse
import json
import time

import boto3

from profiling.instance_matrix import get_configs


def main():
    parser = argparse.ArgumentParser(description="A/B: MKL vs OpenBLAS")
    parser.add_argument("--vcpus", type=int, default=8, help="vCPU tier (1,2,4,8,16)")
    parser.add_argument("--preset", default="quick", help="Profiling preset")
    parser.add_argument("--backends", default="numpy,pytorch", help="Backends to benchmark")
    args = parser.parse_args()

    with open("profiling/.infra-config.json") as f:
        infra = json.load(f)

    region = infra["region"]
    ecr_repo = infra["ecr_repo_uri"]
    cluster = infra["cluster_name"]
    s3_bucket = infra["s3_bucket"]
    exec_role = infra["execution_role_arn"]
    task_role = infra["task_role_arn"]
    log_group = infra["log_group"]

    # Find the matching config
    configs = get_configs()
    config_name = f"compute-{args.vcpus}vcpu"
    if config_name not in configs:
        print(f"ERROR: No config for {args.vcpus} vCPUs. Available: {list(configs.keys())}")
        return

    config = configs[config_name]
    cpu_units = config["cpu"]
    memory = config["memory"]
    max_threads = cpu_units // 1024

    run_id = f"blas-ab-{time.strftime('%Y%m%d-%H%M%S')}"

    ecs = boto3.client("ecs", region_name=region)
    ec2 = boto3.client("ec2", region_name=region)

    subnets = [
        s["SubnetId"]
        for s in ec2.describe_subnets(
            Filters=[{"Name": "default-for-az", "Values": ["true"]}]
        )["Subnets"]
    ][:3]

    variants = [
        ("mkl", f"{ecr_repo}:mkl"),
        ("openblas", f"{ecr_repo}:openblas"),
    ]

    print(f"=== BLAS A/B Comparison ===")
    print(f"Run ID:   {run_id}")
    print(f"Config:   {config_name} ({max_threads} threads)")
    print(f"Preset:   {args.preset}")
    print(f"Backends: {args.backends}")
    print()

    task_arns = {}
    for variant_name, image_uri in variants:
        ab_config_name = f"{config_name}-{variant_name}"

        task_def = {
            "family": f"nestim-blas-{variant_name}",
            "networkMode": "awsvpc",
            "requiresCompatibilities": ["FARGATE"],
            "cpu": str(cpu_units),
            "memory": str(memory),
            "executionRoleArn": exec_role,
            "taskRoleArn": task_role,
            "containerDefinitions": [
                {
                    "name": "profiler",
                    "image": image_uri,
                    "essential": True,
                    "environment": [
                        {"name": "RUN_ID", "value": run_id},
                        {"name": "CONFIG_NAME", "value": ab_config_name},
                        {"name": "S3_BUCKET", "value": s3_bucket},
                        {"name": "PRESET", "value": args.preset},
                        {"name": "BACKENDS", "value": args.backends},
                        {"name": "MAX_THREADS", "value": str(max_threads)},
                        {"name": "TIMEOUT_MINUTES", "value": "120"},
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": log_group,
                            "awslogs-region": region,
                            "awslogs-stream-prefix": run_id,
                        },
                    },
                }
            ],
        }

        reg = ecs.register_task_definition(**task_def)
        td_arn = reg["taskDefinition"]["taskDefinitionArn"]

        resp = ecs.run_task(
            cluster=cluster,
            taskDefinition=td_arn,
            launchType="FARGATE",
            count=1,
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": subnets,
                    "assignPublicIp": "ENABLED",
                }
            },
        )

        task_arn = resp["tasks"][0]["taskArn"]
        task_id = task_arn.split("/")[-1]
        task_arns[variant_name] = task_id
        print(f"  Launched {variant_name:10s} → {task_id}")

    print()
    print(f"S3 output: s3://{s3_bucket}/{run_id}/")
    print()
    print("Monitor with:")
    print(f"  aws s3 ls s3://{s3_bucket}/{run_id}/ --region {region}")
    for variant_name, task_id in task_arns.items():
        print(f"  # {variant_name} logs:")
        print(f"  aws logs tail /ecs/nestim-profiling --log-stream-name-prefix {run_id}/profiler/{task_id} --follow --region {region}")
    print()
    print("Compare results after completion:")
    print(f"  PYTHONPATH=. python profiling/collect_results.py --run-id {run_id}")


if __name__ == "__main__":
    main()
