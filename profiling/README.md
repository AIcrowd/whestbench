# Cloud Profiling Infrastructure

Run `nestim profile-simulation` benchmarks across a matrix of AWS Fargate
CPU/memory configurations, collect results to S3, and aggregate locally.

## Prerequisites

- **AWS CLI v2** — configured with valid credentials (`aws configure`)
- **Docker** — for building the profiler image
- **Python 3.10+** — for the orchestrator and collector scripts
- **Fargate quota** — default is 6 vCPU on-demand per region. Request an increase
  via AWS Service Quotas if running the full matrix (which needs ~60 concurrent vCPUs).

## Quick Start

```bash
# 1. Create AWS infrastructure (S3, ECR, ECS, IAM)
bash profiling/setup_infra.sh

# 2. Build and push the Docker image
bash profiling/build_and_push.sh

# 3. Run benchmarks (all 9 configs, exhaustive preset)
python profiling/run_benchmarks.py

# 4. Collect and aggregate results
python profiling/collect_results.py --run-id <run-id-from-step-3>
```

## Infrastructure

### Setup

```bash
# Defaults to us-east-1. Override with AWS_REGION:
AWS_REGION=eu-west-1 bash profiling/setup_infra.sh
```

Creates these resources:

| Resource | Name | Purpose |
|----------|------|---------|
| S3 bucket | `nestim-profiling-{account-id}` | Result storage |
| ECR repository | `nestim-profiler` | Docker image registry |
| ECS cluster | `nestim-profiling` | Fargate task execution |
| IAM execution role | `nestim-profiler-execution` | ECR pull + CloudWatch |
| IAM task role | `nestim-profiler-task` | S3 upload (scoped) |
| CloudWatch log group | `/ecs/nestim-profiling` | Task logs |

Resource ARNs are written to `profiling/.infra-config.json` (gitignored).

### Teardown

```bash
bash profiling/teardown_infra.sh
```

Prompts for confirmation. Deletes all resources including S3 contents.

## Building the Docker Image

```bash
bash profiling/build_and_push.sh
```

Rebuild after any code changes to `src/` or backend dependencies. The image includes
all 6 CPU backends (NumPy, SciPy, PyTorch CPU, Numba, JAX CPU, Cython) with pre-warmed
JIT caches.

## Running Benchmarks

### Full run (default)

```bash
python profiling/run_benchmarks.py
```

Launches all 9 Fargate configs with `--preset exhaustive`. Shows a live status table.

### Debug run

```bash
python profiling/run_benchmarks.py --preset super-quick --configs compute-small
```

Fast iteration: single small config, minimal profiling.

### All options

```
python profiling/run_benchmarks.py \
    --preset exhaustive              # super-quick|quick|standard|exhaustive
    --configs compute-small,general-large  # filter to specific configs
    --run-id my-custom-run           # override auto-generated ID
    --backends numpy,pytorch         # only profile specific backends
    --max-threads 4                  # cap CPU threads
    --timeout 90                     # minutes before aborting (default: 60)
    --verbose                        # pass --verbose to nestim profiler
    --dry-run                        # show plan without launching
```

### Run IDs

Auto-generated format: `YYYY-MM-DD-HHMMSS-<git-hash>[-dirty]`
Example: `2026-03-20-143000-bc385ad`

The git hash ties results to a specific code version.

## Collecting Results

```bash
python profiling/collect_results.py --run-id 2026-03-20-143000-bc385ad
```

Downloads individual JSONs from S3 and produces:

- `profiling/results/{run-id}/combined.json` — all configs in one file
- `profiling/results/{run-id}/summary.csv` — flattened for pandas/notebooks

### Options

```
python profiling/collect_results.py \
    --run-id <id>                    # required
    --output ./my-results            # override output directory
    --format json,csv                # or just json, or just csv
```

## Instance Matrix

Defined in `profiling/instance_matrix.py`. Default configs:

### Compute-Optimized (c-series)

| Name | vCPUs | Memory |
|------|-------|--------|
| compute-small | 1 | 2 GB |
| compute-medium | 2 | 4 GB |
| compute-large | 4 | 8 GB |
| compute-xlarge | 8 | 16 GB |
| compute-2xlarge | 16 | 32 GB |

### General-Purpose (m-series)

| Name | vCPUs | Memory |
|------|-------|--------|
| general-small | 1 | 4 GB |
| general-medium | 2 | 8 GB |
| general-large | 4 | 16 GB |
| general-xlarge | 8 | 32 GB |

To add a custom config, edit `INSTANCE_MATRIX` in `instance_matrix.py`.

## Troubleshooting

### "No default subnets found"

The orchestrator uses the default VPC. If your account doesn't have one:
```bash
aws ec2 create-default-vpc
```

### Tasks stuck in PENDING

Check Fargate vCPU quota:
```bash
aws service-quotas get-service-quota \
    --service-code fargate \
    --quota-code L-3032A538
```

### Task fails immediately

Check CloudWatch logs:
```bash
aws logs tail /ecs/nestim-profiling --since 1h
```

### S3 upload fails in container

Ensure the task role has `s3:PutObject` permission and the bucket name matches.

### Docker build fails on Cython

Ensure `_cython_kernels.pyx` is present in `src/network_estimation/`.

## Cost Estimates

Rough per-run costs for the default 9-config matrix with `exhaustive` preset:

- **Fargate compute:** ~$2-5 per full run (depends on task duration)
- **S3 storage:** negligible (~$0.001 per run)
- **ECR storage:** ~$0.10/month for the image
- **CloudWatch:** ~$0.50/GB ingested

Total: **~$3-6 per full matrix run**

Use `--preset super-quick --configs compute-small` for near-free debug runs.
