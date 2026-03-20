# Cloud Profiling Infrastructure Design

**Date:** 2026-03-20
**Status:** Approved
**Scope:** New `profiling/` directory for running benchmarks across AWS Fargate instances

## Problem

The existing `nestim profile-simulation` command produces detailed per-backend benchmarks
locally, but there is no way to systematically run these benchmarks across different
compute configurations and aggregate the results. This makes it hard to understand how
backend performance varies with CPU count and memory — critical for advising participants
on optimal estimator strategies.

## Goal

Run the existing profiler across a matrix of Fargate CPU/memory configurations in
parallel, collect results to S3, and aggregate them locally for analysis. Plumbing only —
visualization/dashboard is a follow-up.

## Non-Goals

- GPU instances (CPU-only focus)
- Dashboard or visualization (future work)
- CI/CD integration
- EC2 instances (Fargate only)

## Directory Structure

```
profiling/
├── Dockerfile                  # Single image with all 6 CPU backends
├── entrypoint.sh               # Container entrypoint: run profiler → upload to S3
├── setup_infra.sh              # Creates ECS cluster, S3 bucket, IAM roles, ECR repo
├── teardown_infra.sh           # Cleans up all AWS resources
├── build_and_push.sh           # Builds Docker image, pushes to ECR
├── run_benchmarks.py           # Orchestrator: launch, monitor, collect
├── collect_results.py          # Pull from S3, aggregate into combined report
├── instance_matrix.py          # Defines Fargate task configs (CPU/memory combos)
├── README.md                   # Comprehensive usage documentation
└── results/                    # Local results directory (gitignored)
```

## Dockerfile

Single optimized image targeting all 6 CPU backends:

- **Base:** `python:3.10-slim`
- **System deps:** `libopenblas-dev`, `gcc`, `g++`, `make` (for Cython and BLAS)
- **Python deps:** Project installed with `pip install -e .` followed by explicit backend
  installs (`pip install torch --index-url https://download.pytorch.org/whl/cpu`,
  `pip install numba "jax[cpu]"`) since optional backends are defined in
  `[dependency-groups]` not `[project.optional-dependencies]`
- **Cython:** Pre-compiled via `python setup_cython.py build_ext --inplace`
- **JIT pre-warm:** Numba and JAX caches warmed at build time to avoid cold-start overhead
  during profiling runs
- **Entrypoint:** `entrypoint.sh`

## Entrypoint (`entrypoint.sh`)

Receives configuration via environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `PRESET` | No | Profiler preset (default: `exhaustive`) |
| `RUN_ID` | Yes | Run identifier for S3 key prefix |
| `CONFIG_NAME` | Yes | Instance config name (e.g., `compute-large`) |
| `S3_BUCKET` | Yes | Target S3 bucket |
| `BACKENDS` | No | Comma-separated backend filter |
| `MAX_THREADS` | No | Thread cap passed to `--max-threads` |

The script uses `set -e` for exit-on-error so that failed profiler runs are correctly
reported as FAILED tasks in ECS (no S3 upload of partial/missing data).

Steps:
1. Run `nestim profile-simulation --preset $PRESET --output /tmp/results.json`
   (plus optional `--backends` and `--max-threads` if set)
2. Upload `/tmp/results.json` to `s3://$S3_BUCKET/$RUN_ID/$CONFIG_NAME.json` with
   retry (3 attempts with exponential backoff) to handle transient network failures

## Instance Matrix (`instance_matrix.py`)

Fargate doesn't map 1:1 to EC2 instance types. Instead, we define configs by varying
CPU/memory allocations to simulate compute-optimized (c-series) and general-purpose
(m-series) characteristics:

### Compute-Optimized (c-series equivalents)

| Name | vCPUs | Memory | Fargate CPU | Fargate Memory |
|------|-------|--------|-------------|----------------|
| `compute-small` | 1 | 2 GB | 1024 | 2048 |
| `compute-medium` | 2 | 4 GB | 2048 | 4096 |
| `compute-large` | 4 | 8 GB | 4096 | 8192 |
| `compute-xlarge` | 8 | 16 GB | 8192 | 16384 |
| `compute-2xlarge` | 16 | 32 GB | 16384 | 32768 |

> **Note:** The 8 and 16 vCPU configs require Fargate "large task" support (available
> since late 2023, limited regions). New AWS accounts have a default Fargate on-demand
> vCPU quota of 6 per region — request a quota increase before running the full matrix.
> The `compute-2xlarge` config is optional and can be removed from the matrix if quotas
> are not available.

### General-Purpose (m-series equivalents)

| Name | vCPUs | Memory | Fargate CPU | Fargate Memory |
|------|-------|--------|-------------|----------------|
| `general-small` | 1 | 4 GB | 1024 | 4096 |
| `general-medium` | 2 | 8 GB | 2048 | 8192 |
| `general-large` | 4 | 16 GB | 4096 | 16384 |
| `general-xlarge` | 8 | 32 GB | 8192 | 32768 |

Users add/remove entries in `instance_matrix.py` to customize the matrix.

## Infrastructure Setup (`setup_infra.sh`)

Creates the following AWS resources using AWS CLI. The script is **idempotent** — safe to
re-run without duplicating resources.

| Resource | Name/ID | Purpose |
|----------|---------|---------|
| S3 bucket | `nestim-profiling-{account-id}` | Stores result JSONs |
| ECR repository | `nestim-profiler` | Hosts Docker image |
| ECS cluster | `nestim-profiling` | Fargate-only cluster |
| IAM execution role | `nestim-profiler-execution` | Pull from ECR, write CloudWatch logs |
| IAM task role | `nestim-profiler-task` | S3 PutObject scoped to `arn:aws:s3:::nestim-profiling-{account-id}/*` |
| CloudWatch log group | `/ecs/nestim-profiling` | Task logs |

**S3 bucket hardening:**
- Public access block enabled (all four flags)
- Default SSE-S3 server-side encryption
- Optional lifecycle rule to auto-expire results after 90 days

**IAM least privilege:** The task role policy is scoped to `s3:PutObject` on the specific
bucket ARN only. The execution role uses the AWS-managed `AmazonECSTaskExecutionRolePolicy`.

Writes all created resource ARNs to `profiling/.infra-config.json` (gitignored, file
permissions 600) so other scripts can reference them without hardcoding.

**Expected environment:**
- AWS CLI configured with valid credentials
- `AWS_REGION` environment variable (defaults to `us-east-1`)

## Infrastructure Teardown (`teardown_infra.sh`)

Reads `.infra-config.json` and deletes all resources in reverse dependency order.
Prompts for confirmation before proceeding.

## Build & Push (`build_and_push.sh`)

1. Builds the Docker image using `docker build -f profiling/Dockerfile .` from the
   **repository root** (the Dockerfile needs access to `src/`, `pyproject.toml`,
   `setup_cython.py`)
2. Authenticates to ECR via `aws ecr get-login-password`
3. Tags and pushes the image
4. Updates `.infra-config.json` with the image URI

## Orchestrator (`run_benchmarks.py`)

### CLI Interface

```
python profiling/run_benchmarks.py \
    --preset exhaustive \                    # default: exhaustive
    --configs compute-small,compute-large \  # optional: filter configs
    --run-id 2026-03-20-v1 \                # optional: auto-generated if omitted
    --backends numpy,pytorch \               # optional: passed to nestim
    --max-threads 4 \                        # optional: passed to nestim
    --dry-run                                # show what would launch
```

### Run ID Generation

Auto-generated format: `YYYY-MM-DD-HHMMSS-{short-git-hash}`
Example: `2026-03-20-143000-bc385ad`

- Short git hash is first 7 chars of HEAD
- If working tree is dirty, appends `-dirty` (with a warning)
- Full commit hash stored in run metadata
- Can be overridden with `--run-id`

### Launch Phase

1. Read `.infra-config.json` for resource ARNs
2. Load instance matrix (all configs or filtered via `--configs`)
3. Register a Fargate task definition per config (CPU/memory vary; image and entrypoint
   are shared)
4. Run all tasks via `aws ecs run-task`, staggering launches with a 1-second delay
   between calls to avoid ECS `RunTask` API throttling (1 burst/sec default)
5. Each task receives `RUN_ID`, `CONFIG_NAME`, `S3_BUCKET`, `PRESET` (and optional
   `BACKENDS`, `MAX_THREADS`) as environment variable overrides

### Monitor Phase

- Poll `aws ecs describe-tasks` every 10 seconds
- Print a live status table to terminal:
  ```
  Config              Status    Duration
  compute-small       RUNNING   2m 34s
  compute-medium      RUNNING   2m 34s
  compute-large       PENDING   —
  general-small       STOPPED   4m 12s ✓
  ```
- Stream CloudWatch log tails for any task that fails
- Optional `--timeout` flag (default: 60 minutes) to abort and stop tasks if they exceed
  the time limit
- Exit when all tasks reach STOPPED state

## Result Collection (`collect_results.py`)

### CLI Interface

```
python profiling/collect_results.py \
    --run-id 2026-03-20-143000-bc385ad \  # required
    --output profiling/results/ \          # default
    --format json,csv                      # default: both
```

### Outputs

**Individual JSONs:** Downloaded to `profiling/results/{run-id}/{config-name}.json`

**`combined.json`** — structured aggregate:
```json
{
    "run_id": "2026-03-20-143000-bc385ad",
    "git_commit": "bc385ad1234567890abcdef",
    "git_dirty": false,
    "collected_at": "2026-03-20T14:35:00Z",
    "configs": {
        "compute-small": { "...full profiler JSON output..." : "..." },
        "compute-large": { "...full profiler JSON output..." : "..." }
    }
}
```

Each config's JSON already contains hardware fingerprint, backend timings, and primitive
breakdowns from the existing profiler output — no transformation needed.

**`summary.csv`** — flattened for pandas/notebook analysis:
```
config_name,cpu,memory,backend,width,depth,n_samples,run_mlp_time,output_stats_time,speedup_vs_numpy
```

Note: `speedup_vs_numpy` is relative to NumPy **on the same config**, not a global baseline.

**Partial run handling:** If some configs are missing from S3 (task failures), the collector
reports which are missing, produces partial `combined.json` and `summary.csv` with
available data, and exits with a warning (not an error).

### Terminal Output

Simple summary table: per-config backend count, fastest backend, total runtime. Just
enough to confirm the run succeeded.

## Documentation (`profiling/README.md`)

Must cover:

1. **Prerequisites** — AWS CLI, Docker, valid AWS credentials, required IAM permissions
2. **Quick Start** — end-to-end example from setup to results
3. **Infrastructure Setup/Teardown** — what gets created, how to clean up
4. **Building the Docker Image** — how to rebuild after code changes
5. **Running Benchmarks** — all CLI flags, examples for full runs vs. debug runs
6. **Collecting Results** — how to pull and aggregate
7. **Instance Matrix** — how to customize configs
8. **Troubleshooting** — common issues (permissions, Fargate limits, task failures)
9. **Cost Estimates** — rough per-run cost for the default matrix

## Testing Strategy

- **Local:** `--dry-run` flag on `run_benchmarks.py` to validate task definitions without
  launching
- **Smoke test:** Run with `--preset super-quick --configs compute-small` for fast
  validation of the full pipeline
- **Entrypoint:** Can be tested locally with `docker run` passing env vars pointing to a
  local/test S3 bucket
