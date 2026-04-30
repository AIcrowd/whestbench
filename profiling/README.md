# Cloud Profiling Infrastructure

Run `whest profile-simulation` benchmarks across a matrix of AWS Fargate
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
| S3 bucket | `whest-profiling-{account-id}` | Result storage |
| ECR repository | `whest-profiler` | Docker image registry |
| ECS cluster | `whest-profiling` | Fargate task execution |
| IAM execution role | `whest-profiler-execution` | ECR pull + CloudWatch |
| IAM task role | `whest-profiler-task` | S3 upload (scoped) |
| CloudWatch log group | `/ecs/whest-profiling` | Task logs |

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

Rebuild after any code changes to `src/` or backend dependencies.

The image includes all 6 CPU backends (NumPy, SciPy, PyTorch CPU, Numba,
JAX CPU, Cython). During the build, JIT-compiled functions (Numba `@njit`
and JAX `@jax.jit`) are invoked with representative tensor shapes so the
compiled code is cached in the image layer. This eliminates cold-start JIT
compilation overhead that can take 10-30 seconds per backend at runtime.

**When to rebuild:** Any change to files in `src/`, backend dependency
versions, or the Dockerfile itself requires a rebuild and push.

## Running Benchmarks

### Full run (default)

```bash
python profiling/run_benchmarks.py
```

Launches all 5 Fargate configs with `--preset exhaustive`. Shows a live status table.

### Debug run

```bash
python profiling/run_benchmarks.py --preset super-quick --configs compute-1vcpu
```

Fast iteration: single small config, minimal profiling.

### All options

```
python profiling/run_benchmarks.py \
    --preset exhaustive              # super-quick|quick|standard|exhaustive
    --configs compute-1vcpu,compute-4vcpu  # filter to specific configs
    --run-id my-custom-run           # override auto-generated ID
    --backends numpy,pytorch         # only profile specific backends
    --max-threads 4                  # cap CPU threads
    --timeout 90                     # minutes before aborting (default: 60)
    --verbose                        # pass --verbose to flopscope profiler
    --dry-run                        # show plan without launching
```

### Run IDs

Auto-generated format: `YYYY-MM-DD-HHMMSS-<git-hash>[-dirty]`
Example: `2026-03-20-143000-bc385ad`

The git hash ties results to a specific code version.

## Container Logging

Fargate containers run without a TTY, so the default Rich progress bar
(which uses in-place terminal rendering) produces no output in CloudWatch.

The container entrypoint automatically passes `--log-progress` to the
profiler, which prints one line per benchmark step to stdout:

```
[correctness] numpy ... PASS
[correctness] pytorch ... PASS
[timing] 1/270 numpy run_mlp w=64 d=4 n=1,000 (0s elapsed)
[timing] 2/270 numpy run_mlp_matmul_only w=64 d=4 n=1,000 (1s elapsed)
[warning] cython run_mlp w=256 d=128 n=16,700,000 skipped: MemoryError: ...
...
[done] Timing sweep complete. 268 results. (2 skipped due to errors)
```

If a backend hits an error (e.g., OOM) on a specific combination, it logs
a `[warning]` line and continues with the remaining combinations. Partial
results are preserved and uploaded to S3. The JSON output includes an
`error` field on skipped entries for diagnostics.

These lines appear in CloudWatch under the log group `/ecs/whest-profiling`
with stream prefix `{run-id}/profiler/{task-id}`.

To tail logs in real time:
```bash
aws logs tail /ecs/whest-profiling --follow --since 1h
```

For local testing, pass `--log-progress` directly:
```bash
whest profile-simulation --preset super-quick --log-progress
```

## Timeouts

Each Fargate task has two layers of timeout protection:

1. **Container-level timeout** (`TIMEOUT_MINUTES` env var, passed by
   orchestrator) — the entrypoint wraps the profiler command with `timeout`.
   If the profiler hangs, the process is killed and the task exits with
   code 124 (timeout) or 137 (OOM/SIGKILL).

2. **Orchestrator-level timeout** (`--timeout` flag, default: 60 min) —
   `run_benchmarks.py` monitors all tasks and calls `aws ecs stop-task` on
   any that exceed this limit.

The container timeout should always be shorter than the orchestrator timeout
to allow clean error reporting.

## Thread Pinning

In container environments, `nproc` often reports incorrect values (e.g.,
always 1 in Fargate regardless of allocated vCPUs). This causes backends
to run single-threaded even on multi-vCPU tasks.

The orchestrator solves this by deriving the thread count from Fargate CPU
units (`cpu / 1024`) and passing it as the `MAX_THREADS` environment
variable. The entrypoint then sets all threading environment variables
before the profiler starts:

- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`
- `NUMBA_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`
- `--max-threads` passed to the profiler for runtime enforcement

The entrypoint banner shows the effective thread count and its source:
```
Threads:     4 (source: env, nproc=1)
```

Override by setting `MAX_THREADS` in the task environment or using
`--max-threads` on the orchestrator.

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

Defined in `profiling/instance_matrix.py`. Each CPU tier gets the maximum
Fargate memory allocation so that memory is never the bottleneck — we're
measuring compute scaling, not memory pressure.

| Name | vCPUs | Memory |
|------|-------|--------|
| compute-1vcpu | 1 | 8 GB |
| compute-2vcpu | 2 | 16 GB |
| compute-4vcpu | 4 | 30 GB |
| compute-8vcpu | 8 | 60 GB |
| compute-16vcpu | 16 | 120 GB |

To add a custom config, edit `INSTANCE_MATRIX` in `instance_matrix.py`.

## Troubleshooting

### Tasks stuck with no log output

**Symptom:** Tasks show `RUNNING` for a long time but CloudWatch only has
the startup banner.

**Cause:** Before the `--log-progress` fix, the Rich progress bar produced
no output in non-TTY containers. Rebuild the Docker image with the latest
code:

```bash
bash profiling/build_and_push.sh
```

### Tasks timing out (exit code 124)

**Possible causes:**
- **Preset too heavy for config:** `exhaustive` on `compute-1vcpu` (1 vCPU)
  can take several hours. Use `--preset standard` or increase `--timeout`.
- **JIT compilation overhead:** If the Docker image wasn't rebuilt after code
  changes, JIT functions compile at runtime (adding 10-30s per backend).
  Rebuild the image.
- **Backend deadlock:** Rare, but possible with Numba's `parallel=True` if
  thread counts are misconfigured. The thread pinning in the entrypoint
  should prevent this.

### JIT compilation hangs in container

**Symptom:** Container hangs during the first few benchmark steps.

**Fix:** Rebuild the Docker image. The build now pre-compiles all Numba and
JAX JIT functions with representative tensor shapes. Old images that only
ran `NumbaBackend()` / `jax.numpy.ones(1)` did not actually trigger
compilation.

```bash
bash profiling/build_and_push.sh
```

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
aws logs tail /ecs/whest-profiling --since 1h
```

### S3 upload fails in container

Ensure the task role has `s3:PutObject` permission and the bucket name matches.

### Docker build fails on Cython

Ensure `_cython_kernels.pyx` is present in `src/whestbench/`.

## Cost Estimates

Rough per-run costs for the default 5-config matrix with `exhaustive` preset:

- **Fargate compute:** ~$5-15 per full run (max-memory configs are more
  expensive; the 16 vCPU / 120 GB task dominates cost)
- **S3 storage:** negligible (~$0.001 per run)
- **ECR storage:** ~$0.10/month for the image
- **CloudWatch:** ~$0.50/GB ingested

Total: **~$6-16 per full matrix run**

Use `--preset super-quick --configs compute-1vcpu` for near-free debug runs.
