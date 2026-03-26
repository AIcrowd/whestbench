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
    --verbose                        # pass --verbose to nestim profiler
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

These lines appear in CloudWatch under the log group `/ecs/nestim-profiling`
with stream prefix `{run-id}/profiler/{task-id}`.

To tail logs in real time:
```bash
aws logs tail /ecs/nestim-profiling --follow --since 1h
```

For local testing, pass `--log-progress` directly:
```bash
nestim profile-simulation --preset super-quick --log-progress
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
aws logs tail /ecs/nestim-profiling --since 1h
```

### S3 upload fails in container

Ensure the task role has `s3:PutObject` permission and the bucket name matches.

### Docker build fails on Cython

Ensure `_cython_kernels.pyx` is present in `src/network_estimation/`.

## Cost Estimates

Rough per-run costs for the default 5-config matrix with `exhaustive` preset:

- **Fargate compute:** ~$5-15 per full run (max-memory configs are more
  expensive; the 16 vCPU / 120 GB task dominates cost)
- **S3 storage:** negligible (~$0.001 per run)
- **ECR storage:** ~$0.10/month for the image
- **CloudWatch:** ~$0.50/GB ingested

Total: **~$6-16 per full matrix run**

Use `--preset super-quick --configs compute-1vcpu` for near-free debug runs.

## BLAS Configuration

The Docker image uses **Intel MKL** as the BLAS backend for all 6 simulation
backends. This is an intentional choice for our target hardware.

### Target Hardware

AWS Fargate x86_64 runs exclusively on **Intel Xeon Platinum** processors
(Cascade Lake / Skylake) with full AVX-512 support. The specific CPUs observed
across runs:

- **Xeon Platinum 8259CL** @ 2.50 GHz (1/2/4/8 vCPU tiers)
- **Xeon Platinum 8175M** @ 2.50 GHz (16 vCPU tier)

### Why MKL over OpenBLAS

Isolated benchmarks on c5.4xlarge (Xeon Platinum 8124M @ 3.0 GHz) showed:

| Configuration | Time (4096x256 @ 256x256 sgemm) | GFLOPS | Relative |
|---------------|----------------------------------|--------|----------|
| NumPy + MKL | 482 us | 1104 | **baseline** |
| PyTorch + MKL | 467 us | ~1150 | 1.03x faster |
| NumPy + OpenBLAS | 607 us | 898 | **1.26x slower** |

MKL also scales better with threads: it saturates at 8 threads while
OpenBLAS regresses at 16 threads.

### How each backend gets MKL

| Backend | BLAS source | How |
|---------|-------------|-----|
| NumPy | MKL | conda `blas=*=mkl` |
| SciPy | MKL | conda (shares NumPy's BLAS) |
| Cython | MKL | calls `np.matmul` -> NumPy's MKL |
| Numba | MKL | `@njit` `@` dispatches to SciPy BLAS -> MKL |
| PyTorch | MKL | pip wheel statically links MKL into `libtorch_cpu.so` |
| JAX | MKL | conda-forge `jaxlib` links to conda's `libblas` -> MKL |

The Docker image uses Miniconda (not pip) as the base specifically because
pip's NumPy bundles OpenBLAS with no way to swap it. Conda's `blas` metapackage
allows selecting MKL at install time.

### If targeting AMD hardware

If Fargate ever moves to AMD EPYC processors, re-evaluate this choice.
OpenBLAS or BLIS may be faster on AMD. The Dockerfile header has notes on this.

## Profiling Findings

Results from exhaustive runs across 5 CPU tiers on AWS Fargate, March 2026.
All backends use MKL. Workload: 128-layer MLP with 256-wide hidden layers,
matmul-only (no activation functions).

### Backend Performance Rankings

Performance varies significantly by scale (batch size) and available CPU cores.

**Small batches (n=10,000) — dominated by framework overhead:**

| Rank | 2 vCPU | 4 vCPU | 16 vCPU |
|------|--------|--------|---------|
| 1st | PyTorch (667ms) | PyTorch (659ms) | PyTorch (174ms) |
| 2nd | Cython (766ms) | JAX (667ms) | Numba (197ms) |
| 3rd | NumPy (766ms) | Numba (711ms) | NumPy (199ms) |
| 4th | Numba (770ms) | NumPy (715ms) | JAX (223ms) |
| 5th | SciPy (974ms) | Cython (735ms) | SciPy (371ms) |
| 6th | JAX (1244ms) | SciPy (909ms) | |

At small n, PyTorch's optimized C++ dispatch loop beats everyone. JAX suffers
from XLA compilation/dispatch overhead that doesn't amortize.

**Large batches (n=1,000,000) — dominated by BLAS throughput:**

| Rank | 2 vCPU | 4 vCPU | 16 vCPU |
|------|--------|--------|---------|
| 1st | Cython (89.0s) | **JAX (63.4s)** | **JAX (13.7s)** |
| 2nd | NumPy (89.4s) | NumPy (85.9s) | Cython (18.7s) |
| 3rd | PyTorch (111.8s) | Cython (86.0s) | NumPy (22.6s) |
| 4th | Numba (117.3s) | PyTorch (97.8s) | PyTorch (30.2s) |
| 5th | JAX (122.9s) | Numba (109.6s) | Numba (31.6s) |
| 6th | SciPy (130.1s) | SciPy (124.7s) | SciPy (56.5s) |

At large n with 4+ vCPUs, JAX dominates — XLA's JIT fusion of chained matmuls
pays off massively (1.35-1.65x faster than NumPy). At 2 vCPU JAX can't
exploit this parallelism and falls behind.

### CPU Scaling Behavior

How backends scale from 2 to 16 vCPUs (matmul_only, w=256, d=128, n=1M):

| Backend | 2 vCPU | 4 vCPU | 16 vCPU | Scaling (2->16) |
|---------|--------|--------|---------|-----------------|
| JAX | 122.9s | 63.4s | 13.7s | **9.0x** |
| Cython | 89.0s | 86.0s | 18.7s | 4.8x |
| NumPy | 89.4s | 85.9s | 22.6s | 4.0x |
| PyTorch | 111.8s | 97.8s | 30.2s | 3.7x |
| Numba | 117.3s | 109.6s | 31.6s | 3.7x |
| SciPy | 130.1s | 124.7s | 56.5s | 2.3x |

JAX achieves near-linear scaling (9x on 8 physical cores / 16 logical). All
other backends plateau at 4-5x, limited by memory bandwidth on the chained
matmul workload. SciPy's raw `sgemm` approach scales worst.

### PyTorch vs NumPy: The Dispatch Overhead

PyTorch is 1.14-1.39x slower than NumPy on large matmul-only workloads despite
using the same BLAS (MKL). This is **not a BLAS issue** — it is per-layer
dispatch overhead from the `torch.from_numpy()` -> loop of `x = x @ w` ->
`.numpy()` pattern.

Evidence: the PyTorch/NumPy ratio is identical whether both use MKL or NumPy
uses OpenBLAS. On small batches, PyTorch is actually faster because its C++
loop overhead is fixed per-call and amortized.

We tested several approaches to reduce PyTorch's dispatch overhead
(w=256, d=128, n=10k, 4 threads):

| Method | Time | vs NumPy |
|--------|------|----------|
| NumPy loop (baseline) | 2,326ms | 1.00x |
| `torch.jit.trace` | 4,944ms | 2.13x |
| `nn.Sequential` | 6,207ms | 2.67x |
| Python torch loop (current) | 6,705ms | 2.88x |

- **`nn.Sequential`** is still a Python `for module in self` loop internally,
  so it adds `nn.Module` overhead without reducing dispatch calls.
- **`torch.jit.trace`** compiles the loop and shaves ~26% off, but is still
  2x slower than NumPy — each traced matmul still goes through PyTorch's
  tensor metadata checks, autograd bookkeeping, and BLAS dispatch.
- **`torch.compile`** with `reduce-overhead` mode could fuse operations, but
  the CPU backend provides minimal gains over `jit.trace`.

The dispatch overhead is structural to PyTorch's design on CPU. JAX solves
this by compiling the entire computation graph via XLA, which is why it is
the fastest backend at scale.

### Full MLP (with ReLU activations)

The `run_mlp` operation includes ReLU activations between layers. Rankings shift
because ReLU implementation quality matters:

At 16 vCPU, n=1M, w=256, d=128:

| Backend | run_mlp | matmul_only | ReLU overhead |
|---------|---------|-------------|---------------|
| JAX | 25.5s | 13.7s | 1.86x |
| Cython | 41.8s | 18.7s | 2.23x |
| PyTorch | 44.7s | 30.2s | 1.48x |
| Numba | 49.5s | 31.6s | 1.57x |
| NumPy | 69.3s | 22.6s | 3.07x |
| SciPy | 84.5s | 56.5s | 1.50x |

NumPy's ReLU is 3x slower relative to its matmul because `np.maximum` creates
temporaries. JAX and Cython fuse the ReLU with the matmul loop, keeping the
advantage.

### MKL vs OpenBLAS: Measured Impact on Fargate

We ran identical workloads with OpenBLAS (pip numpy) and MKL (conda numpy)
images across the same Fargate tiers. PyTorch was used as a control since it
ships its own static MKL regardless of the system BLAS.

**Key findings:**

1. **NumPy/SciPy/Cython/Numba: 1-5% faster with MKL** on heavy workloads.
   The improvement is real but modest — much smaller than the 26% seen in
   isolated single-sgemm benchmarks. Our workloads are memory-bound chained
   matmuls (256x256), not single large matrix operations, so BLAS throughput
   is diluted by Python overhead and data movement.

2. **PyTorch: unchanged** (as expected — was already using MKL).

3. **JAX: up to 19% faster** on the largest workloads (d=128, n=1M at 16
   vCPU). The conda-forge `jaxlib` links against MKL for its XLA BLAS calls,
   replacing the bundled Eigen kernels.

4. **The PyTorch-vs-NumPy gap did not close** (ratio stayed at 1.14-1.39x),
   confirming the gap is dispatch overhead, not BLAS-related.

### Run-to-Run Reproducibility

Fargate tasks land on different physical hosts each run. We measured variance
across 4 identical runs on the 16 vCPU tier over 3 days:

**Heavy workloads (w=256, d=128, n>=100k) are highly reproducible:**

| Workload | Run 1 | Run 2 | Run 3 | Run 4 | CV |
|----------|-------|-------|-------|-------|-----|
| PyTorch matmul d=128 n=1M | 29,615ms | 29,654ms | 29,835ms | 30,249ms | **0.8%** |
| NumPy matmul d=128 n=1M | 22,595ms | 22,485ms | 22,360ms | 22,598ms | **0.4%** |
| JAX matmul d=128 n=100k | 1,747ms | 1,740ms | 1,748ms | 1,752ms | **0.2%** |

**Overall variance across all workloads (PyTorch as control):**

| Metric | Value |
|--------|-------|
| Median CV | 6.1% |
| P10 CV | 2.1% |
| P90 CV | 10.5% |

Small workloads (tiny n, small matrices) are noisy (up to 46% CV) due to
scheduling jitter. Heavy workloads are stable to <2% across runs, making
them reliable for A/B comparisons. Any measured improvement below ~5% on
heavy workloads should be treated with caution; below ~10% on light
workloads is noise.

### Generating the Dashboard

```bash
# Collect results
python profiling/collect_results.py --run-id <run-id>

# Generate interactive HTML dashboard
python profiling/generate_dashboard.py \
    --input profiling/results/<run-id>/combined.json \
    --output dashboard.html
```

The dashboard includes backend comparison charts, CPU scaling plots,
heatmaps, and a filterable raw data table. It is a self-contained HTML
file (React + Recharts, no external dependencies).
