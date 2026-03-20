# CLI Reference

## When to use this page

Use this page for exact command syntax and key flags.

## Entry commands

Participant workflow commands:

- `nestim smoke-test`
- `nestim init`
- `nestim validate`
- `nestim run`
- `nestim create-dataset`
- `nestim package`
- `nestim visualizer`
- `nestim profile-simulation`

## `nestim smoke-test`

Run a built-in `CombinedEstimator` dashboard check and print next-step participant commands.

```bash
nestim smoke-test [--detail raw|full] [--profile] [--show-diagnostic-plots] [--max-threads N] [--debug]
```

## `nestim init`

Create starter files in a target directory.

```bash
nestim init [path] [--json] [--debug]
```

## `nestim validate`

Validate estimator loading and output contract.

```bash
nestim validate --estimator <path> [--class <name>] [--json] [--debug]
```

## `nestim run`

Run local scoring with participant estimator.

```bash
nestim run --estimator <path> [options]
```

Default behavior: `nestim run --estimator <path>` is equivalent to `--runner subprocess`.

Key options:

- `--class <name>`
- `--runner inprocess|subprocess`
- `--n-mlps <int>`
- `--detail raw|full`
- `--profile`
- `--show-diagnostic-plots`
- `--json`
- `--dataset <path>` — use pre-created dataset `.npz` file
- `--max-threads N` — limit all backends to at most N CPU threads (see [Thread limiting](#thread-limiting))
- `--debug`

Recommended debug sequence:

```bash
nestim run --estimator ./path/to/estimator.py
nestim run --estimator ./path/to/estimator.py --debug
nestim run --estimator ./path/to/estimator.py --runner inprocess --debug
```

Runner mode tradeoff:

- `subprocess` (default): realistic isolation and safer runtime boundary.
- `inprocess`: clearer estimator-level tracebacks for local debugging.

## `nestim create-dataset`

Pre-create an evaluation dataset for reuse across runs.

```bash
nestim create-dataset [options] -o <output-path>
```

Key options:

- `--n-mlps <int>` (default: 10)
- `--n-samples <int>` (default: 10000)
- `--seed <int>` (optional, auto-generated if omitted)
- `--width <int>`, `--depth <int>`, `--estimator-budget <int>`
- `-o, --output <path>` (default: `eval_dataset.npz`)
- `--max-threads N` — limit all backends to at most N CPU threads (see [Thread limiting](#thread-limiting))
- `--json`
- `--debug`

See [Use Evaluation Datasets](../how-to/use-evaluation-datasets.md) for usage patterns.

## `nestim package`

Build a submission artifact.

```bash
nestim package --estimator <path> [options]
```

Key options:

- `--class <name>`
- `--requirements <path>`
- `--submission-metadata <path>`
- `--approach <path>`
- `--output <path>`
- `--json`
- `--debug`

## `nestim visualizer`

Launch the interactive Network Explorer in a browser.

```bash
nestim visualizer [--host HOST] [--port PORT] [--no-open] [--debug]
```

Checks for Node.js (>= 18), installs dependencies if needed, starts the Vite dev server, and auto-opens the browser.

Key options:

- `--host <address>` (default: `localhost`) — bind address, use `0.0.0.0` for remote access
- `--port <number>` (default: `5173`) — port number
- `--no-open` — suppress auto-open browser
- `--debug` — show full npm/Vite output on errors

On SSH/headless environments, browser auto-open is skipped automatically.

## `nestim profile-simulation`

Benchmark simulation backends head-to-head. Runs a correctness check followed by a timing sweep across a grid of network sizes and sample counts, reporting speedup relative to the NumPy reference.

```bash
nestim profile-simulation [--preset super-quick|quick|standard|exhaustive]
                          [--backends <comma-separated>]
                          [--output <path>]
                          [--max-threads N]
                          [--verbose]
                          [--backends-help]
                          [--debug]
```

Key options:

- `--preset <name>` (default: `standard`) — parameter sweep size:
  - `super-quick` — 1 width (64), 1 depth (4), 1k samples. Sub-second, for testing the debug loop.
  - `quick` — 1 width, 2 depths, 2 sample counts. Finishes in seconds.
  - `standard` — 2 widths, 5 depths, 3 sample counts. A few minutes.
  - `exhaustive` — 3 widths, 5 depths, 5 sample counts (up to 16.7 M samples). Thorough but slow.
- `--backends <list>` — comma-separated list of backends to profile (default: all installed). Valid names: `numpy`, `pytorch`, `numba`, `jax`, `scipy`, `cython`.
- `--output <path>` — save a JSON report with hardware info, library versions, correctness results, and raw timing data.
- `--max-threads N` — limit all backends to at most N CPU threads (see [Thread limiting](#thread-limiting)).
- `--debug` — show full tracebacks on errors.
- `--verbose` — show full timing tables with all columns and raw data.
- `--backends-help` — print install instructions for all backends and exit.

The profiler automatically skips backends whose dependencies are not installed and prints `pip install` hints for them. Only backends that pass the pre-flight correctness check are included in the timing sweep.

Example workflows:

```bash
# Quick smoke test — just NumPy and PyTorch
nestim profile-simulation --preset quick --backends numpy,pytorch

# Full benchmark with JSON export
nestim profile-simulation --preset exhaustive --output profile_results.json

# Check which backends are available (quick correctness-only pass)
nestim profile-simulation --preset quick
```

## Thread limiting

Several commands accept `--max-threads N` to cap the number of CPU threads used by all numerical backends (BLAS, Numba, PyTorch, JAX/XLA). This is useful for:

- **Reproducible benchmarks** — pin thread count so results don't vary with machine load.
- **Shared machines** — avoid saturating all cores when others need them.
- **Single-threaded baselines** — use `--max-threads 1` to measure per-core performance.

### CLI flag

```bash
nestim run --estimator ./my-estimator/estimator.py --max-threads 2
nestim create-dataset --max-threads 2
nestim profile-simulation --max-threads 1 --preset quick
nestim smoke-test --max-threads 4
```

### Environment variable

Set `NESTIM_MAX_THREADS` before launching the process. The CLI flag takes precedence if both are set.

```bash
export NESTIM_MAX_THREADS=2
nestim profile-simulation
```

### What it controls

| Library / pool | Environment variable set |
|----------------|------------------------|
| OpenBLAS | `OPENBLAS_NUM_THREADS` |
| Intel MKL | `MKL_NUM_THREADS` |
| OpenMP | `OMP_NUM_THREADS` |
| Apple vecLib | `VECLIB_MAXIMUM_THREADS` |
| Numba | `NUMBA_NUM_THREADS` |
| NumExpr | `NUMEXPR_NUM_THREADS` |
| JAX / XLA | `XLA_FLAGS` (intra-op parallelism) |
| PyTorch | `torch.set_num_threads()` |

## ➡️ Next step

- [Score Report Fields](./score-report-fields.md)
- [Inspect and Traverse MLP Structure](../how-to/inspect-mlp-structure.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)
