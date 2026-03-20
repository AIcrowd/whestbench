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
nestim smoke-test [--detail raw|full] [--profile] [--show-diagnostic-plots] [--debug]
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
- `--n-samples <int>`
- `--detail raw|full`
- `--profile`
- `--show-diagnostic-plots`
- `--json`
- `--dataset <path>` — use pre-created dataset `.npz` file
- `--strict-baselines` — refuse to run if dataset creation hardware differs from execution runtime hardware
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
- `--width <int>`, `--max-depth <int>`, `--budgets <csv>`
- `-o, --output <path>` (default: `eval_dataset.npz`)
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
nestim profile-simulation [--preset quick|standard|exhaustive]
                          [--backends <comma-separated>]
                          [--output <path>]
                          [--debug]
```

Key options:

- `--preset <name>` (default: `standard`) — parameter sweep size:
  - `quick` — 1 width, 2 depths, 2 sample counts. Finishes in seconds.
  - `standard` — 2 widths, 5 depths, 3 sample counts. A few minutes.
  - `exhaustive` — 3 widths, 5 depths, 5 sample counts (up to 16.7 M samples). Thorough but slow.
- `--backends <list>` — comma-separated list of backends to profile (default: all installed). Valid names: `numpy`, `pytorch`, `numba`, `jax`, `scipy`, `cython`.
- `--output <path>` — save a JSON report with hardware info, library versions, correctness results, and raw timing data.
- `--debug` — show full tracebacks on errors.

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

## ➡️ Next step

- [Score Report Fields](./score-report-fields.md)
- [Inspect and Traverse MLP Structure](../how-to/inspect-mlp-structure.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)
