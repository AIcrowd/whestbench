# CLI Reference

## When to use this page

Use this page for exact command syntax and key flags.

## Entry commands

Participant workflow commands:

- `whest smoke-test`
- `whest init`
- `whest validate`
- `whest run`
- `whest create-dataset`
- `whest package`
- `whest visualizer`
- `whest profile-simulation`

## `whest smoke-test`

Run a built-in `CombinedEstimator` dashboard check and print next-step participant commands.

```bash
whest smoke-test [--detail raw|full] [--profile] [--show-diagnostic-plots] [--no-rich] [--debug]
```

- `--no-rich` — disable Rich live display and progress bars; print a plain-text report. Useful for non-TTY logs and for debugger sessions. Auto-enabled when `sys.gettrace()` is non-None, `PYTHONBREAKPOINT` is set, or `WHESTBENCH_NO_RICH=1`.

## `whest init`

Create starter files in a target directory.

```bash
whest init [path] [--json] [--debug]
```

## `whest validate`

Validate estimator loading and output contract.

```bash
whest validate --estimator <path> [--class <name>] [--json] [--debug]
```

## `whest run`

Run local scoring with participant estimator.

```bash
whest run --estimator <path> [options]
```

Default behavior: `whest run --estimator <path>` is equivalent to `--runner server`.

Key options:

- `--class <name>`
- `--runner local|server`
- `--n-mlps <int>`
- `--wall-time-limit <seconds>` — wall-clock limit per `predict()` call; forwarded to the estimator `BudgetContext`
- `--untracked-time-limit <seconds>` — limit for non-whest time per `predict()` call, enforced by WhestBench after timing is reported
- `--detail raw|full`
- `--profile`
- `--show-diagnostic-plots`
- `--json`
- `--dataset <path>` — use pre-created dataset `.npz` file
- `--no-rich` — disable Rich live display and progress bars; print a plain-text report. Required when attaching `pdb.set_trace()` (the live display would otherwise mask the prompt). Auto-enabled when `sys.gettrace()` is non-None, `PYTHONBREAKPOINT` is set, or `WHESTBENCH_NO_RICH=1`. See [Debugging Checklist](../how-to/debugging-checklist.md#using-pdb--breakpoint-inside-your-estimator).
- `--debug` — include estimator tracebacks in the report's "Estimator Errors" panel (works with any runner).
- `--fail-fast` — stop on the first estimator error and let the raw Python traceback propagate (combine with `--debug` to show it).

Recommended debug sequence:

```bash
whest run --estimator ./path/to/estimator.py
whest run --estimator ./path/to/estimator.py --debug
whest run --estimator ./path/to/estimator.py --debug --fail-fast
whest run --estimator ./path/to/estimator.py --runner local --no-rich   # for pdb.set_trace() / breakpoint()
```

### Exit codes

- `0` — scoring completed; no estimator errors (budget or time exhaustion still exits `0`).
- `1` — at least one MLP raised during `predict`, or setup/runtime failure.

Runner mode tradeoff:

- `server` (default): realistic isolation -- your estimator runs against the whest server.
- `local`: in-process execution with better traceback fidelity while debugging. Required for interactive debuggers (`pdb`, `breakpoint()`).

## `whest create-dataset`

Pre-create an evaluation dataset for reuse across runs.

```bash
whest create-dataset [options] -o <output-path>
```

Key options:

- `--n-mlps <int>` (default: 10)
- `--ground-truth-samples <int>` (default: 10000)
- `--seed <int>` (optional, auto-generated if omitted)
- `--width <int>`, `--depth <int>`, `--flop-budget <int>`
- `-o, --output <path>` (default: `eval_dataset.npz`)
- `--json`
- `--debug`

See [Use Evaluation Datasets](../how-to/use-evaluation-datasets.md) for usage patterns.

## `whest package`

Build a submission artifact.

```bash
whest package --estimator <path> [options]
```

Key options:

- `--class <name>`
- `--requirements <path>`
- `--submission-metadata <path>`
- `--approach <path>`
- `--output <path>`
- `--json`
- `--debug`

## `whest visualizer`

Launch the interactive WhestBench Explorer in a browser.

```bash
whest visualizer [--host HOST] [--port PORT] [--no-open] [--debug]
```

Checks for Node.js (>= 18), installs dependencies if needed, starts the Vite dev server, and auto-opens the browser.

Key options:

- `--host <address>` (default: `localhost`) — bind address, use `0.0.0.0` for remote access
- `--port <number>` (default: `5173`) — port number
- `--no-open` — suppress auto-open browser
- `--debug` — show full npm/Vite output on errors

On SSH/headless environments, browser auto-open is skipped automatically.

## `whest profile-simulation`

Profile whest FLOP accounting and analytical correctness across a grid of network sizes and FLOP budgets.

```bash
whest profile-simulation [--preset super-quick|quick|standard|exhaustive]
                          [--output <path>]
                          [--verbose]
                          [--debug]
```

Key options:

- `--preset <name>` (default: `standard`) — parameter sweep size:
  - `super-quick` — 1 width (256), 1 depth (4), 10 000 samples. Sub-second, for testing the debug loop.
  - `quick` — 1 width (256), 2 depths (4, 128), 2 sample counts (10 000, 100 000). Finishes in seconds.
  - `standard` — 2 widths (64, 256), 3 depths (4, 32, 128), 2 sample counts (10 000, 100 000). Under a minute.
  - `exhaustive` — 2 widths (64, 256), 3 depths (4, 32, 128), 3 sample counts (10 000, 100 000, 1 000 000). Thorough but slow.
- `--output <path>` — save a JSON report with correctness results and FLOP accounting data.
- `--debug` — show full tracebacks on errors.
- `--verbose` — show full tables with all columns and raw data.

Example workflows:

```bash
# Quick correctness check
whest profile-simulation --preset quick

# Full profile with JSON export
whest profile-simulation --preset exhaustive --output profile_results.json
```

## ➡️ Next step

- [Score Report Fields](./score-report-fields.md)
- [Inspect and Traverse MLP Structure](../how-to/inspect-mlp-structure.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)
