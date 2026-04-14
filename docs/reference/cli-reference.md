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
whest smoke-test [--detail raw|full] [--profile] [--show-diagnostic-plots] [--debug]
```

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
- `--detail raw|full`
- `--profile`
- `--show-diagnostic-plots`
- `--json`
- `--dataset <path>` — use pre-created dataset `.npz` file
- `--debug`

Recommended debug sequence:

```bash
whest run --estimator ./path/to/estimator.py
whest run --estimator ./path/to/estimator.py --debug
whest run --estimator ./path/to/estimator.py --runner local --debug
```

Runner mode tradeoff:

- `server` (default): realistic isolation -- your estimator runs against the whest server.
- `local`: in-process execution with better traceback fidelity while debugging.

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
  - `super-quick` — 1 width (64), 1 depth (4). Sub-second, for testing the debug loop.
  - `quick` — 1 width, 2 depths. Finishes in seconds.
  - `standard` — 2 widths, 5 depths. A few minutes.
  - `exhaustive` — 3 widths, 5 depths. Thorough but slow.
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
