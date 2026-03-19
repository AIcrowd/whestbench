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

## ➡️ Next step

- [Score Report Fields](./score-report-fields.md)
- [Inspect and Traverse MLP Structure](../how-to/inspect-circuit-structure.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)
