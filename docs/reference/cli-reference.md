# CLI Reference

## 📌 When to use this page

Use this page for exact command syntax and key flags.

## Entry commands

Participant workflow commands:

- `cestim smoke-test`
- `cestim init`
- `cestim validate`
- `cestim run`
- `cestim package`

## `cestim smoke-test`

Run a built-in `CombinedEstimator` dashboard check and print next-step participant commands.

```bash
cestim smoke-test [--detail raw|full] [--profile] [--show-diagnostic-plots] [--debug]
```

## `cestim init`

Create starter files in a target directory.

```bash
cestim init [path] [--json] [--debug]
```

## `cestim validate`

Validate estimator loading and stream contract.

```bash
cestim validate --estimator <path> [--class <name>] [--json] [--debug]
```

## `cestim run`

Run local scoring with participant estimator.

```bash
cestim run --estimator <path> [options]
```

Default behavior: `cestim run --estimator <path>` is equivalent to `--runner subprocess`.

Key options:

- `--class <name>`
- `--runner inprocess|subprocess`
- `--n-circuits <int>`
- `--n-samples <int>`
- `--detail raw|full`
- `--profile`
- `--show-diagnostic-plots`
- `--json`
- `--debug`

Recommended debug sequence:

```bash
cestim run --estimator ./path/to/estimator.py
cestim run --estimator ./path/to/estimator.py --debug
cestim run --estimator ./path/to/estimator.py --runner inprocess --debug
```

Runner mode tradeoff:

- `subprocess` (default): realistic isolation and safer runtime boundary.
- `inprocess`: clearer estimator-level tracebacks for local debugging.

## `cestim package`

Build a submission artifact.

```bash
cestim package --estimator <path> [options]
```

Key options:

- `--class <name>`
- `--requirements <path>`
- `--submission-metadata <path>`
- `--approach <path>`
- `--output <path>`
- `--json`
- `--debug`

## ➡️ Next step

- [Score Report Fields](./score-report-fields.md)
- [Inspect and Traverse Circuit Structure](../how-to/inspect-circuit-structure.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)
