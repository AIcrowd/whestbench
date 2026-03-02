# CLI Reference

## Entry Commands

Participant workflow commands:

- `cestim init`
- `cestim validate`
- `cestim run`
- `cestim package`

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

## Legacy Mode

When called without participant subcommands, `cestim` falls back to a legacy local scoring dashboard mode.

## Next

- [Score Report Fields](./score-report-fields.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)
