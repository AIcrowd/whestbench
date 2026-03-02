# Install And CLI Quickstart

This is the shortest path from clone to first run.

## 1) Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2) Install the CLI from this repo

```bash
uv tool install -e .
```

This installs:

- `cestim`
- `circuit-estimation`

## 3) Verify CLI works

```bash
cestim --agent-mode
```

## 4) Participant workflow commands

```bash
# scaffold starter estimator files
cestim init ./my-estimator

# validate your estimator contract
cestim validate --estimator ./my-estimator/estimator.py

# run local scoring
cestim run --estimator ./my-estimator/estimator.py --runner subprocess

# package submission artifact
cestim package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz
```

## Worktrees Note

If you switch worktrees often, run from current checkout explicitly:

```bash
uv run --with-editable . cestim --agent-mode
```

See [Worktrees and CLI](../development/worktrees-and-cli.md) for details.
