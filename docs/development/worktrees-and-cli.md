# Worktrees and CLI

This note explains how `cestim` behaves when you use multiple git worktrees.

## Why This Matters

When installed with `uv tool install -e .`, `cestim` is a global shim in your `PATH`.
That shim points to one editable source directory at a time.
If you reinstall from another worktree, global `cestim` will follow that other path.

## Recommended Day-to-Day Usage

When actively switching worktrees, prefer commands that run from the current checkout:

```bash
uv run main.py
```

or:

```bash
uv run --with-editable . cestim
```

These avoid accidental execution from a stale global editable install.

## Repoint Global `cestim` To Current Checkout

From the checkout/worktree you want to use:

```bash
uv tool uninstall circuit-estimation
uv tool install --editable . --force
```

## Verify What Global `cestim` Uses

```bash
TOOL_PY="$(uv tool dir)/circuit-estimation/bin/python"
"$TOOL_PY" - <<'PY'
import circuit_estimation.cli as c
print(c.__file__)
PY
```

The printed path should be under the worktree you expect.
