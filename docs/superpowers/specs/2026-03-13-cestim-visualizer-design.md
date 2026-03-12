# Design: `cestim visualizer` CLI Command

**Date:** 2026-03-13
**Status:** Draft

## Problem

The circuit explorer (interactive web visualizer in `tools/circuit-explorer/`) requires users to manually `cd` into the directory, run `npm install`, and `npm run dev`. This is confusing for challenge participants unfamiliar with the Node.js ecosystem.

## Solution

Add a `cestim visualizer` subcommand that wraps the entire circuit explorer lifecycle — prerequisite checks, dependency installation, dev server launch, and browser opening — into a single command.

## Command Interface

```
cestim visualizer [--host HOST] [--port PORT] [--no-open] [--debug]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `localhost` | Bind address. Server users set `0.0.0.0`. |
| `--port` | `5173` | Port number (Vite's default). |
| `--no-open` | `false` | Suppress auto-open even on desktop environments. |
| `--debug` | `false` | Show full npm/Vite stderr output (consistent with other subcommands). |

No estimator or dataset arguments — this is a standalone visualizer.

## Prerequisite Checks

On invocation, check for `node` and `npm` in PATH using `shutil.which()`.

If missing, display a Rich-formatted panel with install guidance:

- macOS: `brew install node`
- Ubuntu/Debian: `sudo apt install nodejs npm`
- General: link to https://nodejs.org/en/download

If `node` is found, check version (`node --version`) and require >= 18 (Vite 7.x requirement). If too old, display a Rich panel explaining the minimum version and how to upgrade.

Exit with code 1 after prerequisite failure messages.

## Explorer Path Resolution

This command only works when running from a source checkout (the standard workflow for challenge participants who clone the repo). The explorer directory is resolved by walking up from `__file__` looking for `pyproject.toml` to find the repo root, then checking for `tools/circuit-explorer/` relative to it.

If the explorer directory cannot be found (e.g., installed via pip without source), display a clear Rich error:

> "Circuit explorer not found. This command requires a source checkout of the repository. Clone the repo and install with `uv tool install -e .`"

Exit with code 1.

## npm Install & Dev Server Lifecycle

1. **Check `node_modules/`**: If `tools/circuit-explorer/node_modules/` exists, skip `npm ci`.
2. **If no `node_modules/`**: Run `npm ci` (deterministic install from `package-lock.json`) with a Rich spinner/status. On failure, print npm output with context and exit.
3. **Start dev server**: Run `npm run dev -- --host {host} --port {port}` via `subprocess.Popen`, streaming stdout/stderr to the terminal.
4. **Retry on failure**: If the dev server exits non-zero and `npm ci` was not run during this invocation, run `npm ci` (which cleans and reinstalls `node_modules/`) and retry `npm run dev` once. If it fails again, show the error with context and exit.
5. **Signal handling**: Use `process.terminate()` on the subprocess when SIGINT is received (Ctrl+C). This is cross-platform (works on both Unix and Windows).

### Exit Codes

| Scenario | Exit Code |
|----------|-----------|
| Missing Node/npm or version too old | 1 |
| Explorer directory not found | 1 |
| `npm ci` failure | 1 |
| Dev server failure (after retry) | 1 |
| Clean shutdown via Ctrl+C | 0 |

## Browser Auto-Open

After the dev server starts:

1. **Detect headless/SSH**: Check for `SSH_CLIENT`, `SSH_TTY`, or `SSH_CONNECTION` env vars. On Linux, also check if `DISPLAY` is unset. If headless detected, print URL only.
2. **Desktop**: Use `webbrowser.open(f"http://{host}:{port}")` to open the default browser. Always print the URL as well.
3. **`--no-open`**: Skip browser open, just print the URL.
4. **Timing**: Wait for Vite's stdout to emit a line containing `Local:` followed by a URL (Vite 7.x format: `  ➜  Local:   http://localhost:5173/`). Use line-by-line scan of subprocess output. If the pattern is not detected within 30 seconds, print the URL based on the configured host/port and continue without auto-open.

## File Organization

- **New module**: `src/circuit_estimation/visualizer.py` containing:
  - Prerequisite checks (node/npm detection and version check)
  - Explorer path resolution (repo root discovery)
  - npm ci logic
  - Dev server process management
  - Browser-open logic
  - Headless detection
- **CLI registration**: New `visualizer` subparser added in `cli.py` alongside existing commands, following the same argparse pattern. The `--debug` flag is included for consistency with other subcommands.
- **No changes** to the circuit-explorer JavaScript code.

## Documentation Updates

- **`docs/how-to/use-circuit-explorer.md`**: Update to reference `cestim visualizer` as the primary launch method, keeping manual npm instructions as a fallback.
- **`docs/reference/cli-reference.md`**: Add `cestim visualizer` command reference.
- **`README.md`**: Add a one-liner in the quick-start section showing `cestim visualizer`.

## Error Handling

All error output uses Rich formatting consistent with the rest of the CLI:

- **Missing Node/npm**: Rich panel with install instructions and platform-specific commands.
- **Node.js version too old**: Rich panel explaining minimum version (>= 18) with upgrade instructions.
- **Explorer not found**: Rich panel explaining this requires a source checkout.
- **npm ci failure**: Show which step failed ("Installing dependencies failed"), followed by the npm error output.
- **Dev server failure**: Show which step failed ("Dev server exited unexpectedly"), followed by the Vite error output. If retrying after npm ci, indicate that.
- **Port conflict**: Vite's own error message will surface; the context prefix ("Dev server exited unexpectedly") helps the user understand what happened.

## Testing

- **Unit tests** for pure functions: `_is_headless()`, `_find_explorer_dir()`, `_check_node_version()`.
- **Integration tests** with mocked `subprocess` for: npm ci invocation logic, retry-on-failure flow, signal forwarding.
- **Manual testing** checklist: run on macOS desktop, run over SSH, run with missing Node, run with stale node_modules.

## Out of Scope

- Auto-installing Node.js
- Windows support (challenge participants primarily use macOS/Linux; Ctrl+C handling uses `process.terminate()` which works cross-platform, but no Windows-specific testing)
- Passing estimator data to the visualizer
- Building/bundling the explorer for production deployment
- Any changes to the circuit-explorer JavaScript code
