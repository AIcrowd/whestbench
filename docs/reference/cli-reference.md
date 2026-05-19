# CLI Reference

## When to use this page

Use this page for exact command syntax and key flags.

## Environment toggles

- `WHEST_SKIP_HARDWARE_FALLBACK_PROBES=1` — skip OS-native fallback probes when collecting `run_meta.host` or dataset `metadata.hardware`. Cheap fields and `psutil`-backed fields are still collected; fallback-backed fields may remain `null`.

## Entry commands

Participant workflow commands:

- `whest smoke-test`
- `whest doctor`
- `whest init`
- `whest validate`
- `whest run`
- `whest create-dataset`
- `whest package`
- `whest profile-simulation`

## `whest smoke-test`

Run a built-in `CombinedEstimator` dashboard check and print next-step participant commands.

```bash
whest smoke-test [--detail raw|full] [--profile] [--show-diagnostic-plots] [--format rich|plain|json] [--debug]
```

- `--format rich|plain|json` — choose styled terminal output, plain log-friendly output, or JSON. Defaults to `rich` on TTYs and `plain` otherwise. Under a debugger, `smoke-test` automatically forces `plain` if `rich` was requested.

## `whest doctor`

Run install and environment health checks. Prints a pass/fail list for Python version, `uv`/Node.js availability, BLAS thread pool, disk space, and working-directory writability. Useful for first-hour setup troubleshooting and for CI gates.

```bash
whest doctor [--format rich|plain|json] [--json] [--strict] [--debug]
```

Key options:

- `--format rich|plain|json` — choose styled terminal output, plain log-friendly output (`[OK]`/`[WARN]`/`[FAIL]` tokens, no box-drawing), or JSON (`schema_version`, `checks`, `counts`, `overall`). Defaults to `rich` on TTYs and `plain` otherwise.
- `--json` — alias for `--format json`.
- `--strict` — treat warnings as failures for exit-code purposes. Rendering is unchanged.
- `--debug` — re-raise exceptions from crashing checks instead of capturing them as `fail`.

### Severity model

- `ok` — the check passed.
- `warn` — the check found something worth knowing but not blocking. Examples: `uv` missing (safe to ignore if you installed via pip), less than 1 GiB free disk in the current directory.
- `fail` — the check found a genuine blocker. Examples: Python version below `requires-python`, `threadpoolctl` failed to import, cannot write to the working directory.

### Exit codes

- Default: `0` if all checks are `ok` or `warn`; `1` if any `fail`.
- `--strict`: `0` only if all checks are `ok`; `1` otherwise.

### Example

```bash
# Interactive first-hour check
whest doctor

# CI pre-flight (treat anything that isn't OK as a failure)
whest doctor --strict --json
```

## `whest init`

Create starter files in a target directory.

```bash
whest init [path] [--format rich|plain|json] [--json] [--debug]
```

## `whest validate`

Validate estimator loading and output contract.

```bash
whest validate --estimator <path> [--class <name>] [--format rich|plain|json] [--json] [--debug]
```

## `whest run`

Run local scoring with participant estimator.

```bash
whest run --estimator <path> [options]
```

Default behavior: `whest run --estimator <path>` is equivalent to `--runner local`.

Key options:

- `--class <name>`
- `--runner local|subprocess|server|inprocess`
- `--n-mlps <int>`
- `--flop-budget <int>` — cap on effective compute C_m = F_m + λ·R_m per MLP. Default: `68_000_000_000` (6.8e10).
- `--wall-time-limit <seconds>` (default: `60.0`) — wall-clock limit per `predict()` call; forwarded to the estimator `BudgetContext`. Operational backstop matching the Phase 1 grader cap; the primary compute constraint is `--flop-budget`.
- `--residual-wall-time-limit <seconds>` — limit for non-flopscope time per `predict()` call, enforced by WhestBench after timing is reported
- `--detail raw|full`
- `--seed <int>` — random seed for the run.
  - Without `--dataset`: seeds both MLP generation and estimator setup (`ctx.seed`).
  - With `--dataset`: MLP seeds come from the dataset; this flag seeds estimator setup (`ctx.seed`) only.
  Default: omitted (`ctx.seed` defaults to 0; `run_config.seed` is `null` in the JSON output).
  See [estimator-contract.md](estimator-contract.md) for the `ctx.seed` reproducibility contract.
- `--profile`
- `--show-diagnostic-plots`
- `--format rich|plain|json` — choose styled terminal output, plain log-friendly output, or JSON. Defaults to `rich` on TTYs and `plain` otherwise.
- `--json` — alias for `--format json`
- `--dataset <path>` — use pre-created dataset `.npz` file
- `--debug` — include estimator tracebacks in the report's "Estimator Errors" panel (works with any runner).
- `--fail-fast` — stop on the first estimator error and let the raw Python traceback propagate (combine with `--debug` to show it).

Recommended debug sequence:

```bash
whest run --estimator ./path/to/estimator.py
whest run --estimator ./path/to/estimator.py --debug
whest run --estimator ./path/to/estimator.py --debug --fail-fast
whest run --estimator ./path/to/estimator.py --runner local --format plain   # for pdb.set_trace() / breakpoint()
```

### Exit codes

- `0` — scoring completed; no estimator errors (budget or time exhaustion still exits `0`).
- `1` — at least one MLP raised during `predict`, or setup/runtime failure.

Runner mode tradeoff:

- `local` (default): in-process execution with better traceback fidelity while debugging. Required for interactive debuggers (`pdb`, `breakpoint()`).
- `subprocess`: isolated execution in a separate process via the subprocess runner.
- `server`: legacy alias for `subprocess`.
- `inprocess`: alias for `local`.

## `whest create-dataset`

Pre-create an evaluation dataset for reuse across runs.

```bash
whest create-dataset [options] -o <output-path>
```

Key options:

- `--n-mlps <int>` (default: 10)
- `--n-samples <int>` (default: 10000)
- `--seed <int>` (optional, auto-generated if omitted)
- `--width <int>` (default: `256`) — neuron count per layer of the generated MLPs.
- `--depth <int>` (default: `8`) — number of weight matrices per MLP.
- `--flop-budget <int>` (default: `68_000_000_000`) — caps effective compute `C_m = F_m + λ·R_m` (not just analytical FLOPs). See [flopscope-primer.md](./flopscope-primer.md) for the formula.
- `-o, --output <path>` (default: `eval_dataset.npz`)
- `--format rich|plain|json`
- `--json` — alias for `--format json`
- `--debug`

See [Use Evaluation Datasets](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/how-to/use-evaluation-datasets.md) (in the starter kit) for usage patterns.

### GPU / torch backend (optional, for large datasets)

Pass `--device auto|cuda|mps|cpu` to use a torch-backed implementation. See
[GPU Dataset Generation](../how-to/gpu-dataset-generation.md) for the full
guide. Requires `pip install whestbench[gpu]`. With `--device`, `--max-threads`
is rejected — torch manages threading internally. Output schema is identical
to the default path.

- `--device <name>` — `auto` (resolves cuda > mps > cpu), `cuda`, `mps`, or `cpu`. Default omitted = flopscope path.

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
- `--format rich|plain|json`
- `--json` — alias for `--format json`
- `--debug`

## `whest profile-simulation`

Profile flopscope FLOP accounting and analytical correctness across a grid of network sizes and FLOP budgets.

```bash
whest profile-simulation [--preset super-quick|quick|standard|exhaustive]
                          [--output <path>]
                          [--format rich|plain|json]
                          [--json]
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
- `--format rich|plain|json` — choose styled terminal output, plain log-friendly output, or JSON. Defaults to `rich` on TTYs and `plain` otherwise.
- `--json` — alias for `--format json`
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
- [Inspect and Traverse MLP Structure](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/how-to/inspect-mlp-structure.md) (in the starter kit)
- [Validate, Run, and Package](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/how-to/validate-run-package.md) (in the starter kit)
