# CLI Reference

## When to use this page

Use this page for exact command syntax and key flags.

## Environment variables

- `WHEST_SKIP_HARDWARE_FALLBACK_PROBES=1` — skip OS-native fallback probes when collecting `run_meta.host` or dataset `metadata.hardware`. Cheap fields and `psutil`-backed fields are still collected; fallback-backed fields may remain `null`.
- `HF_TOKEN` — HuggingFace Hub authentication token. Used by `whest dataset push`, `whest dataset pull`, and `whest run --dataset hf://...` as a fallback when `--token` is not provided.

## Commands

Participant workflow commands:

- `whest smoke-test`
- `whest doctor`
- `whest init`
- `whest validate`
- `whest run`
- `whest dataset` (bake / push / pull / merge / inspect)
- `whest package`
- `whest profile-simulation`

> **Migration note:** `whest create-dataset` is replaced by `whest dataset bake`. Running `whest create-dataset` prints a redirect and exits.

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

Run local scoring with a participant estimator.

```bash
whest run --estimator <path> [options]
```

Default behavior: `whest run --estimator <path>` is equivalent to `--runner local`.

Key options:

- `--class <name>` — estimator class name (if the module exports more than one).
- `--runner local|subprocess|server|inprocess`
- `--n-mlps <int>` — number of MLPs to evaluate. Default: 10 without `--dataset`; full dataset size with `--dataset`. Clamped to dataset size when `--dataset` is set.
- `--flop-budget <int>` — cap on effective compute C_m = F_m + λ·R_m per MLP. Default: `68_000_000_000` (6.8e10). Always honored; any `flop_budget` stored in `--dataset`'s metadata is ignored.
- `--wall-time-limit <seconds>` (default: `60.0`) — wall-clock limit per `predict()` call; forwarded to the estimator `BudgetContext`. Operational backstop matching the Phase 1 grader cap; the primary compute constraint is `--flop-budget`.
- `--residual-wall-time-limit <seconds>` — limit for non-flopscope time per `predict()` call, enforced by WhestBench after timing is reported.
- `--detail raw|full`
- `--seed <int>` — random seed for the run.
  - Without `--dataset`: seeds both MLP generation and estimator setup (`ctx.seed`).
  - With `--dataset`: MLP seeds come from the dataset; this flag seeds estimator setup (`ctx.seed`) only.
  Default: omitted (`ctx.seed` defaults to 0; `run_config.seed` is `null` in the JSON output).
  See [estimator-contract.md](estimator-contract.md) for the `ctx.seed` reproducibility contract.
- `--profile`
- `--show-diagnostic-plots`
- `--format rich|plain|json` — choose styled terminal output, plain log-friendly output, or JSON. Defaults to `rich` on TTYs and `plain` otherwise.
- `--json` — alias for `--format json`.
- `--dataset <path>` — dataset source. Accepts:
  - Local directory: `./my-eval` or `/abs/path/my-eval`
  - HF Hub with inline revision: `hf://owner/repo@v1` or `hf://aicrowd/arc-whestbench-2026-eval@v1`
  - HF Hub with `--revision` flag: `aicrowd/arc-whestbench-2026-eval --revision v1`
  Bare `owner/repo` without `--revision` is rejected (revision must be explicit).
- `--revision <tag>` — HF Hub git tag or commit SHA for `--dataset`. Ignored for local paths.
- `--n-samples <int>` — ground truth samples per MLP when generating on-the-fly (without `--dataset`). Default: `width*width*256`.
- `--debug` — include estimator tracebacks in the report's "Estimator Errors" panel.
- `--fail-fast` — stop on the first estimator error and let the raw Python traceback propagate. Combine with `--debug` to show it.
- `--max-threads <N>` — limit BLAS to at most N CPU threads.

Recommended debug sequence:

```bash
whest run --estimator ./path/to/estimator.py
whest run --estimator ./path/to/estimator.py --debug
whest run --estimator ./path/to/estimator.py --debug --fail-fast
whest run --estimator ./path/to/estimator.py --runner local --format plain   # for pdb.set_trace() / breakpoint()
```

### Using a pre-baked dataset

```bash
# Local directory (schema 3.0)
whest run --estimator ./estimator.py --dataset ./my-eval

# HF Hub with inline revision (preferred)
whest run --estimator ./estimator.py --dataset hf://aicrowd/arc-whestbench-2026-eval@v1

# HF Hub with separate --revision flag
whest run --estimator ./estimator.py \
    --dataset aicrowd/arc-whestbench-2026-eval \
    --revision v1
```

### Exit codes

- `0` — scoring completed; no estimator errors (budget or time exhaustion still exits `0`).
- `1` — at least one MLP raised during `predict`, or setup/runtime failure.

Runner mode tradeoff:

- `local` (default): in-process execution with better traceback fidelity while debugging. Required for interactive debuggers (`pdb`, `breakpoint()`).
- `subprocess`: isolated execution in a separate process via the subprocess runner.
- `server`: legacy alias for `subprocess`.
- `inprocess`: alias for `local`.

## `whest dataset`

Dataset management commands. All subcommands share the `whest dataset <sub>` prefix.

```bash
whest dataset {bake,push,pull,merge,inspect} ...
```

### `whest dataset bake`

Bake a new evaluation dataset to a local directory.

```bash
whest dataset bake \
    --n-mlps N --n-samples N --width W --depth D [--seed S] \
    [--split public|holdout] \
    --output DIR \
    [--torch] [--device auto|cuda|mps|cpu] \
    [--mlps-per-batch N] [--chunk-size N] \
    [--slice K/N | --mlp-range START-END]
```

Required options:

- `--n-mlps <int>` — total number of MLPs in the logical dataset.
- `--n-samples <int>` — ground-truth samples per MLP. Larger values give lower-noise ground truth. Default for on-the-fly runs is `width*width*256` (~16.7M for 256-wide).
- `--width <int>` — neuron count per layer.
- `--depth <int>` — number of weight matrices per MLP.
- `--output <dir>` — output directory (must not exist).

Key optional options:

- `--seed <int>` — reproducibility seed. Auto-generated if omitted; printed on completion.
- `--split public|holdout` — dataset split name. Default: `public`.
- `--torch` — use the GPU/torch backend (requires `pip install whestbench[gpu]`). See [GPU Dataset Generation](./gpu-dataset-generation.md).
- `--device auto|cuda|mps|cpu` — device when `--torch` is active. `auto` resolves `cuda > mps > cpu`.
- `--mlps-per-batch <int>` — torch backend: MLPs processed in parallel on device.
- `--chunk-size <int>` — torch backend: samples per chunk per step.
- `--slice K/N` — bake only the K-th slice of N total slices (0-indexed). Produces a partial dataset. Combine with `whest dataset merge` to assemble the full dataset. Example: `--slice 0/4` for the first of four workers.
- `--mlp-range START-END` — bake only MLP indices [START, END] inclusive (both ends). Alternative to `--slice` for irregular splits.

**Bit-equivalence guarantee:** a worker baking `--slice K/N` produces rows that are bitwise identical to the corresponding rows of a single-host bake with the same `--seed` and `--n-mlps`.

Output is a directory with:
```
<output>/
├── data/<split>-00000-of-00001.parquet
├── metadata.json
└── README.md
```

### Example

```bash
# Full bake (10 MLPs, 10M samples each)
whest dataset bake \
    --n-mlps 10 --n-samples 10_000_000 \
    --width 256 --depth 8 \
    --seed 42 \
    --output ./my-eval

# Partial bake (slice 0 of 4)
whest dataset bake \
    --n-mlps 100 --n-samples 1_000_000_000 \
    --width 256 --depth 8 \
    --seed 42 \
    --slice 0/4 \
    --output ./partial-0

# GPU bake
whest dataset bake \
    --n-mlps 100 --n-samples 1_000_000_000 \
    --width 256 --depth 8 \
    --seed 42 --torch --device auto \
    --output ./gpu-eval
```

### `whest dataset inspect`

Print metadata from a local directory or a HF Hub repo.

```bash
whest dataset inspect <DIR_OR_REPO_ID> [--revision REV]
```

Arguments:

- `DIR_OR_REPO_ID` — local dataset directory, or HF Hub repo id (e.g. `aicrowd/arc-whestbench-2026-eval`).
- `--revision <tag>` — HF Hub git tag or commit SHA (for remote repos).

### Example

```bash
# Local
whest dataset inspect ./my-eval

# Remote
whest dataset inspect aicrowd/arc-whestbench-2026-eval --revision v1
```

Output prints key metadata fields: `schema_version`, `format`, `backend`, `seed`, `n_mlps`, `n_samples`, `width`, `depth`, `created_at_utc`, and device provenance for torch bakes.

### `whest dataset push`

Upload a baked dataset directory to HuggingFace Hub. Requires `HF_TOKEN` set in the environment or `--token`.

```bash
whest dataset push <LOCAL_DIR> \
    --repo REPO_ID \
    [--tag TAG] \
    [--private] \
    [--token TOKEN] \
    [--message MSG]
```

Arguments:

- `LOCAL_DIR` — local directory produced by `whest dataset bake` or `whest dataset merge`.
- `--repo <repo_id>` — HF Hub repo id, e.g. `aicrowd/arc-whestbench-2026-eval`.
- `--tag <tag>` — optional git tag to create on the uploaded commit (e.g. `v1`). Recommended for versioning.
- `--private` — create the repo as private if it doesn't exist yet.
- `--token <token>` — HF Hub write token. Falls back to `HF_TOKEN` env var, then the `huggingface-cli login` cache.
- `--message <msg>` — commit message for the HF Hub upload.

### Example

```bash
# Publish with a version tag
whest dataset push ./my-eval \
    --repo aicrowd/arc-whestbench-2026-eval \
    --tag v1 \
    --message "Bake: 10 MLPs, seed=42"

# Private repo
whest dataset push ./my-eval \
    --repo aicrowd/arc-whestbench-2026-holdout \
    --tag v1 \
    --private
```

### `whest dataset pull`

Download a dataset from HuggingFace Hub to a local directory.

```bash
whest dataset pull <REPO_ID> \
    [--revision REV] \
    --output DIR \
    [--token TOKEN]
```

Arguments:

- `REPO_ID` — HF Hub repo id (e.g. `aicrowd/arc-whestbench-2026-eval`).
- `--revision <tag>` — HF Hub git tag or commit SHA. Default: `main`.
- `--output <dir>` — local destination directory.
- `--token <token>` — HF Hub token for private repos. Falls back to `HF_TOKEN` env var.

### Example

```bash
whest dataset pull aicrowd/arc-whestbench-2026-eval \
    --revision v1 \
    --output ./eval-v1
```

### `whest dataset merge`

Merge partial bakes (produced with `--slice` or `--mlp-range`) into a single canonical dataset.

```bash
whest dataset merge <DIR> [<DIR>...] --output <DIR>
```

Arguments:

- `<DIR>...` — two or more partial dataset directories.
- `--output <dir>` — destination for the merged dataset (must not exist).

All partial datasets must share the same `--seed`, `--n-mlps`, `--n-samples`, `--width`, `--depth`, and `--backend`. Their `mlp_range` values must together cover `[0, total_n_mlps)` exactly once (no gaps, no overlaps).

The merged result is bit-equivalent to a single-host bake with the same parameters.

### Example

```bash
# After baking 4 slices on separate workers:
whest dataset merge \
    ./partial-0 ./partial-1 ./partial-2 ./partial-3 \
    --output ./final-eval
```

## End-to-end example (bake → inspect → push → pull → run)

```bash
# 1. Bake
whest dataset bake \
    --n-mlps 10 --n-samples 10_000_000 \
    --width 256 --depth 8 --seed 42 \
    --output ./my-eval

# 2. Inspect locally
whest dataset inspect ./my-eval

# 3. Publish
export HF_TOKEN=hf_...
whest dataset push ./my-eval \
    --repo aicrowd/arc-whestbench-2026-eval \
    --tag v1

# 4. Pull on another machine
whest dataset pull aicrowd/arc-whestbench-2026-eval \
    --revision v1 --output ./local-copy

# 5. Run evaluation
whest run --estimator ./estimator.py \
    --dataset hf://aicrowd/arc-whestbench-2026-eval@v1
```

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
- `--json` — alias for `--format json`.
- `--debug` — show full tracebacks on errors.
- `--verbose` — show full tables with all columns and raw data.

Example workflows:

```bash
# Quick correctness check
whest profile-simulation --preset quick

# Full profile with JSON export
whest profile-simulation --preset exhaustive --output profile_results.json
```

## Next step

- [Dataset Format](./dataset-format.md) — schema 3.0 specification
- [Score Report Fields](./score-report-fields.md)
- [GPU Dataset Generation](./gpu-dataset-generation.md)
- [Inspect and Traverse MLP Structure](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/how-to/inspect-mlp-structure.md) (in the starter kit)
- [Validate, Run, and Package](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/how-to/validate-run-package.md) (in the starter kit)
