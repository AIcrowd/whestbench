# CLI Reference

## When to use this page

Use this page for exact command syntax and key flags.

## Environment variables

- `WHEST_SKIP_HARDWARE_FALLBACK_PROBES=1` — skip OS-native fallback probes when collecting `run_meta.host` or dataset `metadata.hardware`. Cheap fields and `psutil`-backed fields are still collected; fallback-backed fields may remain `null`.
- `HF_TOKEN` — HuggingFace Hub authentication token. Used by `whest dataset upload`, `whest dataset download`, and `whest run --dataset hf://...` as a fallback when `--token` is not provided.

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
- `whest version`

All JSON outputs include a top-level `whestbench_version` string for traceability.

## `whest version`

Print installed whestbench version.

```bash
whest version [--format rich|plain|json] [--json]
```

JSON output is:

```json
{
  "ok": true,
  "command": "version",
  "name": "whestbench",
  "version": "0.2.0",
  "whestbench_version": "0.2.0"
}
```

Examples:

```bash
whest version
whest version --json
```

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
- `--flop-budget <int>` — cap on effective compute C_m per MLP. With the default `--lambda-flops-per-second 0` this is a pure FLOP cap (`C_m = F_m`). Default: `2_199_023_255_552` (`2**41`, the Phase 2 per-MLP budget; for reference, Phase 1 used `272_000_000_000` / 2.72e11 and the `v1-warmup` round `6.8e10`). Always honored; any `flop_budget` stored in `--dataset`'s metadata is ignored.
- `--wall-time-limit <seconds>` (default: `120.0`) — wall-clock limit per `predict()` call; forwarded to the estimator `BudgetContext`. Matches the grader's per-`predict()` wall cap; lower it to rehearse a tighter budget. Exceeding it zeroes that MLP's predictions. The primary compute constraint is `--flop-budget`.
- `--setup-timeout <seconds>` (default: `5.0`) — wall-clock limit for the one-time `setup()` call. Matches the grader's setup cap. Exceeding it fails the whole submission rather than a single MLP, so a submission that loads large weights in `setup()` is worth timing against this locally. Must be positive.
- `--residual-wall-time-limit <seconds>` (default: `0.4`) — limit on non-flopscope time per `predict()` call, enforced by WhestBench after timing is reported. Matches the graded cap. Residual time is plumbing (unpacking `mlp`, control flow around your `fnp` calls, assembling the result), not computation; crossing it zeroes that MLP's predictions and sets `residual_wall_time_exhausted`. Raise it to debug under a profiler.
- `--no-residual-wall-time-limit` — disable the residual gate entirely. Needed to re-score an earlier round that priced residual time through λ rather than gating it; overrides `--residual-wall-time-limit`.
- `--lambda-flops-per-second <rate>` (default: `0`) — price of one second of residual wall time, in FLOP-equivalents, for `C_m = F_m + λ·R_m`. `0` means residual time is **gated, not priced** (see [flopscope primer](./flopscope-primer.md#residual-wall-time-gated-not-priced)), so `C_m = F_m`. Pass `1e11` with `--no-residual-wall-time-limit` to reproduce the earlier priced regime. Must not be negative.
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
  - HF Hub with inline revision: `hf://owner/repo@v1` or `hf://aicrowd/arc-whestbench-public-2026@v1`
  - HF Hub with `--revision` flag: `aicrowd/arc-whestbench-public-2026 --revision v1`
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
whest run --estimator ./estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1

# HF Hub with separate --revision flag
whest run --estimator ./estimator.py \
    --dataset aicrowd/arc-whestbench-public-2026 \
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
    --n-mlps N --n-samples N --width W --depth D \
    [--split SPLIT] [--config CONFIG] \
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

- `--split <name>` — dataset split name. Default: `public`.
- `--config <name>` — HF dataset config name for this split. Default: `default`. Use this for authoring config-per-split datasets such as `default/mini + full/full` or `default/public + holdout/holdout`.
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
    --output ./my-eval

# Partial bake (slice 0 of 4)
whest dataset bake \
    --n-mlps 100 --n-samples 1_000_000_000 \
    --width 256 --depth 8 \
    --slice 0/4 \
    --output ./partial-0

# GPU bake
whest dataset bake \
    --n-mlps 100 --n-samples 1_000_000_000 \
    --width 256 --depth 8 \
    --torch --device auto \
    --output ./gpu-eval
```

### `whest dataset info`

Print metadata from a local directory or a HF Hub repo.

```bash
whest dataset info <DIR_OR_REPO_ID> [--revision REV]
```

Arguments:

- `DIR_OR_REPO_ID` — local dataset directory, or HF Hub repo id (e.g. `aicrowd/arc-whestbench-public-2026`).
- `--revision <tag>` — HF Hub git tag or commit SHA (for remote repos).

### Example

```bash
# Local
whest dataset info ./my-eval

# Remote
whest dataset info aicrowd/arc-whestbench-public-2026 --revision v1
```

Output prints key metadata fields: `schema_version`, `format`, `backend`, `split`, `config`, `n_mlps`, `n_samples`, `width`, `depth`, `created_at_utc`, and device provenance for torch bakes. Multi-split datasets print each split's `config` when present.

### `whest dataset upload`

Upload a baked dataset directory to HuggingFace Hub. Requires `HF_TOKEN` set in the environment or `--token`.

```bash
whest dataset upload <LOCAL_DIR> \
    --repo REPO_ID \
    [--tag TAG] \
    [--private] \
    [--token TOKEN] \
    [--message MSG]
```

Arguments:

- `LOCAL_DIR` — local directory produced by `whest dataset bake` or `whest dataset merge`.
- `--repo <repo_id>` — HF Hub repo id, e.g. `aicrowd/arc-whestbench-public-2026`.
- `--tag <tag>` — optional git tag to create on the uploaded commit (e.g. `v1`). Recommended for versioning.
- `--private` — create the repo as private if it doesn't exist yet.
- `--token <token>` — HF Hub write token. Falls back to `HF_TOKEN` env var, then the `huggingface-cli login` cache.
- `--message <msg>` — commit message for the HF Hub upload.

### Example

```bash
# Publish with a version tag
whest dataset upload ./my-eval \
    --repo aicrowd/arc-whestbench-public-2026 \
    --tag v1 \
    --message "Bake: 10 MLPs, seed=42"

# Private repo
whest dataset upload ./my-eval \
    --repo aicrowd/arc-whestbench-evals-2026 \
    --tag v1 \
    --private
```

### `whest dataset download`

Download a dataset from HuggingFace Hub. By default the files land in the HF
hub cache only (honouring `HF_HUB_CACHE` / `HF_HOME`), so a later
`whest run --dataset hf://…` is a cache hit. Pass `--output` to additionally
materialise a copy into a local directory.

```bash
whest dataset download <REPO_ID> \
    [--revision REV] \
    [--output DIR] \
    [--token TOKEN] \
    [--split SPLIT]
```

Arguments:

- `REPO_ID` — HF Hub repo id (e.g. `aicrowd/arc-whestbench-public-2026`).
- `--revision <tag>` — HF Hub git tag or commit SHA. Default: `main`.
- `--output <dir>` — optional: also materialise the files into this directory.
  Without it, the dataset is fetched into the HF hub cache only.
- `--token <token>` — HF Hub token for private repos. Falls back to `HF_TOKEN` env var.
- `--split <name>` — optional: download only the specified split's parquet
  (plus `metadata.json` and `README.md`). Errors if the split matches no
  parquet files in the repo.

### Example

```bash
# Prefetch into the HF cache (no local copy)
whest dataset download aicrowd/arc-whestbench-public-2026 \
    --revision v1

# Materialise an on-disk copy as well
whest dataset download aicrowd/arc-whestbench-public-2026 \
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
    --width 256 --depth 8 \
    --output ./my-eval

# 2. Inspect locally
whest dataset info ./my-eval

# 3. Publish
export HF_TOKEN=hf_...
whest dataset upload ./my-eval \
    --repo aicrowd/arc-whestbench-public-2026 \
    --tag v1

# 4. Prefetch on another machine (lands in the HF cache)
whest dataset download aicrowd/arc-whestbench-public-2026 \
    --revision v1

# 5. Run evaluation (cache hit — no re-download)
whest run --estimator ./estimator.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v1
```

## `whest package`

Build a submission artifact from your estimator.

```bash
whest package --estimator <path> [--output <path>] [--yes|-y] [options]
```

`--estimator` accepts **either a directory or a file**, and that choice decides what ships:

- **A directory** (e.g. `--estimator .` or `--estimator ./my-submission`) packages the **whole folder**: every file in it, minus the built-in ignore set, credential files (see below), and any `.gitignore` / `.whestignore` patterns. Use this when your estimator imports sibling modules or loads weights/data files.
- **A file** (e.g. `--estimator ./estimator.py`) packages **only that one file**. Use this for a self-contained single-file estimator. Sibling modules and data files are **not** included; `whest package` names exactly what it is leaving behind and points you at the folder form. If `estimator.py` **imports** one of those siblings, packaging **fails** rather than writing an archive that cannot import at grade time.

> **Credential files are never bundled, in either mode.** Patterns like `.env`, `.env.*`, `*.pem`, `*.key`, `id_rsa`, `.netrc`, `.aws/`, `.ssh/` are always excluded for security — a submission ships to a public leaderboard, so a leaked secret would be exposed. Excluded credential files are listed in the preview so you can see exactly what was dropped. Only files that were submission candidates are listed — a match inside a directory the ignore set already drops (a `*.pem` CA bundle vendored into `.venv/`, say) never shipped in the first place and is not reported.

Before writing the archive, `whest package` prints a **preview**: the submission mode, every file that will ship and its size, the total size and count, any credential files excluded for security, and any `.py` files not reachable by import from `estimator.py` (likely dead code; add to `.whestignore` if they shouldn't ship). In **folder mode** it then asks for confirmation unless `--yes` / `-y` is passed or stdin is non-interactive. **File mode** does not prompt — it ships exactly the one file you named, after warning you that's all it ships and naming the siblings it drops.

Submissions are capped at **50 MB** total and **50 files**. Exceeding either cap aborts with an actionable error that names the largest files and suggests using `.whestignore`.

Key options:

- `--class <name>` — estimator class name (auto-detected if omitted)
- `--output <path>` — output path for the `.tar.gz` archive (default: `submission-<timestamp>.tar.gz` in the current directory)
- `--yes` / `-y` — skip the folder-submission confirmation prompt (for CI)
- `--format rich|plain|json`
- `--json` — alias for `--format json`
- `--debug`

> **Deprecated:** `--requirements`, `--submission-metadata`, and `--approach` no longer do anything — files are bundled by being present in the submission folder, not by being named on the command line. Passing them prints a warning; they will be removed in a future release. Note that the grader installs **no** third-party packages (the sandbox provides only `flopscope`, the `whestbench` API, and the Python stdlib), so a shipped `requirements.txt` has no effect either — do dependency-heavy work offline and ship the result as data.

### Built-in ignore set

The following are always excluded, regardless of `.gitignore` / `.whestignore`:

`.git`, `.hg`, `.svn`, `.venv`, `venv`, `env`, `__pycache__`, `*.pyc`, `*.pyo`,
`.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `.ipynb_checkpoints`, `.DS_Store`,
`*.tar.gz`, `*.tgz`, `*.zip`, `.whestignore`, `.gitignore`, `manifest.json`

Credential files are **always** excluded too, and (unlike the patterns above) this cannot be overridden — rename or remove the file if a match is a false positive:

`.env`, `.env.*`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `*.keystore`, `*.jks`,
`id_rsa`, `id_dsa`, `id_ecdsa`, `id_ed25519`, `.netrc`, `.pypirc`, `.npmrc`,
`.aws/`, `.ssh/`, `.gnupg/`, `credentials.json`, `credentials.yaml`, `credentials.yml`

To exclude additional files (scratch data, notebooks, large test fixtures), add patterns to `.whestignore` using the same glob syntax as `.gitignore`. `whest init` creates a starter `.whestignore` in new project directories.

### Examples

```bash
# Package your whole project folder (recommended for multi-file submissions)
whest package --estimator .

# Package a single self-contained file (ships only this file)
whest package --estimator ./estimator.py

# Skip the folder confirmation (CI)
whest package --estimator . --yes

# Custom output path
whest package --estimator . --output ./my-submission.tar.gz --yes
```

## `whest validate-package`

Verify a packaged submission archive against its bundled `manifest.json` — the same
integrity check the grader runs: the entrypoint module is present, and every
`files[]` entry is a regular file whose SHA-256 matches the archive bytes (no
directory entries). Exits non-zero and lists each problem if the archive is invalid.
`whest submit` runs this automatically before uploading.

```bash
whest validate-package <submission.tar.gz> [--format rich|plain|json] [--json]
```

## `whest submit`

Submit a packaged artifact (or an estimator folder) to the AIcrowd leaderboard.

```bash
whest submit <artifact.tar.gz> [options]
whest submit --estimator <path> [options]
whest submit --estimator <path> --dry-run [options]
```

Key options:

- `--estimator <path>` — package on-the-fly before submitting (equivalent to running `whest package` then `whest submit`). Accepts a **directory** (packages the whole folder) or a **file** (packages only that file), exactly like `whest package`. The same preview is shown before upload — in folder mode it asks for confirmation unless `--yes` is passed.
- `--yes` / `-y` — skip the folder-submission confirmation prompt (for CI).
- `--dry-run` — preview what would be uploaded (shows the mode, file list, sizes, and — in folder mode — any credential files excluded for security), then stop without submitting. Useful for inspecting a submission before it goes to the leaderboard.
- `--class <name>` — estimator class name (for `--estimator` packaging).
- `--description <text>` — label attached to the submission on AIcrowd (default: `"Submitted via whest submit"`).
- `--format rich|plain|json`
- `--json` — alias for `--format json`
- `--debug`

### Examples

```bash
# Submit a pre-packaged artifact
whest submit ./my-submission.tar.gz

# Package the whole folder and submit in one step (shows a preview + confirm)
whest submit --estimator .

# Package a single self-contained file and submit
whest submit --estimator ./estimator.py

# Preview (dry run) without submitting
whest submit --estimator . --dry-run
```

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
