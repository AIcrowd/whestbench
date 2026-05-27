# Changelog

## Unreleased

### Feat

- **metadata**: optional `default_split` field on multi-split datasets.
  When set, `whest run --dataset hf://…` without `--split` projects to the
  named split instead of erroring out. Set at bake time via the new
  `combine_split_datasets(default_split=…)` kwarg. `validate_metadata`
  enforces that the value is a string naming one of the dataset's splits.
  The dataset-card template surfaces it in the `whest run` quick-start
  block. Older whestbench versions ignore the field (forward-compatible).

## v0.6.0 (2026-05-27)

### Feat

- add whest version command and version metadata in JSON
- **cli**: validate/init/smoke-test/profile-simulation adopt unified copy
- **cli**: package gets a bytes progress bar
- **cli**: doctor wraps probes in a status spinner + bookends
- **cli**: merge gets spinner + before/after copy
- **cli**: download surfaces preflight summary + progress + completion
- **cli**: upload gets a real progress bar + before/after copy
- **cli**: bake gets phased progress bars + before/after copy
- **cli**: rename dataset push/pull/inspect to upload/download/info + deprecation
- **cli**: --streaming end-to-end with prominent cache-trade-off warning
- **cli**: add --streaming flag to whest run
- **cli**: use metadata-based n_mlps clamp when ds is streaming
- **scoring**: make_contest_from_dataset supports IterableDataset
- **cli**: wrap hf:// dataset load with hf_download progress UI
- **hf_progress**: add hf_upload context manager
- **hf_progress**: add hf_download context manager with three modes
- **hf_progress**: add RichHFTqdm that forwards into active Rich Progress
- **hf_progress**: add hf_preflight() with cache detection
- **hf_progress**: add HFPreflight dataclass
- **ui**: add status spinner context manager + finalize ui.py
- **ui**: add progress_count context manager
- **ui**: add progress_bytes context manager
- **ui**: add say.* message helpers (intent/step/ok/warn/hint)
- **ui**: add format_throughput helper
- **ui**: add format_duration helper
- **ui**: add format_bytes helper
- **template**: emit configs: block in YAML for explicit split ordering
- **package**: record tool and runtime versions in submission manifest

### Fix

- avoid duplicate JSON output in validate command
- keep final_layer_mse in narrow score subtitle
- guard profile-simulation JSON payload type for metadata wrapper
- **cli**: cache-hit download says "Loaded from cache" not "Downloaded"
- **cli**: drop stray comma in cache-miss download ok line
- **hf_progress**: bail preflight when revision cannot be resolved
- **hf_progress**: drop unused empty top-level upload task
- **hf_progress**: raise on nested hf_download/hf_upload
- **hf_progress**: subclass HF tqdm and guard disabled bars
- **ui**: match HF Hub env-var truthy semantics in _progress_disabled
- **ui**: roll over format_bytes at the next-unit boundary
- **dataset_io**: use attr-set for configs to satisfy Pyright

### Refactor

- **ui**: cache the default Console as a module-level singleton
- **ui**: inherit handles from ProgressHandle Protocol nominally

## v0.5.1 (2026-05-27)

### Feat

- **template**: mini+full quick-start snippet leads with split="mini"
- **template**: recognise mini+full split pair in dataset card

### Fix

- **template**: restore print(ds[0]['mlp_name']) smoke-test in generic quickstart fallback
- **template**: scope companion-disclaimer to public+holdout, fix whitespace + spelling
- **test**: import datasets.config submodule explicitly for pyright
- **dataset_io**: scope merge_datasets HF cache to tempdir by default

## v0.5.0 (2026-05-27)

### Feat

- **load_dataset**: add streaming=True support (closes #55)
- **readme**: per-split MLP counts + tighter Compute/Reproducibility wording
- **readme**: companion_repo template var + collapse hardware_fingerprints

### Fix

- **lint**: silence intentional type-violation in mlp_at streaming test
- **lint**: narrow load_dataset return type via Literal[streaming] overloads
- **lint**: narrow set element types before sort in fingerprint collapse

## v0.4.0 (2026-05-26)

### Added

- `seed_protocol 3.0` (`whestbench_explicit_per_mlp_seeds`): each MLP's seed is an independent input rather than a derivation from a single root. Each `mlp_seed` value in the parquet column is the canonical input seed. Within-MLP three-stream derivation (weight/sample/estimator) is preserved via `SeedSequence(mlp_seed).spawn(3)`.
- `whest dataset bake --mlp-seeds FILE` (JSON array of N ints) for explicit per-MLP seeds. Omitting both `--mlp-seeds` and `--seed` auto-generates via `secrets.randbits(63)`.
- `create_dataset(mlp_seeds=[...])` / `create_dataset_torch(mlp_seeds=[...])`.
- `MLP.from_row(row, *, seed_protocol_version=...)`: protocol-aware estimator-seed derivation.
- Frozen fixture `tests/fixtures/single_split_v3_protocol/` for schema-drift regression.
- Multi-split dataset support: dataset directories can now contain multiple Parquet files in `data/`, one per split, described by an optional `splits:` sub-dict in `metadata.json`. Backward-compatible — single-split datasets are unchanged.
- `whest dataset combine-splits INPUT_DIR... --output OUTPUT_DIR` CLI subcommand for assembling multi-split datasets from N complete single-split inputs.
- `whestbench.combine_split_datasets()` Python helper (re-exported from `whestbench`).
- `whest dataset bake --split <name>` now accepts arbitrary split names matching `[a-z][a-z0-9]*(-[a-z0-9]+)*` (previously restricted to `public` / `holdout`).
- `whest dataset pull --split <name>` and `whest run --dataset ... --split <name>` for selecting one split from multi-split datasets.

### Changed

- `create_dataset(seed=...)` / `create_dataset_torch(seed=...)` and `whest dataset bake --seed N` now reject with a migration hint pointing at `--mlp-seeds`.
- Parquet `mlp_seed` column semantics: under 3.0, the column stores the **input** seed (was: derived estimator seed under 2.0). `MLP.seed` (participant-facing) is unchanged across protocols — derived locally from the input under 3.0.
- `whest dataset inspect` now recognises multi-split datasets and prints a per-split summary, plus the `seed_protocol: <name> (version <version>)` line for all datasets.
- `whestbench.load_dataset()` returns `Dataset | DatasetDict` based on the dataset shape; explicit `split=` always returns `Dataset`.
- `whestbench.metadata()` accepts a `DatasetDict` and an optional `split=` filter that projects to single-split-shaped metadata.
- The dataset-card template gains a multi-split branch with leaderboard-specific wording when splits are `{public, holdout}`; the single-split `public` branch's wording is updated to point at the new evaluation repo.

### Compatibility

- `whestbench.load_dataset` reads both `seed_protocol 2.0` and `3.0` datasets indefinitely. Existing published datasets (e.g. `aicrowd/arc-whestbench-2026-smoke-test`) continue to work unchanged.
- New bakes only write 3.0.
- `schema_version` stays at `"3.0"`. The protocol discriminator is `seed_protocol.{name,version}`.
- The `splits:` field is purely additive.
- Old whestbench reading new multi-split datasets fails loudly with a missing-`n_mlps` error — upgrade whestbench to read multi-split.

---

## 0.3.0 — 2026-05-25

### BREAKING

- **Dataset format migrated from `.npz` to HF Parquet+sidecar (schema 2.4 → 3.0).**
  Datasets are now directories with `data/<split>-NNNNN.parquet`, `metadata.json`,
  and `README.md`. The `whest create-dataset` command is replaced by
  `whest dataset bake`. The `DatasetBundle` dataclass is removed; internal
  consumers operate on `datasets.Dataset` directly.
- **Public estimator interface unchanged.** Estimators still receive `MLP`
  instances via `predict(mlp: MLP)`.

### NEW

- `whestbench.load_dataset(path_or_repo, revision=..., split=..., token=...)` loads from local directories OR HF Hub.
- `whestbench.iter_mlps(ds)`, `whestbench.mlp_at(ds, i)`, `whestbench.metadata(ds)`.
- `whestbench.publish_dataset(local_dir, repo_id=..., tag=..., ...)` for HF Hub uploads.
- `whestbench.merge_datasets(input_dirs, output_dir=...)` — concatenate partial bakes.
- `whest dataset {bake, push, pull, merge, inspect}` CLI subcommands.
- Parallel bake via `--slice K/N` or `--mlp-range START-END` flags; merge with `whest dataset merge`.
- `whest run --dataset` now accepts HF Hub repos: `hf://owner/repo@v1` (inline revision) or `owner/repo --revision v1`.

### MIGRATION

- Legacy `.npz` datasets cannot be loaded by 0.3.0. Re-bake with `whest dataset bake` at the same `--seed` to reproduce.
- See `docs/reference/dataset-format.md` for the schema 3.0 specification.
