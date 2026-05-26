# Changelog

## Unreleased

### Added

- `generate_readme(... companion_repo=...)`: new keyword argument that controls
  the cross-link text between paired public-release and evals datasets (e.g. a
  public-release README now points at the evals repo it pairs with). When unset,
  falls back to the canonical `aicrowd/arc-whestbench-2026[-evals]` literals so
  the change is backwards-compatible.
- `metadata.partials_count` recorded by `merge_datasets`: total number of
  partials merged (used by the new Provenance summary).

### Changed

- `merge_datasets` now collapses `hardware_fingerprints` by **coarse hardware
  signature** (GPU + capability + torch version + determinism env), rather than
  one entry per partial. The previous one-bullet-per-partial Provenance render
  was unusable at production scale (1000 single-MLP partials → 1000-line
  README). Each collapsed entry carries the full hardware tech stack
  (RAM, CPU, Python/Torch/NumPy/whestbench/flopscope versions, determinism
  flags) plus `mlp_count`, `drivers_seen` (aggregated CUDA driver versions),
  `kernels_seen` (aggregated kernel patches), and `representative_hostnames`.
  The per-partial `mlp_range` field is dropped from collapsed entries.
- Dataset card template `templates/dataset_card.md.j2`: Provenance section
  renders rich per-signature blocks instead of a per-partial bullet list.
  Reproducibility wording strengthened — `weights`, `all_layer_means`,
  `final_means` are documented as **bit-exact** under matched determinism env
  on a fixed `torch` version + GPU architecture. When
  `metadata.cross_driver_verified` is True, additionally states cross-driver
  bit-equivalence is empirically verified.
- Dataset card template: hardcoded companion-repo names replaced with the
  `companion_repo` template variable (backwards-compatible fallback).

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
