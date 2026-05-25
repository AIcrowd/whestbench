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
