---
configs:
- config_name: default
  data_files:
  - path: data/public-*.parquet
    split: public
- config_name: holdout
  data_files:
  - path: data/holdout-*.parquet
    split: holdout
homepage: https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026
language:
- code
license: cc-by-4.0
pretty_name: 'WhestBench 2026: ARC White-Box Estimation Challenge'
repository: https://github.com/AIcrowd/whestbench
size_categories:
- n<1K
tags:
- whestbench
- alignment
- neural-network-statistics
- benchmark
- white-box
- multi-split
task_categories:
- other
---

<p align="center">
  <a href="https://github.com/AIcrowd/whestbench">
    <img src="https://raw.githubusercontent.com/AIcrowd/whestbench/main/assets/logo/logo.png" width="320" alt="WhestBench logo">
  </a>
</p>

<p align="center">
  Organized by:
  <a href="https://www.alignment.org/"><b>Alignment Research Center (ARC)</b></a>,
  <a href="https://www.aicrowd.com/"><b>AIcrowd</b></a>
</p>

# WhestBench 2026: ARC White-Box Estimation Challenge

<p align="center">
  <a href="https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026"><img alt="Challenge" src="https://img.shields.io/badge/AIcrowd-Challenge_Page-f0524d?style=for-the-badge"></a>
  <a href="https://github.com/AIcrowd/whestbench"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-AIcrowd%2Fwhestbench-181717?style=for-the-badge&logo=github&logoColor=white"></a>
  <a href="https://github.com/AIcrowd/whest-starterkit"><img alt="Starter Kit" src="https://img.shields.io/badge/Starter_Kit-whest--starterkit-f57c00?style=for-the-badge&logo=github&logoColor=white"></a>
  <a href="https://aicrowd.github.io/whestbench-explorer/"><img alt="MLP Explorer" src="https://img.shields.io/badge/MLP_Explorer-Interactive-7e57c2?style=for-the-badge"></a>
  <a href="https://github.com/AIcrowd/flopscope"><img alt="flopscope" src="https://img.shields.io/badge/FLOP_Tracking-flopscope-009688?style=for-the-badge&logo=github&logoColor=white"></a>
  <a href="https://huggingface.co/datasets/<your-repo>/tree/main"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97-View_on_HF_Hub-ffd54f?style=for-the-badge"></a>
</p>

WhestBench is a benchmark for *white-box activation estimation*: given the weights of a randomly initialized ReLU multi-layer perceptron (MLP) and a strict floating-point-operation (FLOP) budget, predict the average post-activation value of every neuron when the network is fed standard Gaussian inputs.

**This is the WhestBench 2026 Evaluation Dataset** — per-round MLP ground-truth that powers the leaderboards. It contains disjoint splits with different visibility:

- **`public`** (2 MLPs) — submissions are scored against this split with scores visible to participants on the **public leaderboard** in real time.
- **`holdout`** (2 MLPs) — submissions are also scored against this split, but the scores power the **private/final leaderboard** and are revealed only at the conclusion of the round.

This dataset is intentionally not part of the Public Release at [`aicrowd/arc-whestbench-2026`](https://huggingface.co/datasets/aicrowd/arc-whestbench-2026); develop your estimator against the public release first.

## Quick start

The pure HuggingFace path (no whestbench install required):
```python
from datasets import load_dataset

# Load all splits
dsd = load_dataset("<your-repo>", revision="main")
print(dsd)

# Or load one split:
ds = load_dataset("<your-repo>", revision="main", split="public")
print(ds[0]["mlp_name"])
```

The whestbench convenience wrapper (adds schema validation + `metadata.json` access):

```python
import whestbench

ds = whestbench.load_dataset("<your-repo>", revision="main")
for mlp in whestbench.iter_mlps(ds):
    # `mlp` is a whestbench.MLP with .weights, .seed, .name, .width, .depth
    ...

provenance = whestbench.metadata(ds)
print(provenance["n_samples"], provenance["created_at_utc"])
```

Run an estimator end-to-end via the CLI:
```bash
whest run \
    --estimator my_estimator.py \
    --dataset hf://<your-repo>@main
# (automatically uses metadata.default_split: "public"; pass --split <name> to override)
```

➡️ **New to the challenge?** Head over to the **[WhestBench starter kit](https://github.com/AIcrowd/whest-starterkit)** for a worked example estimator, the recommended project layout, FLOP-tracking patterns with [`flopscope`](https://github.com/AIcrowd/flopscope), local testing tips, and the submission workflow.


## Schema

Each row is one MLP. Eight columns:

| Column | Type / shape | What this is |
|---|---|---|
| `mlp_id` | `int32` | 0-based index of this MLP within the dataset (the absolute index across all parallel-bake slices). |
| `mlp_name` | `string` | Stable, deterministic human-readable slug like `"danielle-johnson"`, derived from `mlp_seed`. Useful for log lines; carries no information beyond `mlp_seed`. |
| `mlp_seed` | `int64` | Per-MLP seed exposed in the dataset. Under seed_protocol 3.0 (`whestbench_explicit_per_mlp_seeds`), this is the **input** seed for the MLP — `MLP.seed` (the estimator seed) is derived locally. Under seed_protocol 2.0 (`whestbench_seedsequence_hierarchy`, legacy), this is the **derived** estimator seed itself. Estimators read `mlp.seed` and see the same kind of value in both protocols (a deterministic int derived from the dataset's seed material). |
| `weights` | `float32[depth, width, width]` | The MLP's layer weight matrices. The network has **no biases** and **no separate linear output layer** — every weight matrix is followed by a ReLU. Layer `l` computes `h_l(x) = max(0, W_l @ h_{l-1}(x))`. Weights are drawn i.i.d. from `N(0, 2/width)` (He initialization) at bake time. |
| `all_layer_means` | `float32[depth, width]` | **Ground truth.** Entry `[l, j]` is the empirical mean of neuron `j`'s post-ReLU output at layer `l`, averaged over **N = 100** independent Gaussian inputs: `E_{x ~ N(0, I)}[ h_l(x)_j ] ≈ (1/N) Σ_i h_l(x_i)_j`. Computed by direct Monte Carlo. **This is what your estimator predicts.** |
| `final_means` | `float32[width]` | The last row of `all_layer_means` — i.e. `E[h_{depth}(x)_j]` for each output neuron `j`, again over N = 100 samples. Materialised as its own column because the **primary scoring metric** (`final_layer_mse`) only looks at this row. |
| `avg_variance` | `float64` | Per-MLP mean of the per-neuron output variance at the final layer: `(1/width) Σ_j Var[h_{depth}(x)_j]`. A single scalar per MLP, computed alongside the means over the same Monte Carlo draws. Shipped as **diagnostic provenance** — useful for normalising your own MSE locally or as input to variance-aware estimators. **Not** consumed by the active scoring formula (the score is `mse_final · max(0.1, C_m / B_m)`). |
| `sampling_budget_breakdown` | `string` (JSON) | FLOP accounting for the bake that produced the ground truth for **this** row — useful as provenance. **Not** related to the estimator's FLOP budget at evaluation time. Decode with `json.loads(...)` to get a dict with keys: `flop_budget`, `flops_used`, `flops_remaining`, `wall_time_s`, `flopscope_backend_time_s`, `flopscope_overhead_time_s`, `residual_wall_time_s`, and `by_namespace` (per-namespace nested breakdown: each namespace key maps to its own `{flops_used, calls, flopscope_backend_time_s, flopscope_overhead_time_s, operations}`). |



## How the ground truth was made

> **Monte Carlo with N = 100 samples per MLP.** Every entry in `all_layer_means` and `final_means` is the empirical mean over this many independent standard-Gaussian input draws. The production WhestBench 2026 release uses **N = 10⁹**, where the per-neuron standard deviation of the published value is `σ_activation/√N ≈ 3×10⁻⁵` (in activation units; equivalently, ground-truth MSE on the order of `10⁻⁹`) — orders of magnitude smaller than any meaningful estimator gap.

**Input distribution.** Every Monte Carlo sample is a fresh `x ~ N(0, I)` of shape `(width,)`. The same input is forward-propagated through all `depth` layers in one pass, so the per-layer means at indices `[0..depth-1]` share the same input draws.

**Estimator.** Sums of post-ReLU activations are accumulated in `float64` for numerical stability, then divided by N at the end and downcast to `float32`. The final-layer variance scalar (`avg_variance`) comes from `E[h²] - (E[h])²` over the same N draws.

**Compute.** Sampling is chunked so memory stays bounded (`~4MB` per chunk on the flopscope CPU backend; tunable on the torch GPU backend via `chunk_size`). On the torch backend, under matched determinism config (`torch.use_deterministic_algorithms=True`, `cudnn.deterministic=True`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`) on a fixed `torch` version and GPU architecture, the bake is **bit-exact**: `weights`, `all_layer_means`, and `final_means` reproduce byte-for-byte; `avg_variance` agrees to `1e-12` relative tolerance (1 float64 ULP from the `(sum_sq/n − mean²)` step). The two backends (flopscope CPU and torch GPU) produce statistically equivalent output: means agree within `~3×10⁻⁵` at N=10⁹.

## Dataset summary

|  |  |
|---|---|
| Splits | public, holdout |
| MLPs total | 4 |
| `public` split MLPs | 2 |
| `holdout` split MLPs | 2 |
| Width | 8 |
| Depth | 2 |
| Monte Carlo samples per MLP (N) | **100** |
| Schema version | 3.0 |
| Seed protocol | `whestbench_explicit_per_mlp_seeds` v3.0 |

## Working with this dataset

Three ways in, fastest first:

- **Score an estimator** — `whest run --dataset hf://<your-repo>@main` runs your estimator end-to-end against the default `public` split (add `--split <name>` to switch). No manual download step — the data is fetched and cached for you.
- **Load in Python** — `whestbench.load_dataset("<your-repo>", revision="main", split="public")` returns a HuggingFace `Dataset` with schema validation plus `whestbench.metadata(ds)` provenance; iterate ready-to-use MLP objects with `whestbench.iter_mlps(ds)`.
- **Raw rows** — plain `datasets.load_dataset(...)` works too, if you'd rather not install whestbench (see [Quick start](#quick-start)).

📖 **Full walkthrough** — choosing a split, FLOP budgeting, the estimator contract, and the submission flow — see **[Use Evaluation Datasets](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/how-to/use-evaluation-datasets.md)** in the starter kit.

## Reproducibility

This dataset was baked with:

- **Backend:** `flopscope`
- **Seed protocol:** `whestbench_explicit_per_mlp_seeds` v3.0

- **Created (UTC):** `2026-06-15T22:19:28.040384+00:00`



This dataset was baked with **seed_protocol 3.0 (explicit per-MLP seeds)**: every MLP's input seed lives in the parquet `mlp_seed` column, so the bake is fully reproducible from the published data alone. You almost never need to do this — but if you want to regenerate the dataset byte-for-byte, expand the recipe below.

<details>
<summary><b>Re-bake this dataset locally from its seeds</b></summary>

**1. Extract the per-MLP seeds** into one JSON file per split:

```python
import json

import whestbench

for split in ["public", "holdout"]:
    ds = whestbench.load_dataset("<your-repo>", revision="main", split=split)
    json.dump([int(r["mlp_seed"]) for r in ds], open(f"{split}-seeds.json", "w"))
```

**2. Re-bake** each split with the same seeds:

```bash
whest dataset bake \
    --n-mlps 2 \
    --n-samples 100 \
    --width 8 --depth 2 \
    --split public \
    --mlp-seeds public-seeds.json \
    --output ./public-rebake
whest dataset bake \
    --n-mlps 2 \
    --n-samples 100 \
    --width 8 --depth 2 \
    --split holdout \
    --mlp-seeds holdout-seeds.json \
    --output ./holdout-rebake

```

</details>


The pins required for the bit-exactness guarantee (see *Compute* above) are recorded in `metadata.bake_config` alongside the `whestbench` and `faker` versions.

## Provenance


Single-host bake on `arm`.

## Citation

If you use this dataset, please cite the challenge:


```bibtex
@misc{whestbench2026,
  title        = {{WhestBench 2026: ARC White-Box Estimation Challenge}},
  author       = {{Alignment Research Center} and {AIcrowd}},
  year         = {2026},
  howpublished = {\url{https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026}},
}
```


## License

Released under **CC-BY-4.0**. Use is encouraged for research, competition entries, and educational material; please credit the WhestBench team and the AIcrowd challenge.