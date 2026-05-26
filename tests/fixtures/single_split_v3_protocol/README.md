---
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

WhestBench is a benchmark for *white-box activation estimation*: given the weights of a small ReLU multi-layer perceptron (MLP) and a strict floating-point-operation (FLOP) budget, predict the average post-activation value of every neuron when the network is fed standard Gaussian inputs.

**This is the Public Dataset Release for WhestBench 2026** — pre-baked MLPs paired with their ground-truth activation statistics, free to download and use while you develop your estimator. The actual contest evaluation runs against the private `public` and `holdout` splits of [`aicrowd/arc-whestbench-2026-evals`](https://huggingface.co/datasets/aicrowd/arc-whestbench-2026-evals), released per round under the same schema.

## Quick start

The pure HuggingFace path (no whestbench install required):

```python
from datasets import load_dataset

ds = load_dataset(
    "<your-repo>",
    revision="main",
    split="public",
)
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
```

➡️ **New to the challenge?** Head over to the **[WhestBench starter kit](https://github.com/AIcrowd/whest-starterkit)** for a worked example estimator, the recommended project layout, FLOP-tracking patterns with [`flopscope`](https://github.com/AIcrowd/flopscope), local testing tips, and the submission workflow.


## What's in this dataset

Each row is one MLP, paired with the Monte-Carlo–computed ground-truth statistics of its post-activation outputs. Four things travel together per row:

1. The **MLP weights** (the network you'll analyse).
2. The **per-layer ground-truth means** — what your estimator is trying to predict. Computed by direct Monte Carlo over **N = 100** independent standard-Gaussian input draws per MLP (production WhestBench 2026 uses **N = 10⁹**, for standard errors `~1/√N ≈ 3×10⁻⁵` per neuron).
3. A **per-MLP variance scalar** — final-layer mean per-neuron variance, computed alongside the means over the same N draws. Shipped as diagnostic provenance; not consumed by the score.
4. A **per-MLP seed** — passed to your estimator if it uses any randomness, so submissions reproduce under regrade.

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
| `sampling_budget_breakdown` | `string` (JSON) | FLOP accounting for the bake that produced the ground truth for **this** row — useful as provenance. **Not** related to the estimator's FLOP budget at evaluation time. Decode with `json.loads(...)`. |


## Your task as a participant

You implement a class with a `predict(mlp: MLP, budget: int) -> ndarray[depth, width]` method that returns your estimate of `all_layer_means` for the given MLP. The harness gives you the weights and a strict per-MLP FLOP budget `B_m` — **B_m = 6.8 × 10¹⁰** for the initial competition configuration (roughly 6.5 × 10⁴ Monte Carlo forward passes' worth of compute).

Your spend is tracked as **effective compute**:

```
C_m = F_m + λ · R_m
```

where `F_m` is analytical FLOPs counted by [`flopscope`](https://github.com/AIcrowd/flopscope), `R_m` is residual wall-clock time spent in uninstrumented code, and `λ = 10¹¹` FLOPs/s is the per-phase wall-time-to-FLOPs conversion rate. Going outside `flopscope` is allowed; you just pay for it.

Your **primary score** is final-layer MSE — your prediction's last row vs. `final_means` — multiplied by a budget-adjusted factor:

```
s_m = mse_final · max(0.1, C_m / B_m)
```

so that staying well under budget pays off (down to a factor-of-ten cap). The **earlier-layer rows** (`all_layer_means[0..depth-2]`) are a **diagnostic** — they show where approximation error accumulates across layers but do not enter the primary score.

If your estimator goes over budget, raises, returns non-finite or wrong-shape output, or trips an operational guard (per-MLP wall-clock cap, memory cap), the grader **zeros your prediction for that MLP AND forces the multiplier to 1.0** — no compute discount on failed runs. Other MLPs in the suite are unaffected.

See the [starter kit](https://github.com/AIcrowd/whest-starterkit) for a worked end-to-end example, plus baselines for Monte Carlo sampling, mean propagation, covariance propagation, and a budget-aware combined estimator.


## How the ground truth was made

> **Monte Carlo with N = 100 samples per MLP.** Every entry in `all_layer_means` and `final_means` is the empirical mean over this many independent standard-Gaussian input draws. The production WhestBench 2026 release uses **N = 10⁹**, which gives standard errors on the order of `1/√N ≈ 3×10⁻⁵` per neuron — well below any meaningful estimator gap.

**Input distribution.** Every Monte Carlo sample is a fresh `x ~ N(0, I)` of shape `(width,)`. The same input is forward-propagated through all `depth` layers in one pass, so the per-layer means at indices `[0..depth-1]` share the same input draws.

**Estimator.** Sums of post-ReLU activations are accumulated in `float64` for numerical stability, then divided by N at the end and downcast to `float32`. The final-layer variance scalar (`avg_variance`) comes from `E[h²] - (E[h])²` over the same N draws.

**Compute.** Sampling is chunked so memory stays bounded (`~4MB` per chunk on the flopscope CPU backend; tunable on the torch GPU backend via `chunk_size`). The two backends produce *statistically* equivalent output (means agree within `~3×10⁻⁵` at N = 10⁹); the flopscope path is additionally bit-exact run-to-run at the same seed.

## Dataset summary

|  |  |
|---|---|
| Split | `public` |
| MLPs | 4 |
| Width | 4 |
| Depth | 2 |
| Monte Carlo samples per MLP (N) | **100** |
| Schema version | 3.0 |
| Seed protocol | `whestbench_explicit_per_mlp_seeds` v3.0 |

## Reproducibility

This dataset was baked with:

- **Backend:** `flopscope`
- **Seed protocol:** `whestbench_explicit_per_mlp_seeds` v3.0

- **Created (UTC):** `2026-05-26T00:00:00+00:00`



This dataset was baked with **seed_protocol 3.0 (explicit per-MLP seeds)**. Each MLP's seed is recorded in the parquet `mlp_seed` column. To re-bake locally with the same seed list:

```bash
# Extract the per-MLP seeds from the published dataset:
python -c "
import json
from datasets import load_dataset
ds = load_dataset('<your-repo>', revision='main')

seeds = [int(row['mlp_seed']) for row in ds['public']]
open('seeds.json', 'w').write(json.dumps(seeds))

"

# Re-bake:

whest dataset bake \
    --n-mlps 4 \
    --n-samples 100 \
    --width 4 --depth 2 \
    --split public \
    --mlp-seeds seeds.json \
    --output ./rebake

```


Bit-exact reproducibility requires the same `whestbench` and `faker` versions pinned at bake time; statistical reproducibility (means within `~3e-5`) holds across hardware on the `torch` backend.

## Provenance


Single-host bake on `unspecified host`.


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