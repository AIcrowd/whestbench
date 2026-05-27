# Datasets — a complete guide

WhestBench uses [HuggingFace Datasets](https://huggingface.co/docs/datasets) as
its dataset format and HF Hub as the distribution channel. This guide walks you
through every dataset-related verb in `whest`, in the order you'd typically
encounter them.

> If you only have 5 minutes, read the **Quick start** below. The rest of the
> guide builds on it.

## Quick start

You have a working estimator at `./estimator.py`. Bake a tiny evaluation
dataset locally, then score against it:

```bash
# 1. Generate 10 MLPs with ground-truth statistics → ./my-eval/
whest dataset bake --n-mlps 10 --n-samples 1000 --width 64 --depth 4 \
                   --output ./my-eval

# 2. Inspect what got written
whest dataset info ./my-eval

# 3. Score your estimator against the same MLPs every run
whest run --estimator estimator.py --dataset ./my-eval
```

Why this matters: without `--dataset`, `whest run` regenerates MLPs and ground
truth on every invocation. Baking a dataset once and reusing it makes your runs
deterministic and ~10× faster.

[Continue to: lifecycle ↓](#the-dataset-lifecycle)
