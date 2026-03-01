# Circuit Explorer

Interactive React app for visualizing small Boolean circuits — the core problem behind the [Mechanistic Estimation Challenge](https://www.alignment.org/blog/competing-with-sampling/).

> **This is a developer/educational tool**, not part of the competition submission interface.
> Participants working on the core problem can safely ignore this directory.

## Quick Start

```bash
cd tools/circuit-explorer
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## What It Shows

| Panel | Description |
|---|---|
| **Circuit Graph** | SVG visualization of wires + gates, color-coded by mean value |
| **Signal Heatmap** | How wire statistics evolve across layers |
| **Estimator Comparison** | Per-layer MSE: sampling (brute-force) vs. mean propagation (analytic) |
| **Controls** | Width, depth, seed, and sampling budget sliders |

## How It Relates to the Python Code

`circuit.js` and `estimators.js` are JavaScript ports of `circuit.py` and `estimators.py` (mean propagation only). They implement the same circuit generation and forward pass logic, using the same gate algebra:

```
output = const + first_coeff * x + second_coeff * y + product_coeff * x * y
```

This lets you explore the problem visually without running any Python.
