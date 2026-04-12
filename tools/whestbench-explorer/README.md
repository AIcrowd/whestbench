<img src="../../assets/logo/logo.png" alt="ARC Whitebox Estimation Challenge logo" style="height: 80px;">

# WhestBench Explorer

Interactive React app for visualizing small random MLPs and estimator behavior.

> WhestBench Explorer is optional.
> It is an educational and debugging aid, not the submission interface.

## Quick Start

```bash
cd tools/whestbench-explorer
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## What you should see

| Panel | Description |
|---|---|
| Network Graph | Layer/neuron structure with value-oriented coloring |
| Signal Heatmap | Layer-wise activation behavior |
| Estimator Comparison | Per-layer error across available estimators |
| Controls | Width, depth, seed, and budget knobs |

## Suggested use during estimator iteration

1. Reproduce a pattern on small MLPs.
2. Inspect where errors spike by depth.
3. Convert that intuition into estimator logic.
4. Re-test with official local scorer.

Official local score behavior is defined by:

```bash
whest run --estimator <path> --runner server
```

## Participant docs

- [Documentation Index](../../docs/index.md)
- [Use WhestBench Explorer](../../docs/how-to/use-whestbench-explorer.md)
