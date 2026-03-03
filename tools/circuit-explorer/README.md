<img src="../../assets/logo/logo.png" alt="Circuit Estimation Challenge logo" style="height: 80px;">

# Circuit Explorer

Interactive React app for visualizing small random circuits and estimator behavior.

> Circuit Explorer is optional.
> It is an educational and debugging aid, not the submission interface.

## 🚀 Quick Start

```bash
cd tools/circuit-explorer
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## ✅ What you should see

| Panel | Description |
|---|---|
| Circuit Graph | Wire/gate structure with value-oriented coloring |
| Signal Heatmap | Layer-wise activation behavior |
| Estimator Comparison | Per-layer error across available estimators |
| Controls | Width, depth, seed, and budget-like knobs |

## 🧪 Suggested use during estimator iteration

1. Reproduce a pattern on small circuits.
2. Inspect where errors spike by depth.
3. Convert that intuition into estimator logic.
4. Re-test with official local scorer.

Official local score behavior is defined by:

```bash
cestim run --estimator <path> --runner subprocess
```

## Participant docs

- [Documentation Index](../../docs/index.md)
- [Use Circuit Explorer](../../docs/how-to/use-circuit-explorer.md)
