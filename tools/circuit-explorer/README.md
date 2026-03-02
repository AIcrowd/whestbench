# Circuit Explorer

Interactive React app for visualizing small random circuits and estimator behavior.

> Circuit Explorer is optional.
> It is an educational and debugging aid, not the submission interface.

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
| Circuit Graph | Wire/gate structure with value-oriented coloring |
| Signal Heatmap | Layer-wise activation behavior |
| Estimator Comparison | Per-layer error across available estimators |
| Controls | Width, depth, seed, and budget-like knobs |

## Relationship to Python Scoring

Explorer logic mirrors the same core gate algebra used by the Python package.
Use it to build intuition and debug ideas.

Official local score behavior is defined by:

```bash
cestim run --estimator <path> --runner subprocess
```

## Participant Docs

For the core participant workflow, start here:

- [Documentation Index](../../docs/index.md)
- [Use Circuit Explorer](../../docs/how-to/use-circuit-explorer.md)
