# How To Use Circuit Explorer

Circuit Explorer is an interactive UI for building intuition about circuit dynamics and estimator behavior.

It is a learning and debugging tool, not the submission interface.

## Start Explorer

```bash
cd tools/circuit-explorer
npm install
npm run dev
```

Open `http://localhost:5173`.

## What To Look At

- Circuit Graph: wire and gate structure with value-oriented coloring
- Signal Heatmap: how layer-wise activations evolve
- Estimator Comparison: per-layer error across estimators
- Controls: width, depth, random seed, and budget-like knobs

## Suggested Workflow

1. Keep circuits small at first (`width` and `depth` low).
2. Change seed to see structural variability.
3. Compare simple estimator behavior against sampling references.
4. Observe where error concentrates by layer/gate type.
5. Use those observations to design budget-aware heuristics in Python estimator code.

## How It Connects To Python Runtime

Explorer logic mirrors the same core gate algebra used in the Python package.
That makes it useful for intuition, while official scoring still comes from `cestim run` and the evaluator.

For implementation and scoring details, see:

- [How To Write Your Own Estimator](how-to-write-your-own-estimator.md)
- [What Is The Problem And How Is It Scored?](what-is-the-problem-and-how-is-it-scored.md)
