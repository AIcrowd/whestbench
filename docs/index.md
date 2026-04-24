# whestbench Documentation

This repo's docs cover the **library and CLI internals only**.

## Participant docs moved

The participant-facing tutorial, examples, and stage-by-stage walkthroughs now live in [whest-starterkit](https://github.com/AIcrowd/whest-starterkit).

## Library / CLI reference

- [reference/cli-reference.md](reference/cli-reference.md) — full `whest` CLI surface
- [reference/whest-primer.md](reference/whest-primer.md) — whest internals overview
- [reference/code-patterns.md](reference/code-patterns.md) — common code patterns inside the library
- [reference/estimator-contract.md](reference/estimator-contract.md) — `BaseEstimator` SDK contract
- [reference/score-report-fields.md](reference/score-report-fields.md) — score-report schema

## Source map

- `src/whestbench/cli.py` — CLI entry point
- `src/whestbench/sdk.py` — `BaseEstimator` contract
- `src/whestbench/domain.py` — `MLP` dataclass
- `src/whestbench/simulation.py` — forward-pass primitives
- `src/whestbench/runners/` — local, subprocess, (future) docker runners
- `src/whestbench/scoring.py` — score computation
- `src/whestbench/templates/estimator.py.tmpl` — what `whest init` scaffolds
