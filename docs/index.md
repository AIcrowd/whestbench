# Whestbench docs

> Participant-facing curriculum (getting-started, concepts, how-to,
> troubleshooting) lives in the [whest-starterkit](https://github.com/AIcrowd/whest-starterkit).
> This site documents the library and CLI surface only.

## Library reference

- [CLI reference](reference/cli-reference.md)
- [Estimator contract](reference/estimator-contract.md)
- [Score report fields](reference/score-report-fields.md)
- [Code patterns (flopscope cheat sheet)](reference/code-patterns.md)
- [Flopscope primer (BudgetContext, FLOP costs)](reference/flopscope-primer.md)

## Source map

- `src/whestbench/cli.py` — `whest` / `whestbench` entry point
- `src/whestbench/domain.py` — `MLP`, `SetupContext`, scoring contracts
- `src/whestbench/estimators.py` — `BaseEstimator` + reference impls
- `src/whestbench/generation.py` — `sample_mlp`
- `src/whestbench/simulation.py` — Monte Carlo ground truth via flopscope
- `src/whestbench/scoring.py` — `evaluate_estimator`, `ContestSpec`
- `src/whestbench/reporting.py` — Rich score report
- `src/whestbench/protocol.py` — server-runner JSON protocol

## Releases

See [`RELEASING.md`](RELEASING.md).

Underlying FLOP-counting library: [`AIcrowd/flopscope`](https://github.com/AIcrowd/flopscope).
