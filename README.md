# whestbench

Library + CLI for the ARC Whitebox Estimation Challenge. Provides:

- `whestbench` Python package: `BaseEstimator` SDK, `MLP` domain model, simulation primitives, scoring runners.
- `whest` CLI: `validate`, `run`, `package`, `doctor`, `init`.

## For challenge participants

You almost certainly want the starter kit, not this repo:

→ **[github.com/AIcrowd/whest-starterkit](https://github.com/AIcrowd/whest-starterkit)** ←

It walks you through the **ladder of formality** — start with `python estimator.py`, climb to `whest run --runner docker` and `whest package` when you're ready.

## For library consumers

Install the latest tagged release:

```bash
uv add "whestbench @ git+https://github.com/AIcrowd/whestbench.git@vX.Y.Z"
```

(Replace `vX.Y.Z` with the latest tag.)

For local development:

```bash
git clone https://github.com/AIcrowd/whestbench.git
cd whestbench
uv sync --group dev
uv run pytest
```

## SDK reference

```python
import whest as we
from whestbench import BaseEstimator, MLP

class MyEstimator(BaseEstimator):
    def predict(self, mlp: MLP, budget: int) -> we.ndarray:
        return we.zeros((mlp.depth, mlp.width))
```

See `src/whestbench/sdk.py` for the full contract (`predict`, `setup`, `teardown`, `SetupContext`).

## CLI reference

See [docs/reference/cli-reference.md](docs/reference/cli-reference.md) for the full surface.

Common commands:

```bash
whest doctor                                    # diagnose your environment
whest init my-estimator/                        # scaffold a fresh estimator dir
whest validate --estimator my_estimator.py      # 30-second sanity check
whest run --estimator my_estimator.py --n-mlps 3   # local scoring
whest package --estimator my_estimator.py       # build a submission tarball
```

## Repository layout

```
src/whestbench/    # library + CLI source
docs/reference/    # canonical CLI + whest-internals reference
tests/             # unit + integration tests
tools/             # internal dev tools (whestbench-explorer, etc.)
profiling/         # internal benchmarks
```

## Releases

Tagged releases are the public API. The `whest-starterkit` pins a specific tag (or commit SHA); bump that pin to roll out new features to participants.

## Contributing

This repo is the library. Participant-facing examples and docs live in [whest-starterkit](https://github.com/AIcrowd/whest-starterkit). Library and CLI changes go here.

## License

See [LICENSE](LICENSE).
