# Circuit-to-MLP Refactor Design

**Date:** 2026-03-19
**Status:** Approved
**Approach:** A — In-place rename + rewrite

## Summary

Refactor the entire codebase from a circuit estimation contest platform to an MLP estimation contest platform. Rename package from `circuit_estimation` to `network_estimation`, CLI from `cestim` to `nestim`. Replace all domain objects, generation, simulation, scoring, and participant interfaces.

## Decisions

| Decision | Choice |
|----------|--------|
| Package name | `network_estimation`, CLI `nestim` |
| Domain model | MLP with He-init weight matrices, ReLU activation, Gaussian N(0,1) inputs |
| Estimation target | All-layer predictions returned as single `(depth, width)` array |
| Primary score | Final-layer MSE normalized by `sampling_mse` |
| Secondary score | All-layer MSE (diagnostic, not primary) |
| Budget model | Single budget per run (extensible to multi-budget later) |
| Spec model | Single `(width, depth)` per run |
| Participant interface | Class-based: `setup(context)` + `predict(mlp, budget) -> NDArray` |

## Domain Model (`domain.py`)

Replace `Layer`, `Circuit`, and `VectorizedCircuit` with:

```python
Weights = List[NDArray[np.float32]]  # list of (width, width) matrices, length = depth

@dataclass(frozen=True, slots=True)
class MLP:
    width: int
    depth: int
    weights: Weights  # len == depth, each shape (width, width)

    def validate(self) -> None:
        # width > 0, depth > 0, len(weights) == depth
        # each matrix is (width, width) float32
```

No wire indices, no coefficients. Just weight matrices.

## Generation (`generation.py`)

He-init MLP sampling:

```python
def sample_mlp(width: int, depth: int, rng=None) -> MLP:
    rng = rng or np.random.default_rng()
    scale = np.sqrt(2.0 / width)
    weights = [
        (rng.standard_normal((width, width)) * scale).astype(np.float32)
        for _ in range(depth)
    ]
    return MLP(width=width, depth=depth, weights=weights)
```

Scale = `sqrt(2/width)` per He init for ReLU networks, zero-mean Gaussian entries.

## Simulation (`simulation.py`)

MLP forward pass with ReLU:

```python
def relu(x: NDArray) -> NDArray:
    return np.maximum(x, 0.0)

def run_mlp(mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
    """Forward pass. Returns final-layer activations, shape (samples, width)."""
    x = inputs
    for w in mlp.weights:
        x = relu(x @ w)
    return x

def run_mlp_all_layers(mlp: MLP, inputs: NDArray[np.float32]) -> list[NDArray[np.float32]]:
    """Forward pass returning activations after each layer."""
    x = inputs
    layers = []
    for w in mlp.weights:
        x = relu(x @ w)
        layers.append(x)
    return layers

def output_stats(mlp, n_samples) -> tuple[NDArray, NDArray, float]:
    """Compute per-layer means and average variance of final layer.

    Returns:
        all_layer_means: shape (depth, width)
        final_mean: shape (width,)
        avg_variance: scalar for sampling_mse normalization
    """
    inputs = np.random.randn(n_samples, mlp.width).astype(np.float32)
    layer_outputs = run_mlp_all_layers(mlp, inputs)
    all_layer_means = np.stack([np.mean(out, axis=0) for out in layer_outputs])
    final_outputs = layer_outputs[-1]
    final_mean = np.mean(final_outputs, axis=0)
    avg_variance = float(np.mean(np.var(final_outputs, axis=0)))
    return all_layer_means, final_mean, avg_variance
```

Key: Gaussian N(0,1) inputs, float32 throughout, ReLU activation.

## Scoring (`scoring.py`)

### ContestSpec

```python
@dataclass(slots=True)
class ContestSpec:
    width: int
    depth: int
    n_mlps: int
    estimator_budget: int
    ground_truth_budget: int

default_spec = ContestSpec(
    width=256, depth=16, n_mlps=10,
    estimator_budget=256*256*4,
    ground_truth_budget=256*256*256,
)
```

### ContestData

```python
@dataclass(slots=True)
class ContestData:
    spec: ContestSpec
    mlps: list[MLP]
    all_layer_targets: list[NDArray[np.float32]]  # (depth, width) per MLP
    final_targets: list[NDArray[np.float32]]       # (width,) per MLP
    avg_variances: list[float]                      # per-MLP final-layer avg variance
```

`ground_truth_budget` is the `n_samples` passed to `output_stats()` when computing ground truth means and variances. It controls the accuracy of the ground truth — larger values give more precise targets.

### Scoring flow per MLP

1. **Baseline time** — draw `estimator_budget` random Gaussian inputs and run `run_mlp(mlp, inputs)`. Measure wall time. This wall time is the `time_budget` for this MLP.
2. **Call estimator** — `predictions = estimator.predict(mlp, budget)` returns `(depth, width)`.
3. **Time check** — if `time_spent > time_budget`, predictions become zeros.
4. **Time credit** — `fraction_spent = max(time_spent / time_budget, 0.5)`. Estimators faster than 50% of baseline get no additional credit.
5. **Final-layer score** — `sampling_mse = avg_variances[i] / (estimator_budget * fraction_spent)`, then `score = mean((final_predictions - final_targets) ** 2) / sampling_mse`. The MSE averages over the `width` dimension.
6. **Secondary all-layer score** — `mean((all_layer_predictions - all_layer_targets) ** 2) / sampling_mse`, averaging over both depth and width dimensions. Uses the same final-layer `sampling_mse` normalization. This is a diagnostic metric only.

Return: average primary score across all MLPs, plus report dict with per-MLP details.

## Participant Interface (`sdk.py`)

```python
@dataclass(frozen=True, slots=True)
class SetupContext:
    width: int
    depth: int
    estimator_budget: int
    api_version: str
    scratch_dir: str | None = None

class BaseEstimator(ABC):
    @abstractmethod
    def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
        """Return predicted means for all layers, shape (depth, width)."""
        raise NotImplementedError

    def setup(self, context: SetupContext) -> None:
        return None

    def teardown(self) -> None:
        return None
```

No streaming. Single array return.

## Supporting Modules

### `streaming.py` — Delete

No per-depth streaming. Validation of the `(depth, width)` return happens inline in scoring.

### `runner.py` — Simplify

- Remove `DepthRowOutcome` and streaming iterator protocol.
- `EstimatorRunner.predict(mlp, budget) -> NDArray` returns single array.
- `InProcessRunner` / `SubprocessRunner` — same structure, simplified predict.
- `_circuit_to_payload` becomes `_mlp_to_payload` (serialize weight matrices).
- Shape/finiteness validation on returned `(depth, width)` array.

### `dataset.py` — Adapt to MLP

- Pack weight matrices instead of circuit wire indices + coefficients.
- Store `all_layer_means`, `final_means`, and `avg_variance`.
- `load_dataset` reconstructs `MLP` objects from weight arrays.

### `estimators.py` — Rewrite reference estimators

- `MeanPropagationEstimator` — propagate means through ReLU layers analytically.
- `CovariancePropagationEstimator` — propagate mean + covariance through ReLU.
- `CombinedEstimator` — budget-aware routing (same pattern).
- Math changes entirely: ReLU moments instead of bilinear gate moments.

### `protocol.py` — Minor field updates

Match new `ContestSpec` fields.

### `hardware.py` — No change

Domain-agnostic hardware fingerprinting. Survives as-is.

### `loader.py` — Rename internals

Update the dynamic module naming prefix from `_circuit_estimation_submission_` to `_network_estimation_submission_`. No structural change.

### `reporting.py` — Text updates

Replace all circuit-domain terminology ("Circuit Estimation Report", `cestim` references, circuit labels) with MLP/network equivalents. Structural layout stays the same.

### `subprocess_worker.py` — Rewrite

Replace circuit deserialization (`_payload_to_circuit`) with MLP deserialization (`_payload_to_mlp`). Update `SetupContext` construction to use new fields (`depth`, `estimator_budget` instead of `max_depth`, `budgets`, `time_tolerance`). Remove streaming depth-row protocol — worker returns a single `(depth, width)` array per predict call.

### `visualizer.py` — Text updates

Update "circuit-explorer" references to "network-explorer" or equivalent. Functionality unchanged.

### `__init__.py` — Update exports

Replace `Circuit`, `Layer`, `VectorizedCircuit`, `random_circuit`, `random_gates` with `MLP`, `sample_mlp`, and new public API.

### `main.py` — Update import

Change `from circuit_estimation.cli` to `from network_estimation.cli`.

### `cli.py` — Rename

Entrypoint `cestim` becomes `nestim`. Update subcommands and help text.

### `examples/estimators/` — Rewrite

All four examples rewritten for MLP interface.

### `tests/` — Rewrite

All tests rewritten for MLP domain objects. Same coverage goals.

### `docs/` — Full update

Problem setup, scoring model, estimator contract, CLI reference, how-to guides, README.

### `pyproject.toml` — Rename

Package name, `[project.scripts]` entries: `cestim` becomes `nestim`, `circuit-estimation` becomes `network-estimation`.

## Modules Deleted

- `streaming.py` — no longer needed

## Modules Renamed/Rewritten (all in `src/network_estimation/`)

| Module | Change |
|--------|--------|
| `__init__.py` | Update exports: `MLP`, `sample_mlp`, new public API |
| `domain.py` | `Layer`/`Circuit`/`VectorizedCircuit` → `MLP` |
| `generation.py` | `random_gates`/`random_circuit` → `sample_mlp` |
| `simulation.py` | Circuit forward pass → MLP + ReLU forward pass |
| `scoring.py` | Multi-budget per-depth → single-budget final-layer scoring |
| `sdk.py` | `predict(Circuit, int) -> Iterator` → `predict(MLP, int) -> NDArray` |
| `runner.py` | Drop streaming, simplify to single-array predict |
| `subprocess_worker.py` | Circuit deserialization → MLP deserialization, drop streaming protocol |
| `dataset.py` | Circuit packing → weight matrix packing |
| `estimators.py` | Bilinear gate math → ReLU moment propagation math |
| `protocol.py` | Field updates |
| `hardware.py` | No change (domain-agnostic) |
| `loader.py` | Rename dynamic module prefix |
| `reporting.py` | Replace circuit terminology with MLP/network terminology |
| `visualizer.py` | Update "circuit-explorer" references |
| `cli.py` | `cestim` → `nestim` |
| `main.py` | Update import path |
| `pyproject.toml` | Package name, script entries |
