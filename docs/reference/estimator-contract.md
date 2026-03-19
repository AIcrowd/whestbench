# Estimator Contract

## When to use this page

Use this page when you need exact estimator I/O requirements.

## Required interface

`predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]`

Optional lifecycle hooks:

- `setup(self, context: SetupContext) -> None`
- `teardown(self) -> None`

### `SetupContext` fields

| Field | Type | Description |
|---|---|---|
| `width` | `int` | Neuron count for generated MLPs |
| `max_depth` | `int` | Number of layers per MLP |
| `budgets` | `tuple[int, ...]` | Sampling budgets used during evaluation |
| `time_tolerance` | `float` | Relative slack for timeout/floor semantics |
| `api_version` | `str` | Contract version string |
| `scratch_dir` | `str \| None` | Optional writable directory for caching |

## Input object quick reference

| Object | Field | Meaning |
|---|---|---|
| `MLP` | `width` | Number of neurons per layer |
| `MLP` | `depth` | Number of weight matrices (layers) |
| `MLP` | `weights` | Ordered weight matrices, each `(width, width)` |

For traversal examples, see [Inspect and Traverse MLP Structure](../how-to/inspect-circuit-structure.md).

## Output requirements per `predict` call

| Requirement | Rule |
|---|---|
| Shape | Return a 2D array with shape `(mlp.depth, mlp.width)` |
| Numeric validity | Every value is finite |

## Failure semantics

When validation fails (wrong shape, non-finite values), the affected prediction is treated as a **zero-filled row**. The scoring loop continues and produces a valid report -- errors are reflected as increased MSE rather than hard failures.

## Next step

- [Write an Estimator](../how-to/write-an-estimator.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
