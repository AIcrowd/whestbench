# Estimator Contract

## 📌 When to use this page

Use this page when you need exact estimator I/O requirements.

## Required interface

`predict(self, circuit: Circuit, budget: int) -> Iterator[NDArray[np.float32]]`

Optional lifecycle hooks:

- `setup(self, context: SetupContext) -> None`
- `teardown(self) -> None`

## Input object quick reference

| Object | Field | Meaning |
|---|---|---|
| `Circuit` | `n` | Number of wires per layer |
| `Circuit` | `d` | Number of transition layers |
| `Circuit` | `gates` | Ordered `Layer` list of length `d` |
| `Circuit` | `to_vectorized()` | Packed depth-major tensors for fast traversal |
| `Layer` | `first`, `second` | Parent wire indices per output wire |
| `Layer` | `first_coeff`, `second_coeff`, `const`, `product_coeff` | Per-wire gate coefficients |

For traversal examples, see [Inspect and Traverse Circuit Structure](../how-to/inspect-circuit-structure.md).

## Output requirements per `predict` call

| Requirement | Rule |
|---|---|
| Row count | Emit exactly `circuit.d` rows |
| Row shape | Each row is 1D with shape `(circuit.n,)` |
| Numeric validity | Every value is finite |
| Emission style | Stream rows incrementally with `yield` |

Returning one final `(depth, width)` tensor is invalid.

## Failure semantics

Validation can fail for:

- wrong row shape,
- too few or too many rows,
- non-finite values,
- non-iterable `predict` output.

## ➡️ Next step

- [Write an Estimator](../how-to/write-an-estimator.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
