# Estimator Contract

## Signature

Required method:

- `predict(self, circuit: Circuit, budget: int) -> Iterator[NDArray[np.float32]]`

Optional lifecycle hooks:

- `setup(self, context: SetupContext) -> None`
- `teardown(self) -> None`

## Output Requirements

For each `predict` call:

- emit exactly `circuit.d` rows,
- each row must be a 1D vector of shape `(circuit.n,)`,
- all values must be finite,
- rows are emitted incrementally with `yield`.

Returning one final `(depth, width)` tensor is invalid.

## Failure Semantics

Validation can fail for:

- wrong row shape,
- too few or too many rows,
- non-finite values,
- non-iterable `predict` output.

## Related References

- [Write an Estimator](../how-to/write-an-estimator.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
