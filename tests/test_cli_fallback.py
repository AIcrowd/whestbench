from __future__ import annotations

import circuit_estimation.cli as cli


def test_error_code_mapping_for_stream_contract_messages() -> None:
    not_iterable = "Estimator must return an iterator of depth-row outputs."
    assert (
        cli._error_code(ValueError(not_iterable), not_iterable) == "ESTIMATOR_STREAM_NOT_ITERABLE"
    )
    too_many = "Estimator emitted more than max_depth rows."
    assert cli._error_code(ValueError(too_many), too_many) == "ESTIMATOR_STREAM_TOO_MANY_ROWS"
    too_few = "Estimator must emit exactly max_depth rows."
    assert cli._error_code(ValueError(too_few), too_few) == "ESTIMATOR_STREAM_TOO_FEW_ROWS"
    bad_shape = "Estimator row at depth 0 must have shape (4,), got (1,)."
    assert cli._error_code(ValueError(bad_shape), bad_shape) == "ESTIMATOR_STREAM_BAD_ROW_SHAPE"
    non_finite = "Estimator row at depth 1 must contain finite values."
    assert cli._error_code(ValueError(non_finite), non_finite) == "ESTIMATOR_STREAM_NON_FINITE_ROW"


def test_error_payload_shape_is_stable() -> None:
    payload = cli._error_payload(ValueError("bad row"), include_traceback=False)
    assert payload["ok"] is False
    assert payload["error"]["stage"] == "scoring"
    assert payload["error"]["code"] == "SCORING_VALIDATION_ERROR"
    assert payload["error"]["message"] == "bad row"
