from circuit_estimation.protocol import ScoreRequest, ScoreResponse


def test_score_request_has_versioned_schema() -> None:
    req = ScoreRequest(schema_version="1.0", n_circuits=2, n_samples=16, budget=100)
    assert req.schema_version == "1.0"


def test_score_response_round_trip_dict() -> None:
    response = ScoreResponse(
        schema_version="1.0",
        score=0.123,
        message="ok",
    )
    payload = response.to_dict()
    round_trip = ScoreResponse.from_dict(payload)
    assert round_trip == response
