from network_estimation.protocol import ScoreRequest, ScoreResponse


def test_score_request_has_versioned_schema() -> None:
    # Schema version enables forward-compatible RPC evolution.
    req = ScoreRequest(
        schema_version="1.0",
        n_mlps=2,
        n_samples=16,
        flop_budget=100,
        width=256,
        depth=16,
    )
    assert req.schema_version == "1.0"
    assert req.width == 256
    assert req.depth == 16


def test_score_response_round_trip_dict() -> None:
    # Serialization round-trip should preserve payload exactly.
    response = ScoreResponse(
        schema_version="1.0",
        score=0.123,
        message="ok",
    )
    payload = response.to_dict()
    round_trip = ScoreResponse.from_dict(payload)
    assert round_trip == response
