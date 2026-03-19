"""Serializable request/response schemas for future RPC-compatible APIs."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class ScoreRequest:
    """Serializable request payload for a scoring RPC-style interface."""

    schema_version: str
    n_circuits: int
    n_samples: int
    budget: int

    def to_dict(self) -> dict[str, str | int]:
        """Serialize this request dataclass to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, str | int]) -> "ScoreRequest":
        """Parse a request object from a loosely typed dictionary payload."""
        return cls(
            schema_version=str(payload["schema_version"]),
            n_circuits=int(payload["n_circuits"]),
            n_samples=int(payload["n_samples"]),
            budget=int(payload["budget"]),
        )


@dataclass(slots=True)
class ScoreResponse:
    """Serializable response payload for score + message results."""

    schema_version: str
    score: float
    message: str

    def to_dict(self) -> dict[str, str | float]:
        """Serialize this response dataclass to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, str | float]) -> "ScoreResponse":
        """Parse a response object from a loosely typed dictionary payload."""
        return cls(
            schema_version=str(payload["schema_version"]),
            score=float(payload["score"]),
            message=str(payload["message"]),
        )
