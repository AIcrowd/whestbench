"""Serializable request/response schemas for future RPC-compatible APIs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class ScoreRequest:
    """Serializable request payload for a scoring RPC-style interface."""

    schema_version: str
    n_mlps: int
    n_samples: int
    estimator_budget: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this request dataclass to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScoreRequest":
        """Parse a request object from a loosely typed dictionary payload."""
        return cls(
            schema_version=str(payload["schema_version"]),
            n_mlps=int(payload["n_mlps"]),
            n_samples=int(payload["n_samples"]),
            estimator_budget=int(payload["estimator_budget"]),
        )


@dataclass
class ScoreResponse:
    """Serializable response payload for score + message results."""

    schema_version: str
    score: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this response dataclass to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScoreResponse":
        """Parse a response object from a loosely typed dictionary payload."""
        return cls(
            schema_version=str(payload["schema_version"]),
            score=float(payload["score"]),
            message=str(payload["message"]),
        )
