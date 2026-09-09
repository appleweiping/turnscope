"""Dependency-free sparse vector normalization and deterministic cosine retrieval."""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


def normalize_sparse(
    vector: Mapping[str, float], *, normalization: str = "l2"
) -> Mapping[str, float]:
    """Validate finite coordinates, omit zeros, and apply none/L1/L2 normalization."""
    if normalization not in ("none", "l1", "l2"):
        raise ValueError("normalization must be none, l1, or l2")
    if not isinstance(vector, Mapping):
        raise TypeError("vector must be a mapping")
    values: dict[str, float] = {}
    for key, value in vector.items():
        if not isinstance(key, str) or not key:
            raise ValueError("vector terms must be non-empty strings")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("vector coordinates must be finite numbers")
        if value != 0:
            values[key] = float(value)
    if values and normalization != "none":
        # Scaling first avoids overflow in both the norm and the squared sum.
        scale = max(abs(value) for value in values.values())
        values = {key: value / scale for key, value in values.items()}
        divisor = (
            math.fsum(abs(value) for value in values.values())
            if normalization == "l1"
            else math.sqrt(math.fsum(value * value for value in values.values()))
        )
        values = {key: value / divisor for key, value in values.items()}
    return MappingProxyType(dict(sorted(values.items())))


def sparse_cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Cosine in [-1, 1]; an empty or all-zero vector has similarity zero."""
    first, second = normalize_sparse(left), normalize_sparse(right)
    if len(first) > len(second):
        first, second = second, first
    return max(
        -1.0, min(1.0, math.fsum(value * second.get(key, 0) for key, value in first.items()))
    )


@dataclass(frozen=True, slots=True)
class VectorMatch:
    """A nonzero sparse cosine match to a stable document identifier."""

    id: str
    score: float


class SparseSimilarityIndex:
    """Frozen normalized vectors and inverted postings for top-k cosine matches."""

    def __init__(self, vectors: Mapping[str, Mapping[str, float]]) -> None:
        if not isinstance(vectors, Mapping):
            raise TypeError("vectors must be a mapping")
        self._postings: dict[str, dict[str, float]] = {}
        for identifier, vector in vectors.items():
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("vector IDs must be non-empty strings")
            for term, weight in normalize_sparse(vector).items():
                self._postings.setdefault(term, {})[identifier] = weight

    def query(
        self,
        vector: Mapping[str, float],
        *,
        limit: int = 10,
        minimum_score: float = 0.0,
        exclude_id: str | None = None,
    ) -> tuple[VectorMatch, ...]:
        """Return scores strictly above the cutoff, ordered by score then ID.

        The cutoff must be nonnegative: zero-overlap documents are never
        enumerated. Candidate weights are normalized with the same L2 contract
        as queries, independent of their original scale.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if (
            isinstance(minimum_score, bool)
            or not isinstance(minimum_score, (int, float))
            or not math.isfinite(minimum_score)
            or not 0 <= minimum_score <= 1
        ):
            raise ValueError("minimum_score must be a finite number in [0, 1]")
        products: dict[str, list[float]] = {}
        for term, weight in normalize_sparse(vector).items():
            for identifier, other in self._postings.get(term, {}).items():
                if identifier != exclude_id:
                    products.setdefault(identifier, []).append(weight * other)
        matches = (
            VectorMatch(identifier, max(-1.0, min(1.0, math.fsum(values))))
            for identifier, values in products.items()
        )
        return tuple(
            heapq.nsmallest(
                limit,
                (match for match in matches if match.score > minimum_score),
                key=lambda match: (-match.score, match.id),
            )
        )
