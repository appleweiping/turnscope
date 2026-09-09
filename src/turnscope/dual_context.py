"""Shared-space, two-direction expected-context representations with frozen inference."""

from __future__ import annotations

import hashlib
import math
import struct
import unicodedata
from collections import Counter, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .transformers import _tokens

MODEL_FORMAT = "turnscope.dual-context.shared-svd.v1"
TOKENIZER_VERSION = f"turnscope.regex-casefold.v1/ucd-{unicodedata.unidata_version}"
MAX_RECORD_TEXT_BYTES = 1024 * 1024
ZERO_TOLERANCE = 1e-12


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _text(value: Any, name: str, maximum: int, *, identifier: bool = False) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError(f"{name} must be bounded text")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as error:
        raise ValueError(f"{name} contains invalid Unicode") from error
    if size > maximum or (identifier and (not value.strip() or "\x00" in value)):
        raise ValueError(
            f"{name} must be bounded nonempty text"
            if identifier
            else f"{name} exceeds its byte limit"
        )
    return value


def _unit(values: Iterable[float]) -> tuple[float, ...] | None:
    vector = tuple(values)
    norm = math.hypot(*vector)
    if not math.isfinite(norm):
        raise ValueError("dual-context projection exceeded finite numeric range")
    return tuple(value / norm for value in vector) if norm > ZERO_TOLERANCE else None


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


@dataclass(frozen=True, slots=True)
class ContextRecord:
    conversation_id: str
    utterance_id: str
    text: str

    def __post_init__(self) -> None:
        _text(self.conversation_id, "conversation ID", 256, identifier=True)
        _text(self.utterance_id, "utterance ID", 256, identifier=True)
        _text(self.text, "training text", MAX_RECORD_TEXT_BYTES)

    @property
    def key(self) -> tuple[str, str]:
        return self.conversation_id, self.utterance_id

    def to_dict(self) -> dict[str, str]:
        return {
            "conversation_id": self.conversation_id,
            "utterance_id": self.utterance_id,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class ContextEdge:
    """An explicitly supplied within-conversation source-to-context relationship."""

    conversation_id: str
    source_id: str
    context_id: str

    def __post_init__(self) -> None:
        for name in ("conversation_id", "source_id", "context_id"):
            _text(getattr(self, name), name, 256, identifier=True)
        if self.source_id == self.context_id:
            raise ValueError("context edges must not be self references")

    @property
    def source_key(self) -> tuple[str, str]:
        return self.conversation_id, self.source_id

    @property
    def context_key(self) -> tuple[str, str]:
        return self.conversation_id, self.context_id

    def to_dict(self) -> dict[str, str]:
        return {
            "conversation_id": self.conversation_id,
            "source_id": self.source_id,
            "context_id": self.context_id,
        }


@dataclass(frozen=True, slots=True)
class DualContextConfig:
    n_components: int = 16
    drop_first: bool = False
    min_document_frequency: int = 1
    max_features: int = 256
    n_clusters: int = 8
    max_kmeans_iterations: int = 100
    seed: int = 0
    max_catalog_records: int = 30_000
    max_edges_per_direction: int = 60_000
    max_training_text_bytes: int = 32 * 1024 * 1024
    max_training_tokens: int = 4_000_000
    max_vocabulary_candidates: int = 100_000
    max_dense_cells: int = 64_000_000

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("n_components", 1, 128),
            ("min_document_frequency", 1, 100_000),
            ("max_features", 1, 1024),
            ("n_clusters", 1, 64),
            ("max_kmeans_iterations", 1, 1000),
            ("seed", 0, 2**32 - 1),
            ("max_catalog_records", 1, 100_000),
            ("max_edges_per_direction", 1, 200_000),
            ("max_training_text_bytes", 1, 128 * 1024 * 1024),
            ("max_training_tokens", 1, 10_000_000),
            ("max_vocabulary_candidates", 1, 500_000),
            ("max_dense_cells", 1, 128_000_000),
        ):
            _integer(getattr(self, name), name, minimum, maximum)
        if type(self.drop_first) is not bool:
            raise ValueError("drop_first must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ClusterState:
    centers: tuple[tuple[float, ...], ...]
    counts: tuple[int, ...]
    iterations: int
    converged: bool
    objective: float
    training_vectors: int
    distinct_vectors: int
    reason: str

    @property
    def effective_clusters(self) -> int:
        return len(self.centers)


@dataclass(frozen=True, slots=True)
class DirectionState:
    term_map: tuple[tuple[float, ...], ...]
    term_ranges: tuple[float | None, ...]
    support_edges: tuple[int, ...]
    projectable_edges: tuple[int, ...]
    clustering: ClusterState


@dataclass(frozen=True, slots=True)
class DualContextState:
    vocabulary: tuple[str, ...]
    document_frequencies: tuple[int, ...]
    column_norms: tuple[float, ...]
    basis: tuple[tuple[float, ...], ...]
    singular_values: tuple[float, ...]
    forward: DirectionState
    backward: DirectionState
    catalog_records: int
    context_records: int
    forward_edges: int
    backward_edges: int
    training_text_bytes: int
    training_tokens: int
    vocabulary_candidates: int
    numerical_rank: int
    svd_cutoff: float
    estimated_dense_cells: int
    catalog_digest: str
    forward_digest: str
    backward_digest: str

    @property
    def dimensions(self) -> int:
        return len(self.singular_values)

    @property
    def idf(self) -> tuple[float, ...]:
        return tuple(
            math.log((1 + self.catalog_records) / (1 + count)) + 1.0
            for count in self.document_frequencies
        )


@dataclass(frozen=True, slots=True)
class DirectionPrediction:
    vector: tuple[float, ...] | None
    range: float | None
    range_weight_coverage: float
    support_weight_coverage: float
    reason: str
    cluster_id: int | None
    cluster_distance: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector": list(self.vector) if self.vector is not None else None,
            "range": self.range,
            "range_weight_coverage": self.range_weight_coverage,
            "support_weight_coverage": self.support_weight_coverage,
            "reason": self.reason,
            "cluster_id": self.cluster_id,
            "cluster_distance": self.cluster_distance,
        }


@dataclass(frozen=True, slots=True)
class ContextProjection:
    vector: tuple[float, ...] | None
    tokens: int
    known_tokens: int
    reason: str
    forward_cluster_id: int | None
    forward_cluster_distance: float | None
    backward_cluster_id: int | None
    backward_cluster_distance: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
            "vector": list(self.vector) if self.vector is not None else None,
        }


@dataclass(frozen=True, slots=True)
class DualContextPrediction:
    forward: DirectionPrediction
    backward: DirectionPrediction
    orientation: float | None
    shift: float | None
    tokens: int
    known_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "forward": self.forward.to_dict(),
            "backward": self.backward.to_dict(),
            "orientation": self.orientation,
            "shift": self.shift,
            "tokens": self.tokens,
            "known_tokens": self.known_tokens,
        }


def _fingerprint(rows: Iterable[tuple[str, ...]], domain: str) -> str:
    digest = hashlib.sha256(domain.encode("ascii"))
    for row in rows:
        for item in row:
            data = item.encode("utf-8")
            digest.update(struct.pack(">Q", len(data)))
            digest.update(data)
    return digest.hexdigest()


def _edges(
    values: Iterable[ContextEdge], lookup: Mapping[tuple[str, str], int], maximum: int
) -> tuple[ContextEdge, ...]:
    result: list[ContextEdge] = []
    seen: set[ContextEdge] = set()
    adjacency: dict[tuple[str, str], list[tuple[str, str]]] = {}
    indegree: Counter[tuple[str, str]] = Counter()
    for edge in values:
        if len(result) == maximum:
            raise ValueError("context edges exceed max_edges_per_direction")
        if not isinstance(edge, ContextEdge):
            raise ValueError("context edges must be ContextEdge values")
        if edge.source_key not in lookup or edge.context_key not in lookup:
            raise ValueError("context edge refers to a missing catalog endpoint")
        if edge in seen:
            raise ValueError("context edges must not contain duplicates")
        seen.add(edge)
        result.append(edge)
        adjacency.setdefault(edge.source_key, []).append(edge.context_key)
        indegree[edge.context_key] += 1
        indegree.setdefault(edge.source_key, 0)
    if not result:
        raise ValueError("each context direction requires at least one edge")
    queue = deque(key for key, degree in indegree.items() if not degree)
    processed = 0
    while queue:
        key = queue.popleft()
        processed += 1
        for context in adjacency.get(key, ()):
            indegree[context] -= 1
            if not indegree[context]:
                queue.append(context)
    if processed != len(indegree):
        raise ValueError("each context direction must be acyclic")
    return tuple(
        sorted(result, key=lambda item: (item.conversation_id, item.source_id, item.context_id))
    )


def dense_cells(catalog: int, contexts: int, features: int, dimensions: int, clusters: int) -> int:
    """Conservative cell accounting for input, aligned maps, reduced SVD and Lloyd scratch."""
    reduced = min(contexts, features)
    return (
        4 * catalog * features
        + 4 * contexts * features
        + 2 * contexts * reduced
        + 2 * features * reduced
        + 8 * features * dimensions
        + 4 * catalog * dimensions
        + 4 * catalog * clusters
    )


def _assignment(
    vector: tuple[float, ...] | None, clusters: ClusterState
) -> tuple[int | None, float | None]:
    if vector is None or not clusters.centers:
        return None, None
    distances = tuple(_distance(vector, center) for center in clusters.centers)
    identifier = min(range(len(distances)), key=lambda index: (distances[index], index))
    return identifier, distances[identifier]


class DualContextModel:
    """Fit once into one shared SVD space; predict with only the standard library."""

    def __init__(
        self,
        *,
        n_components: int = 16,
        drop_first: bool = False,
        min_document_frequency: int = 1,
        max_features: int = 256,
        n_clusters: int = 8,
        max_kmeans_iterations: int = 100,
        seed: int = 0,
        max_catalog_records: int = 30_000,
        max_edges_per_direction: int = 60_000,
        max_training_text_bytes: int = 32 * 1024 * 1024,
        max_training_tokens: int = 4_000_000,
        max_vocabulary_candidates: int = 100_000,
        max_dense_cells: int = 64_000_000,
    ) -> None:
        self._config = DualContextConfig(
            n_components,
            drop_first,
            min_document_frequency,
            max_features,
            n_clusters,
            max_kmeans_iterations,
            seed,
            max_catalog_records,
            max_edges_per_direction,
            max_training_text_bytes,
            max_training_tokens,
            max_vocabulary_candidates,
            max_dense_cells,
        )
        self._state: DualContextState | None = None

    @property
    def config(self) -> DualContextConfig:
        return self._config

    @property
    def state(self) -> DualContextState:
        if self._state is None:
            raise ValueError("dual-context model has not been fitted")
        return self._state

    def fit(
        self,
        catalog: Iterable[ContextRecord],
        forward_edges: Iterable[ContextEdge],
        backward_edges: Iterable[ContextEdge],
    ) -> DualContextModel:
        config = self.config
        records: dict[tuple[str, str], ContextRecord] = {}
        text_bytes = 0
        for record in catalog:
            if len(records) == config.max_catalog_records:
                raise ValueError("catalog exceeds max_catalog_records")
            if not isinstance(record, ContextRecord):
                raise ValueError("catalog must contain ContextRecord values")
            if record.key in records:
                raise ValueError("catalog composite IDs must be unique")
            text_bytes += len(record.text.encode("utf-8"))
            if text_bytes > config.max_training_text_bytes:
                raise ValueError("catalog exceeds max_training_text_bytes")
            records[record.key] = record
        if not records:
            raise ValueError("training catalog must not be empty")
        ordered = tuple(records[key] for key in sorted(records))
        lookup = {record.key: index for index, record in enumerate(ordered)}
        forward = _edges(forward_edges, lookup, config.max_edges_per_direction)
        backward = _edges(backward_edges, lookup, config.max_edges_per_direction)
        context_keys = sorted({edge.context_key for edge in (*forward, *backward)})
        context_index = {key: index for index, key in enumerate(context_keys)}
        frequencies: Counter[str] = Counter()
        counts = []
        total_tokens = 0
        for record in ordered:
            terms = _tokens(record.text)
            total_tokens += len(terms)
            if total_tokens > config.max_training_tokens:
                raise ValueError("catalog exceeds max_training_tokens")
            row = Counter(terms)
            for term in row:
                if term not in frequencies and len(frequencies) == config.max_vocabulary_candidates:
                    raise ValueError("catalog exceeds max_vocabulary_candidates")
                frequencies[term] += 1
            counts.append(row)
        selected = sorted(
            (term for term in frequencies if frequencies[term] >= config.min_document_frequency),
            key=lambda term: (-frequencies[term], term),
        )[: config.max_features]
        vocabulary = tuple(sorted(selected))
        if not vocabulary:
            raise ValueError("training vocabulary is empty after filtering")
        dimensions = min(
            config.n_components + int(config.drop_first), len(context_keys), len(vocabulary)
        )
        estimate = dense_cells(
            len(ordered), len(context_keys), len(vocabulary), dimensions, config.n_clusters
        )
        if estimate > config.max_dense_cells:
            raise ValueError(f"estimated dense workspace {estimate} exceeds max_dense_cells")
        indices = {term: index for index, term in enumerate(vocabulary)}
        df = tuple(frequencies[term] for term in vocabulary)
        idf = tuple(math.log((1 + len(ordered)) / (1 + count)) + 1 for count in df)
        sparse_rows = []
        for row in counts:
            weighted = {
                indices[term]: count * idf[indices[term]]
                for term, count in row.items()
                if term in indices
            }
            norm = math.hypot(*weighted.values())
            sparse_rows.append(
                {index: value / norm for index, value in weighted.items()} if norm else {}
            )
        from ._dual_context_numeric import fit_numeric_dual

        numeric = fit_numeric_dual(
            sparse_rows,
            len(vocabulary),
            [lookup[key] for key in context_keys],
            [(lookup[edge.source_key], context_index[edge.context_key]) for edge in forward],
            [(lookup[edge.source_key], context_index[edge.context_key]) for edge in backward],
            config,
        )
        candidate = DualContextState(
            vocabulary,
            df,
            numeric.column_norms,
            numeric.basis,
            numeric.singular_values,
            numeric.forward,
            numeric.backward,
            len(ordered),
            len(context_keys),
            len(forward),
            len(backward),
            text_bytes,
            total_tokens,
            len(frequencies),
            numeric.numerical_rank,
            numeric.svd_cutoff,
            estimate,
            _fingerprint(
                ((record.conversation_id, record.utterance_id, record.text) for record in ordered),
                "catalog/v1",
            ),
            _fingerprint(
                ((edge.conversation_id, edge.source_id, edge.context_id) for edge in forward),
                "forward/v1",
            ),
            _fingerprint(
                ((edge.conversation_id, edge.source_id, edge.context_id) for edge in backward),
                "backward/v1",
            ),
        )
        from ._dual_context_artifact import encode_model, validate_state

        validate_state(candidate, config)
        encode_model(candidate, config)  # Accepted fitted states must fit their exported envelope.
        self._state = candidate
        return self

    def _weights(self, text: str) -> tuple[tuple[float, ...], int, int]:
        _text(text, "query text", MAX_RECORD_TEXT_BYTES)
        state = self.state
        tokens = _tokens(text)
        counts = Counter(tokens)
        values = tuple(
            counts[term] * idf for term, idf in zip(state.vocabulary, state.idf, strict=True)
        )
        norm = math.hypot(*values)
        weights = (
            tuple(
                value / norm / column
                for value, column in zip(values, state.column_norms, strict=True)
            )
            if norm
            else (0.0,) * len(values)
        )
        return weights, len(tokens), sum(counts[term] for term in state.vocabulary)

    def _direction(
        self, weights: tuple[float, ...], direction: DirectionState, tokens: int, known: int
    ) -> DirectionPrediction:
        state = self.state
        vector = _unit(
            math.fsum(
                weight * row[dimension]
                for weight, row in zip(weights, direction.term_map, strict=True)
            )
            / scale
            for dimension, scale in enumerate(state.singular_values)
        )
        total = math.fsum(weights)
        support = math.fsum(
            weight for weight, count in zip(weights, direction.support_edges, strict=True) if count
        )
        eligible = math.fsum(
            weight
            for weight, value in zip(weights, direction.term_ranges, strict=True)
            if value is not None
        )
        range_value = (
            math.fsum(
                weight * value
                for weight, value in zip(weights, direction.term_ranges, strict=True)
                if value is not None
            )
            / eligible
            if eligible
            else None
        )
        reason = (
            "ok"
            if vector is not None
            else (
                "empty_text"
                if not tokens
                else "out_of_vocabulary"
                if not known
                else "no_direction_support"
                if not support
                else "zero_projection"
            )
        )
        cluster_id, distance = _assignment(vector, direction.clustering)
        return DirectionPrediction(
            vector,
            range_value,
            eligible / total if total else 0.0,
            support / total if total else 0.0,
            reason,
            cluster_id,
            distance,
        )

    def predict(self, text: str) -> DualContextPrediction:
        weights, tokens, known = self._weights(text)
        forward = self._direction(weights, self.state.forward, tokens, known)
        backward = self._direction(weights, self.state.backward, tokens, known)
        orientation = (
            forward.range - backward.range
            if forward.range is not None and backward.range is not None
            else None
        )
        shift = (
            _distance(forward.vector, backward.vector)
            if forward.vector is not None and backward.vector is not None
            else None
        )
        return DualContextPrediction(forward, backward, orientation, shift, tokens, known)

    def transform(self, text: str) -> DualContextPrediction:
        return self.predict(text)

    def project_context(self, text: str) -> ContextProjection:
        weights, tokens, known = self._weights(text)
        state = self.state
        vector = _unit(
            math.fsum(
                weight * row[dimension] for weight, row in zip(weights, state.basis, strict=True)
            )
            / scale
            for dimension, scale in enumerate(state.singular_values)
        )
        forward_id, forward_distance = _assignment(vector, state.forward.clustering)
        backward_id, backward_distance = _assignment(vector, state.backward.clustering)
        return ContextProjection(
            vector,
            tokens,
            known,
            "ok"
            if vector is not None
            else "empty_text"
            if not tokens
            else "out_of_vocabulary"
            if not known
            else "zero_projection",
            forward_id,
            forward_distance,
            backward_id,
            backward_distance,
        )

    def term_statistics(self) -> tuple[dict[str, Any], ...]:
        state = self.state
        rows = []
        for index, term in enumerate(state.vocabulary):
            directions = {}
            vectors = []
            ranges = []
            for name, direction in (("forward", state.forward), ("backward", state.backward)):
                vector = _unit(direction.term_map[index])
                vectors.append(vector)
                ranges.append(direction.term_ranges[index])
                cluster_id, distance = _assignment(vector, direction.clustering)
                directions[name] = {
                    "vector": list(vector) if vector is not None else None,
                    "range": direction.term_ranges[index],
                    "support_edges": direction.support_edges[index],
                    "projectable_edges": direction.projectable_edges[index],
                    "cluster_id": cluster_id,
                    "cluster_distance": distance,
                    "reason": "ok"
                    if vector is not None
                    else "no_direction_support"
                    if not direction.support_edges[index]
                    else "zero_projection",
                }
            rows.append(
                {
                    "term": term,
                    **directions,
                    "orientation": ranges[0] - ranges[1]
                    if ranges[0] is not None and ranges[1] is not None
                    else None,
                    "shift": _distance(vectors[0], vectors[1])
                    if vectors[0] is not None and vectors[1] is not None
                    else None,
                }
            )
        return tuple(rows)

    def training_summary(self) -> dict[str, Any]:
        state = self.state
        return {
            "catalog_records": state.catalog_records,
            "context_records": state.context_records,
            "forward_edges": state.forward_edges,
            "backward_edges": state.backward_edges,
            "training_text_bytes": state.training_text_bytes,
            "training_tokens": state.training_tokens,
            "vocabulary_candidates": state.vocabulary_candidates,
            "features": len(state.vocabulary),
            "numerical_rank": state.numerical_rank,
            "dimensions": state.dimensions,
            "drop_first": self.config.drop_first,
            "svd_fits": 1,
            "estimated_dense_cells": state.estimated_dense_cells,
            "clustering": {
                name: {
                    "requested_clusters": self.config.n_clusters,
                    **{
                        field: getattr(direction.clustering, field)
                        for field in (
                            "effective_clusters",
                            "iterations",
                            "converged",
                            "objective",
                            "training_vectors",
                            "distinct_vectors",
                            "reason",
                        )
                    },
                }
                for name, direction in (("forward", state.forward), ("backward", state.backward))
            },
        }

    def to_dict(self) -> dict[str, Any]:
        from ._dual_context_artifact import encode_model

        return encode_model(self.state, self.config)

    @classmethod
    def from_dict(cls, value: Any) -> DualContextModel:
        from ._dual_context_artifact import decode_model

        return decode_model(value)

    @property
    def digest(self) -> str:
        return str(self.to_dict()["sha256"])

    def save(self, path: str | Path, *, overwrite: bool = False) -> str | None:
        from ._dual_context_artifact import save_model

        return save_model(self.to_dict(), Path(path), overwrite=overwrite)

    @classmethod
    def load(cls, path: str | Path) -> DualContextModel:
        from ._dual_context_artifact import load_model

        return cls.from_dict(load_model(Path(path)))
