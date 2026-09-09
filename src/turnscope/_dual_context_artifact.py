"""Closed, bounded JSON artifacts and atomic publication for dual context models."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import unicodedata
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .dual_context import (
    MAX_RECORD_TEXT_BYTES,
    MODEL_FORMAT,
    TOKENIZER_VERSION,
    ZERO_TOLERANCE,
    ClusterState,
    DirectionState,
    DualContextConfig,
    DualContextModel,
    DualContextState,
    _integer,
    _text,
    dense_cells,
)
from .io import parse_json_value
from .transformers import _tokens

MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_JSON_NODES = 2_000_000
MAX_JSON_DEPTH = 32
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _bounded_graph(value: Any) -> None:
    stack = [(value, 0)]
    nodes = 0
    text_bytes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            raise ValueError("dual-context artifact exceeds JSON structural limits")
        if isinstance(item, str):
            _text(item, "artifact text", MAX_ARTIFACT_BYTES)
            text_bytes += len(item.encode("utf-8"))
            if text_bytes > MAX_ARTIFACT_BYTES:
                raise ValueError("dual-context artifact exceeds its byte limit")
        elif isinstance(item, dict):
            if 2 * len(item) > MAX_JSON_NODES - nodes - len(stack):
                raise ValueError("dual-context artifact has too many fields")
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("dual-context artifact object keys must be text")
                stack.append((key, depth + 1))
                stack.append((child, depth + 1))
        elif isinstance(item, (list, tuple)):
            if len(item) > MAX_JSON_NODES - nodes - len(stack):
                raise ValueError("dual-context artifact array exceeds structural limits")
            stack.extend((child, depth + 1) for child in item)
        elif item is not None and type(item) not in (bool, int, float):
            raise ValueError("dual-context artifact must contain JSON values")
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("dual-context artifact must contain finite numbers")
        elif type(item) is int and not -(2**63) <= item <= 2**63 - 1:
            raise ValueError("dual-context artifact integer exceeds its numeric limit")


def _canonical(value: Any) -> bytes:
    _bounded_graph(value)
    output = bytearray()
    encoder = json.JSONEncoder(
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    for chunk in encoder.iterencode(value):
        encoded = chunk.encode("utf-8")
        if len(output) + len(encoded) > MAX_ARTIFACT_BYTES:
            raise ValueError("dual-context artifact exceeds its byte limit")
        output.extend(encoded)
    return bytes(output)


def _closed(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} must have exactly the supported fields")
    return value


def _sequence(value: Any, size: int, name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} has an invalid shape")
    return tuple(value)


def _real(value: Any, name: str, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be a finite number in [{minimum}, {maximum}]")
    return float(value)


def _matrix(value: Any, rows: int, columns: int, name: str) -> tuple[tuple[Any, ...], ...]:
    return tuple(_sequence(row, columns, name) for row in _sequence(value, rows, name))


def _cluster(value: Any, dimensions: int) -> ClusterState:
    data = _closed(value, set(ClusterState.__dataclass_fields__), "clustering")
    centers = data["centers"]
    if not isinstance(centers, (tuple, list)) or len(centers) > 64:
        raise ValueError("clustering centers have invalid shape")
    return ClusterState(
        _matrix(centers, len(centers), dimensions, "centers"),
        _sequence(data["counts"], len(centers), "cluster counts"),
        **{key: item for key, item in data.items() if key not in ("centers", "counts")},
    )


def _direction(value: Any, features: int, dimensions: int) -> DirectionState:
    data = _closed(value, set(DirectionState.__dataclass_fields__), "direction")
    return DirectionState(
        _matrix(data["term_map"], features, dimensions, "term map"),
        _sequence(data["term_ranges"], features, "term ranges"),
        _sequence(data["support_edges"], features, "term supports"),
        _sequence(data["projectable_edges"], features, "projectable supports"),
        _cluster(data["clustering"], dimensions),
    )


def _validate_cluster(
    cluster: ClusterState, config: DualContextConfig, state: DualContextState
) -> None:
    if (
        type(cluster) is not ClusterState
        or type(cluster.centers) is not tuple
        or type(cluster.counts) is not tuple
    ):
        raise ValueError("clustering state must be immutable")
    effective = len(cluster.centers)
    _integer(effective, "effective clusters", 0, config.n_clusters)
    _integer(cluster.training_vectors, "cluster training vectors", 0, state.catalog_records)
    _integer(cluster.distinct_vectors, "distinct vectors", effective, cluster.training_vectors)
    _integer(cluster.iterations, "cluster iterations", 0, config.max_kmeans_iterations)
    _real(cluster.objective, "cluster objective", 0.0, 4.0 * cluster.training_vectors + 1e-8)
    if type(cluster.converged) is not bool or len(cluster.counts) != effective:
        raise ValueError("invalid cluster convergence or counts")
    for row in cluster.centers:
        if type(row) is not tuple or len(row) != state.dimensions:
            raise ValueError("cluster center shape must match retained dimensions")
        values = tuple(_real(item, "cluster center", -1.0000001, 1.0000001) for item in row)
        if math.hypot(*values) > 1.0000001:
            raise ValueError("cluster centers must be means of unit vectors")
    for count in cluster.counts:
        _integer(count, "cluster count", 1, cluster.training_vectors)
    if sum(cluster.counts) != cluster.training_vectors:
        raise ValueError("cluster counts must sum to the training vector count")
    if not effective:
        if (
            cluster.training_vectors
            or cluster.distinct_vectors
            or cluster.iterations
            or cluster.objective != 0
            or not cluster.converged
            or cluster.reason != "no_nonzero_vectors"
        ):
            raise ValueError("empty clustering has inconsistent summary")
    else:
        if not cluster.iterations or cluster.reason not in (
            "ok",
            "distinct_vectors_below_requested",
            "empty_clusters_removed",
        ):
            raise ValueError("nonempty clustering has inconsistent summary")
        if cluster.reason == "ok" and effective != config.n_clusters:
            raise ValueError("cluster degeneracy must be reported")
        if cluster.reason == "distinct_vectors_below_requested" and not (
            effective == cluster.distinct_vectors < config.n_clusters
        ):
            raise ValueError("distinct-vector degeneracy is inconsistent")
        if cluster.reason == "empty_clusters_removed" and effective >= min(
            config.n_clusters, cluster.distinct_vectors
        ):
            raise ValueError("empty-cluster degeneracy is inconsistent")


def _validate_direction(
    direction: DirectionState, edges: int, state: DualContextState, config: DualContextConfig
) -> None:
    if type(direction) is not DirectionState:
        raise ValueError("direction must be a DirectionState")
    features = len(state.vocabulary)
    for field in ("term_map", "term_ranges", "support_edges", "projectable_edges"):
        values = getattr(direction, field)
        if type(values) is not tuple or len(values) != features:
            raise ValueError("direction arrays must be immutable and match the vocabulary")
    for row, range_value, support, projectable in zip(
        direction.term_map,
        direction.term_ranges,
        direction.support_edges,
        direction.projectable_edges,
        strict=True,
    ):
        if type(row) is not tuple or len(row) != state.dimensions:
            raise ValueError("term map shape must match retained dimensions")
        for item, sigma in zip(row, state.singular_values, strict=True):
            _real(item, "term map coefficient", -edges / sigma - 1e-8, edges / sigma + 1e-8)
        _integer(support, "term support", 0, edges)
        _integer(projectable, "projectable term support", 0, support)
        norm = math.hypot(*row)
        if not support and any(item != 0 for item in row):
            raise ValueError("unsupported terms must have zero maps")
        expected_range = norm > ZERO_TOLERANCE and projectable > 0
        if expected_range != (range_value is not None):
            raise ValueError("term range availability disagrees with support and projection")
        if range_value is not None:
            _real(range_value, "term range", 0.0, 1.0)
    _validate_cluster(direction.clustering, config, state)


def validate_state(state: DualContextState, config: DualContextConfig) -> None:
    if type(state) is not DualContextState or type(config) is not DualContextConfig:
        raise ValueError("invalid fitted state or configuration")
    # Revalidate even deliberately forged frozen dataclasses before using their bounds.
    DualContextConfig(**config.to_dict())
    limits = {
        "catalog_records": (1, config.max_catalog_records),
        "context_records": (1, config.max_catalog_records),
        "forward_edges": (1, config.max_edges_per_direction),
        "backward_edges": (1, config.max_edges_per_direction),
        "training_text_bytes": (1, config.max_training_text_bytes),
        "training_tokens": (1, config.max_training_tokens),
        "vocabulary_candidates": (1, config.max_vocabulary_candidates),
    }
    for name, (lower, upper) in limits.items():
        _integer(getattr(state, name), name, lower, upper)
    if state.context_records > state.catalog_records:
        raise ValueError("context inventory exceeds catalog inventory")
    if state.context_records > state.forward_edges + state.backward_edges:
        raise ValueError("context inventory exceeds its edge support")
    if state.training_tokens > state.training_text_bytes:
        raise ValueError("training token count exceeds UTF-8 text bytes")
    if type(state.vocabulary) is not tuple:
        raise ValueError("vocabulary must be immutable")
    features = _integer(
        len(state.vocabulary), "features", 1, min(config.max_features, state.vocabulary_candidates)
    )
    if state.vocabulary_candidates > state.training_tokens:
        raise ValueError("candidate vocabulary exceeds training tokens")
    for term in state.vocabulary:
        _text(term, "vocabulary term", MAX_RECORD_TEXT_BYTES, identifier=True)
        # Casefold can introduce combining marks (Turkish I, Greek vowels).
        # It runs after regex matching, so re-tokenization is not a fixed point.
        check_term = "".join(
            char for char in term if not unicodedata.category(char).startswith("M")
        )
        if term.casefold() != term or _tokens(check_term) != (check_term,):
            raise ValueError("vocabulary terms must match the frozen tokenizer")
    if tuple(sorted(set(state.vocabulary))) != state.vocabulary:
        raise ValueError("vocabulary must be unique and lexically ordered")
    for field in ("document_frequencies", "column_norms", "basis", "singular_values"):
        if type(getattr(state, field)) is not tuple:
            raise ValueError("numeric state arrays must be immutable")
    if len(state.document_frequencies) != features or len(state.column_norms) != features:
        raise ValueError("feature statistics have invalid shape")
    for count in state.document_frequencies:
        _integer(count, "document frequency", config.min_document_frequency, state.catalog_records)
    if sum(state.document_frequencies) > state.training_tokens:
        raise ValueError("document frequencies exceed token inventory")
    minimum_column = 1 / (state.training_tokens * (math.log(state.catalog_records + 1) + 1))
    for value in state.column_norms:
        _real(value, "column norm", minimum_column, math.sqrt(state.catalog_records) + 1e-8)
    rank = _integer(state.numerical_rank, "numerical rank", 1, min(features, state.context_records))
    dimensions = min(config.n_components, rank - int(config.drop_first))
    if dimensions < 1 or len(state.singular_values) != dimensions:
        raise ValueError("retained dimensions disagree with rank and drop_first")
    _real(
        state.svd_cutoff,
        "SVD cutoff",
        ZERO_TOLERANCE,
        max(
            ZERO_TOLERANCE,
            max(features, state.context_records) * 2.220446049250313e-16 * math.sqrt(features),
        )
        + 1e-15,
    )
    for sigma in state.singular_values:
        _real(sigma, "singular value", ZERO_TOLERANCE, math.sqrt(features) + 1e-8)
        if sigma <= state.svd_cutoff:
            raise ValueError("retained singular values must exceed the numerical cutoff")
    if tuple(sorted(state.singular_values, reverse=True)) != state.singular_values:
        raise ValueError("singular values must be in descending order")
    if len(state.basis) != features:
        raise ValueError("shared basis has invalid feature count")
    for row in state.basis:
        if type(row) is not tuple or len(row) != dimensions:
            raise ValueError("shared basis has invalid dimensions")
        for value in row:
            _real(value, "basis coefficient", -1.0000001, 1.0000001)
    for first in range(dimensions):
        for second in range(first + 1):
            dot = math.fsum(row[first] * row[second] for row in state.basis)
            if not math.isclose(dot, float(first == second), rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError("shared basis columns must be orthonormal")
    estimate = dense_cells(
        state.catalog_records,
        state.context_records,
        features,
        min(config.n_components + int(config.drop_first), state.context_records, features),
        config.n_clusters,
    )
    _integer(state.estimated_dense_cells, "estimated dense cells", 1, config.max_dense_cells)
    if estimate != state.estimated_dense_cells:
        raise ValueError("dense workspace accounting is inconsistent")
    for name in ("catalog_digest", "forward_digest", "backward_digest"):
        if not isinstance(getattr(state, name), str) or not _HASH.fullmatch(getattr(state, name)):
            raise ValueError("training fingerprints must be lowercase SHA-256 values")
    _validate_direction(state.forward, state.forward_edges, state, config)
    _validate_direction(state.backward, state.backward_edges, state, config)


def encode_model(state: DualContextState, config: DualContextConfig) -> dict[str, Any]:
    validate_state(state, config)
    body = {
        "format": MODEL_FORMAT,
        "tokenizer": TOKENIZER_VERSION,
        "config": config.to_dict(),
        "state": asdict(state),
    }
    value = {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    # Normalize tuples to JSON arrays once; check the full envelope, not only body.
    return dict(json.loads(_canonical(value)))


def decode_model(value: Any) -> DualContextModel:
    _canonical(value)
    data = _closed(value, {"format", "tokenizer", "config", "state", "sha256"}, "model")
    if data["format"] != MODEL_FORMAT:
        raise ValueError("unsupported dual-context artifact format")
    if data["tokenizer"] != TOKENIZER_VERSION:
        raise ValueError("unsupported dual-context tokenizer or Unicode database version")
    checksum = data["sha256"]
    if not isinstance(checksum, str) or not _HASH.fullmatch(checksum):
        raise ValueError("invalid dual-context artifact checksum")
    body = {key: item for key, item in data.items() if key != "sha256"}
    if hashlib.sha256(_canonical(body)).hexdigest() != checksum:
        raise ValueError("dual-context artifact checksum mismatch")
    config = DualContextConfig(
        **_closed(data["config"], set(DualContextConfig.__dataclass_fields__), "config")
    )
    raw = _closed(data["state"], set(DualContextState.__dataclass_fields__), "fitted state")
    vocabulary = raw["vocabulary"]
    sigma = raw["singular_values"]
    if not isinstance(vocabulary, (list, tuple)) or not 1 <= len(vocabulary) <= config.max_features:
        raise ValueError("vocabulary has invalid shape")
    if not isinstance(sigma, (list, tuple)) or not 1 <= len(sigma) <= config.n_components:
        raise ValueError("singular values have invalid shape")
    features, dimensions = len(vocabulary), len(sigma)
    replacements = {
        "vocabulary": tuple(vocabulary),
        "singular_values": tuple(sigma),
        "document_frequencies": _sequence(
            raw["document_frequencies"], features, "document frequencies"
        ),
        "column_norms": _sequence(raw["column_norms"], features, "column norms"),
        "basis": _matrix(raw["basis"], features, dimensions, "basis"),
        "forward": _direction(raw["forward"], features, dimensions),
        "backward": _direction(raw["backward"], features, dimensions),
    }
    state = DualContextState(**{**raw, **replacements})
    validate_state(state, config)
    model = DualContextModel(**config.to_dict())
    model._state = state
    return model


def save_model(value: Any, path: Path, *, overwrite: bool = False) -> str | None:
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be boolean")
    encoded = _canonical(value)
    if path.is_symlink():
        raise ValueError("model output must not be a symbolic link")
    if path.exists():
        metadata = path.stat()
        if not overwrite:
            raise FileExistsError(f"model output already exists: {path}")
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("overwrite requires an unaliased regular file")
    descriptor, name = tempfile.mkstemp(prefix=".dual-context-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    published = False
    warning = None
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        published = True
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            if published:
                warning = "model published; private temporary-file cleanup failed"
    return warning


def load_model(path: Path) -> Any:
    with path.open("rb") as stream:
        raw = stream.read(MAX_ARTIFACT_BYTES + 1)
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ValueError("dual-context artifact exceeds its byte limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise ValueError("dual-context artifact must be UTF-8") from error
    depth = 0
    quoted = escaped = False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise ValueError("dual-context artifact exceeds JSON depth limit")
        elif character in "]}":
            depth -= 1
    try:
        return parse_json_value(text)
    except (ValueError, RecursionError) as error:
        raise ValueError("invalid dual-context artifact JSON") from error
