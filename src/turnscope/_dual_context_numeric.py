"""Optional NumPy training implementation; never imported by frozen inference."""

from __future__ import annotations

import hashlib
import importlib
import math
import struct
from dataclasses import dataclass
from typing import Any

from .dual_context import ZERO_TOLERANCE, ClusterState, DirectionState, DualContextConfig


@dataclass(frozen=True, slots=True)
class NumericDualFit:
    column_norms: tuple[float, ...]
    basis: tuple[tuple[float, ...], ...]
    singular_values: tuple[float, ...]
    forward: DirectionState
    backward: DirectionState
    numerical_rank: int
    svd_cutoff: float


def _rows(array: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in array)


def _normalize(np: Any, matrix: Any) -> tuple[Any, Any]:
    norms = np.linalg.norm(matrix, axis=1)
    valid = norms > ZERO_TOLERANCE
    result = np.zeros_like(matrix)
    result[valid] = matrix[valid] / norms[valid, None]
    return result, valid


def _distances(np: Any, vectors: Any, centers: Any) -> Any:
    # The algebra avoids an N x clusters x dimensions temporary.
    return np.maximum(
        0.0,
        np.sum(vectors * vectors, axis=1)[:, None]
        + np.sum(centers * centers, axis=1)[None, :]
        - 2 * vectors @ centers.T,
    )


def fit_kmeans(vectors: Any, config: DualContextConfig, np: Any) -> ClusterState:
    """Bounded Euclidean Lloyd iterations, seeded farthest-first initialization."""
    if not len(vectors):
        return ClusterState((), (), 0, True, 0.0, 0, 0, "no_nonzero_vectors")
    unique = np.unique(vectors, axis=0)
    distinct = len(unique)
    count = min(config.n_clusters, distinct)

    def priority(index: int) -> bytes:
        packed = b"".join(struct.pack(">d", float(value)) for value in unique[index])
        return hashlib.sha256(struct.pack(">I", config.seed) + packed).digest()

    priorities = [priority(index) for index in range(distinct)]
    chosen = [min(range(distinct), key=lambda index: (priorities[index], index))]
    nearest = _distances(np, unique, unique[chosen]).ravel()
    while len(chosen) < count:
        available = [index for index in range(distinct) if index not in chosen]
        index = min(available, key=lambda item: (-float(nearest[item]), priorities[item], item))
        chosen.append(index)
        nearest = np.minimum(nearest, _distances(np, unique, unique[[index]]).ravel())
    centers = unique[chosen].copy()
    previous: Any = None
    converged = False
    iteration = 0
    for _ in range(config.max_kmeans_iterations):
        iteration += 1
        labels = np.argmin(_distances(np, vectors, centers), axis=1)
        occupied = [index for index in range(len(centers)) if np.any(labels == index)]
        updated = np.asarray([vectors[labels == index].mean(axis=0) for index in occupied])
        # Center magnitudes deliberately remain means, not unit/spherical centers.
        stable = previous is not None and np.array_equal(labels, previous)
        centers = updated
        if stable:
            converged = True
            break
        previous = labels
    # A last assignment reports the objective for the centers actually exported.
    distances = _distances(np, vectors, centers)
    labels = np.argmin(distances, axis=1)
    occupied = [index for index in range(len(centers)) if np.any(labels == index)]
    if len(occupied) != len(centers):
        centers = centers[occupied]
        distances = _distances(np, vectors, centers)
        labels = np.argmin(distances, axis=1)
        converged = False
    counts = tuple(int(np.sum(labels == index)) for index in range(len(centers)))
    objective = float(np.sum(distances[np.arange(len(vectors)), labels]))
    reason = "distinct_vectors_below_requested" if distinct < config.n_clusters else "ok"
    if len(centers) < count:
        reason = "empty_clusters_removed"
    return ClusterState(
        _rows(centers), counts, iteration, converged, objective, len(vectors), distinct, reason
    )


def _direction(
    np: Any, x: Any, u: Any, sigma: Any, pairs: list[tuple[int, int]], config: DualContextConfig
) -> DirectionState:
    aligned = np.zeros((len(u), x.shape[1]), dtype=np.float64)
    for source, context in pairs:
        aligned[context] += x[source]
    term_map = (aligned.T @ u) / sigma[None, :]
    term_units, term_valid = _normalize(np, term_map)
    context_units, context_valid = _normalize(np, u)
    support = np.zeros(x.shape[1], dtype=np.int64)
    projectable = np.zeros(x.shape[1], dtype=np.int64)
    totals = np.zeros(x.shape[1], dtype=np.float64)
    for source, context in pairs:
        # Presence, not count or TF-IDF magnitude, weights each edge's range sample.
        present = np.flatnonzero(x[source])
        support[present] += 1
        if context_valid[context]:
            projectable[present] += 1
            distances = np.clip(1 - term_units[present] @ context_units[context], 0.0, 1.0)
            totals[present] += distances
    ranges = tuple(
        float(totals[index] / projectable[index])
        if term_valid[index] and projectable[index]
        else None
        for index in range(x.shape[1])
    )
    utterances, valid = _normalize(np, (x @ term_map) / sigma[None, :])
    if not np.isfinite(term_map).all() or not np.isfinite(utterances).all():
        raise ValueError("dual-context training produced nonfinite values")
    return DirectionState(
        _rows(term_map),
        ranges,
        tuple(int(value) for value in support),
        tuple(int(value) for value in projectable),
        fit_kmeans(utterances[valid], config, np),
    )


def fit_numeric_dual(
    rows: list[dict[int, float]],
    features: int,
    context_indices: list[int],
    forward: list[tuple[int, int]],
    backward: list[tuple[int, int]],
    config: DualContextConfig,
) -> NumericDualFit:
    try:
        np = importlib.import_module("numpy")
    except ImportError as error:
        raise ValueError(
            "fitting dual context requires the optional turnscope[context] extra"
        ) from error
    x = np.zeros((len(rows), features), dtype=np.float64)
    for index, row in enumerate(rows):
        for column, weight in row.items():
            x[index, column] = weight
    column_norms = np.linalg.norm(x, axis=0)
    if np.any(column_norms <= 0) or not np.isfinite(column_norms).all():
        raise ValueError("dual-context training has invalid feature column norms")
    x /= column_norms[None, :]
    contexts = x[context_indices]
    try:
        u, singular, vt = np.linalg.svd(contexts, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("dual-context SVD did not converge") from error
    cutoff = max(
        ZERO_TOLERANCE, max(contexts.shape) * float(np.finfo(np.float64).eps) * float(singular[0])
    )
    rank = int(np.sum(singular > cutoff))
    start = int(config.drop_first)
    stop = min(rank, config.n_components + start)
    if stop <= start:
        raise ValueError("training contexts have no retained numerical rank")
    u = u[:, start:stop].copy()
    sigma = singular[start:stop].copy()
    basis = vt[start:stop].T.copy()
    for dimension in range(len(sigma)):
        pivot = int(np.argmax(np.abs(basis[:, dimension])))
        if basis[pivot, dimension] < 0:
            basis[:, dimension] *= -1
            u[:, dimension] *= -1
    if not all(math.isfinite(float(item)) for item in sigma):
        raise ValueError("dual-context SVD returned nonfinite singular values")
    return NumericDualFit(
        tuple(float(value) for value in column_norms),
        _rows(basis),
        tuple(float(value) for value in sigma),
        _direction(np, x, u, sigma, forward, config),
        _direction(np, x, u, sigma, backward, config),
        rank,
        cutoff,
    )
