"""Frozen, bounded multi-positive retrieval over explicitly supplied candidate pools."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from .dual_context import ContextEdge, ContextRecord, DualContextModel

NodeKey = tuple[str, str]
Vector = tuple[float, ...] | None


def _integer(value: int, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _catalog(
    records: Iterable[ContextRecord], maximum: int, max_bytes: int
) -> tuple[dict[NodeKey, ContextRecord], int]:
    result: dict[NodeKey, ContextRecord] = {}
    size = 0
    for record in records:
        if len(result) >= maximum:
            raise ValueError("retrieval catalog exceeds its record limit")
        if not isinstance(record, ContextRecord):
            raise TypeError("retrieval catalogs require ContextRecord values")
        if record.key in result:
            raise ValueError("retrieval catalog has duplicate composite IDs")
        size += len(record.text.encode("utf-8"))
        if size > max_bytes:
            raise ValueError("retrieval catalogs exceed the text byte limit")
        result[record.key] = record
    if not result:
        raise ValueError("retrieval catalogs must not be empty")
    return dict(sorted(result.items())), size


def _score_query(
    key: NodeKey,
    vector: Vector,
    candidates: Mapping[NodeKey, Vector],
    positives: frozenset[NodeKey],
    ks: tuple[int, ...],
) -> dict[str, Any]:
    """Unprojectable candidates are unretrievable, not removed from gold counts."""
    eligible = {item: value for item, value in candidates.items() if item != key}
    projectable: dict[NodeKey, tuple[float, ...]] = {}
    for item, value in eligible.items():
        if value is not None:
            projectable[item] = value
    ranked = []
    if vector is not None:
        for item, value in projectable.items():
            score = math.fsum(a * b for a, b in zip(vector, value, strict=True))
            if not math.isfinite(score):
                raise ValueError("retrieval score must be finite")
            ranked.append((item, max(-1.0, min(1.0, score))))
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
    first = next((rank for rank, (item, _) in enumerate(ranked, 1) if item in positives), None)
    if not positives:
        reason = "no_positive_context"
    elif vector is None:
        reason = "unprojectable_query"
    elif not projectable:
        reason = "no_projectable_candidates"
    else:
        reason = "ok"
    return {
        "conversation_id": key[0],
        "utterance_id": key[1],
        "reason": reason,
        "query_projectable": vector is not None,
        "candidates": len(eligible),
        "projectable_candidates": len(projectable),
        "positives": len(positives),
        "projectable_positives": len(positives.intersection(projectable)),
        "reciprocal_rank": (1.0 / first if first is not None else 0.0) if positives else None,
        "recall_at_k": {
            str(k): sum(item in positives for item, _ in ranked[:k]) / len(positives)
            if positives
            else None
            for k in ks
        },
        "top_candidates": [
            {"conversation_id": item[0], "utterance_id": item[1], "cosine": score}
            for item, score in ranked[: max(ks)]
        ],
    }


def _aggregate(rows: list[dict[str, Any]], ks: tuple[int, ...]) -> dict[str, Any]:
    gold = [row for row in rows if row["positives"]]
    scored = [row for row in gold if row["query_projectable"]]

    def rates(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "queries": len(selected),
            "mean_reciprocal_rank": math.fsum(row["reciprocal_rank"] for row in selected)
            / len(selected)
            if selected
            else None,
            "mean_recall_at_k": {
                str(k): math.fsum(row["recall_at_k"][str(k)] for row in selected) / len(selected)
                if selected
                else None
                for k in ks
            },
        }

    return {
        "queries": len(rows),
        "queries_without_positives": len(rows) - len(gold),
        "query_coverage": len(scored) / len(gold) if gold else None,
        "all_gold_queries": rates(gold),
        "scorable_gold_queries": rates(scored),
    }


def evaluate_dual_context(
    model: DualContextModel,
    queries: Iterable[ContextRecord],
    candidates: Iterable[ContextRecord],
    edges: Iterable[ContextEdge],
    *,
    direction: str,
    ks: tuple[int, ...] = (1, 5, 10),
    max_queries: int = 1024,
    max_candidates: int = 4096,
    max_score_pairs: int = 1_000_000,
    max_relationships: int = 60_000,
    max_text_bytes: int = 32 * 1024 * 1024,
) -> dict[str, Any]:
    """Evaluate caller-declared relevance without fitting or inventing relationships.

    Zero/OOV queries count as zero in the all-gold-query means and are reported
    separately from scorable-only means. Unprojectable positive candidates remain
    in recall denominators. No-positive queries are undefined, not evidence of a
    correct negative. Self candidates are excluded. This function cannot establish
    train/test separation from a model digest; callers must verify their splits.
    """
    if not isinstance(model, DualContextModel):
        raise TypeError("evaluation requires a fitted DualContextModel")
    if direction not in ("forward", "backward"):
        raise ValueError("direction must be forward or backward")
    _integer(max_queries, "max_queries", 10_000)
    _integer(max_candidates, "max_candidates", 30_000)
    _integer(max_score_pairs, "max_score_pairs", 10_000_000)
    _integer(max_relationships, "max_relationships", 200_000)
    _integer(max_text_bytes, "max_text_bytes", 128 * 1024 * 1024)
    if not isinstance(ks, tuple) or not ks or len(ks) > 100:
        raise ValueError("ks must be a nonempty bounded tuple")
    for k in ks:
        _integer(k, "k", 100)
    if tuple(sorted(set(ks))) != ks:
        raise ValueError("ks must be strictly increasing and unique")
    query_map, size = _catalog(queries, max_queries, max_text_bytes)
    candidate_map, _ = _catalog(candidates, max_candidates, max_text_bytes - size)
    if len(query_map) * len(candidate_map) > max_score_pairs:
        raise ValueError("retrieval work exceeds max_score_pairs")
    for key in query_map.keys() & candidate_map.keys():
        if query_map[key].text != candidate_map[key].text:
            raise ValueError("a shared composite ID must have identical query/context text")
    positives: dict[NodeKey, set[NodeKey]] = {key: set() for key in query_map}
    count = 0
    for edge in edges:
        if count >= max_relationships:
            raise ValueError("retrieval relationships exceed their limit")
        if not isinstance(edge, ContextEdge):
            raise TypeError("retrieval relationships require ContextEdge values")
        if edge.source_key not in query_map or edge.context_key not in candidate_map:
            raise ValueError("retrieval relationship endpoint is absent from its declared catalog")
        if edge.context_key in positives[edge.source_key]:
            raise ValueError("retrieval relationships must be unique")
        positives[edge.source_key].add(edge.context_key)
        count += 1
    before = model.digest
    projected = {
        key: model.project_context(record.text).vector for key, record in candidate_map.items()
    }
    rows = [
        _score_query(
            key,
            getattr(model.predict(record.text), direction).vector,
            projected,
            frozenset(positives[key]),
            ks,
        )
        for key, record in query_map.items()
    ]
    if model.digest != before:
        raise ValueError("frozen model changed during evaluation")
    return {
        "format": "turnscope.dual-context.retrieval.v1",
        "private_data": True,
        "model_digest": before,
        "direction": direction,
        "candidate_policy": "caller-supplied fixed pool; self excluded",
        "tie_policy": "descending cosine then ascending composite ID; exact float ties",
        "training_separation_verified": False,
        "relationships": count,
        "score_pair_upper_bound": len(query_map) * len(candidate_map),
        "summary": _aggregate(rows, ks),
        "rows": rows,
    }
