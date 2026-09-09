"""Hand-calculated fixed-pool retrieval and actual frozen-model integration."""

from __future__ import annotations

import pytest

from turnscope.dual_context import ContextEdge, ContextRecord, DualContextModel
from turnscope.dual_context_evaluation import _aggregate, _score_query, evaluate_dual_context


def test_multipositive_recall_keeps_unprojectable_gold_in_denominator():
    query = ("c", "q")
    candidates = {("c", "a"): (1.0, 0.0), ("c", "b"): (1.0, 0.0), ("c", "z"): None}
    row = _score_query(
        query, (1.0, 0.0), candidates, frozenset((("c", "b"), ("c", "z"))), (1, 2, 9)
    )
    assert row["reciprocal_rank"] == 0.5
    assert row["recall_at_k"] == {"1": 0.0, "2": 0.5, "9": 0.5}
    assert row["positives"] == 2 and row["projectable_positives"] == 1
    assert [item["utterance_id"] for item in row["top_candidates"]] == ["a", "b"]


def test_self_candidate_is_excluded_and_oov_query_not_dropped_from_all_query_mean():
    key = ("c", "q")
    pool = {key: (1.0, 0.0), ("c", "a"): (1.0, 0.0), ("c", "b"): None}
    relevant = frozenset((("c", "a"),))
    good = _score_query(key, (1.0, 0.0), pool, relevant, (1, 3))
    missing = _score_query(key, None, pool, relevant, (1, 3))
    no_gold = _score_query(key, (1.0, 0.0), pool, frozenset(), (1, 3))
    assert good["candidates"] == 2 and good["projectable_candidates"] == 1
    assert good["reciprocal_rank"] == 1.0
    assert missing["reason"] == "unprojectable_query" and missing["reciprocal_rank"] == 0.0
    assert no_gold["reason"] == "no_positive_context" and no_gold["reciprocal_rank"] is None
    summary = _aggregate([good, missing, no_gold], (1, 3))
    assert summary["queries_without_positives"] == 1
    assert summary["query_coverage"] == 0.5
    assert summary["all_gold_queries"]["mean_reciprocal_rank"] == 0.5
    assert summary["scorable_gold_queries"]["mean_reciprocal_rank"] == 1.0


def test_no_projectable_candidates_and_empty_aggregate_remain_explicit():
    row = _score_query(("c", "q"), (1.0,), {("c", "a"): None}, frozenset((("c", "a"),)), (1,))
    assert row["reason"] == "no_projectable_candidates"
    assert row["reciprocal_rank"] == 0.0 and row["recall_at_k"] == {"1": 0.0}
    assert _aggregate([], (1,))["all_gold_queries"]["mean_reciprocal_rank"] is None


@pytest.fixture
def fitted():
    pytest.importorskip("numpy")
    records = [
        ContextRecord("train", name, word)
        for name, word in (
            ("a", "alpha"),
            ("b", "beta"),
            ("d", "gamma"),
            ("ca", "alpha"),
            ("cb", "beta"),
            ("cd", "gamma"),
        )
    ]
    edges = [ContextEdge("train", name, "c" + name) for name in ("a", "b", "d")]
    return DualContextModel(n_components=3, n_clusters=2).fit(records, edges, edges)


def task():
    return (
        [ContextRecord("heldout", "q", "alpha"), ContextRecord("heldout", "unknown", "zzzz")],
        [ContextRecord("heldout", "a", "beta"), ContextRecord("heldout", "z", "alpha")],
        [ContextEdge("heldout", "q", "z"), ContextEdge("heldout", "unknown", "z")],
    )


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_actual_model_evaluation_is_frozen_and_reordering_invariant(fitted, direction):
    queries, candidates, edges = task()
    before = fitted.digest
    report = evaluate_dual_context(fitted, queries, candidates, edges, direction=direction)
    assert report["training_separation_verified"] is False
    assert report["model_digest"] == fitted.digest == before
    assert report["summary"]["all_gold_queries"]["mean_reciprocal_rank"] == 0.5
    assert report["rows"][0]["top_candidates"][0]["utterance_id"] == "z"
    assert report == evaluate_dual_context(
        fitted, reversed(queries), reversed(candidates), reversed(edges), direction=direction
    )


@pytest.mark.parametrize(
    "options",
    [
        {"direction": "other"},
        {"ks": []},
        {"ks": ()},
        {"ks": (2, 1)},
        {"ks": (1, 1)},
        {"ks": (True,)},
        {"ks": (101,)},
        {"max_queries": False},
        {"max_candidates": 0},
        {"max_score_pairs": 0},
        {"max_relationships": True},
        {"max_text_bytes": -1},
    ],
)
def test_invalid_evaluation_options_fail_before_fitting_or_projection(options):
    values = {"direction": "forward", **options}
    with pytest.raises(ValueError):
        evaluate_dual_context(DualContextModel(), *task(), **values)


@pytest.mark.parametrize(
    "case",
    [
        "query_duplicate",
        "candidate_duplicate",
        "different_text",
        "edge_duplicate",
        "missing_source",
        "missing_target",
    ],
)
def test_relationship_and_identity_errors_fail_before_using_unfitted_model(case):
    queries, candidates, edges = task()
    if case == "query_duplicate":
        queries.append(queries[0])
    elif case == "candidate_duplicate":
        candidates.append(candidates[0])
    elif case == "different_text":
        candidates.append(ContextRecord("heldout", "q", "not alpha"))
    elif case == "edge_duplicate":
        edges.append(edges[0])
    elif case == "missing_source":
        edges.append(ContextEdge("heldout", "absent", "z"))
    else:
        edges.append(ContextEdge("heldout", "q", "absent"))
    with pytest.raises(ValueError) as failed:
        evaluate_dual_context(DualContextModel(), queries, candidates, edges, direction="forward")
    assert "not been fitted" not in str(failed.value)


@pytest.mark.parametrize(
    "limits",
    [
        {"max_queries": 1},
        {"max_candidates": 1},
        {"max_score_pairs": 3},
        {"max_relationships": 1},
        {"max_text_bytes": 17},
    ],
)
def test_resource_limits_fail_before_projection(limits):
    with pytest.raises(ValueError) as failed:
        evaluate_dual_context(DualContextModel(), *task(), direction="forward", **limits)
    assert "not been fitted" not in str(failed.value)


def test_exact_input_resource_boundaries_are_accepted(fitted):
    report = evaluate_dual_context(
        fitted,
        *task(),
        direction="forward",
        max_queries=2,
        max_candidates=2,
        max_relationships=2,
        max_score_pairs=4,
        max_text_bytes=18,
    )
    assert report["relationships"] == 2
