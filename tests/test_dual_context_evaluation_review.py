"""Independent public-API arithmetic, admission order and mutation regressions."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from turnscope.dual_context import ContextEdge, ContextRecord, DualContextModel
from turnscope.dual_context_evaluation import evaluate_dual_context


def _load_independent_benchmark():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_dual_context.py"
    name = "turnscope_evaluation_review_benchmark"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Dataclass resolution uses the defining module.
    spec.loader.exec_module(module)
    return module


class AuthoredVectors(DualContextModel):
    """An explicit geometric oracle, not a fitted-model quality assertion."""

    def __init__(self, contexts, forward, backward=None, *, mutate=False):
        self.contexts = contexts
        self.forward = forward
        self.backward = forward if backward is None else backward
        self.calls = []
        self.digest_reads = 0
        self.version = 0
        self.mutate = mutate

    @property
    def digest(self):
        self.digest_reads += 1
        return str(self.version) * 64

    def project_context(self, text):
        self.calls.append(("context", text))
        return SimpleNamespace(vector=self.contexts[text])

    def predict(self, text):
        self.calls.append(("query", text))
        if self.mutate:
            self.version += 1
        return SimpleNamespace(
            forward=SimpleNamespace(vector=self.forward[text]),
            backward=SimpleNamespace(vector=self.backward[text]),
        )


def test_scorable_denominator_matches_independent_benchmark_with_no_candidate_projection():
    benchmark = _load_independent_benchmark()
    queries = [
        ContextRecord("c", "q1", "one"),
        ContextRecord("c", "q2", "two"),
        ContextRecord("c", "q3", "unknown"),
    ]
    candidates = [queries[0], ContextRecord("c", "oov", "unknown-context")]
    edges = [
        ContextEdge("c", "q1", "oov"),
        ContextEdge("c", "q2", "q1"),
        ContextEdge("c", "q3", "q1"),
    ]
    model = AuthoredVectors(
        {"one": (1.0, 0.0), "unknown-context": None},
        {"one": (1.0, 0.0), "two": (1.0, 0.0), "unknown": None},
    )
    report = evaluate_dual_context(model, queries, candidates, edges, direction="forward", ks=(1,))
    rows, summary = report["rows"], report["summary"]
    assert [row["reciprocal_rank"] for row in rows] == [0.0, 1.0, 0.0]
    assert rows[0]["reason"] == "no_projectable_candidates"
    assert rows[0]["query_projectable"] is True
    assert rows[0]["positives"] == 1 and rows[0]["projectable_positives"] == 0
    assert summary["query_coverage"] == pytest.approx(2 / 3)
    assert summary["all_gold_queries"]["mean_reciprocal_rank"] == pytest.approx(1 / 3)
    assert summary["scorable_gold_queries"]["queries"] == 2
    assert summary["scorable_gold_queries"]["mean_reciprocal_rank"] == 0.5
    assert report["training_separation_verified"] is False

    independent = []
    conditional = []
    for query, edge, row in zip(queries, edges, rows, strict=True):
        vector = model.forward[query.text]
        scores = {}
        for candidate in candidates:
            if candidate.key == query.key:
                continue
            context = model.contexts[candidate.text]
            scores[candidate.key] = (
                None
                if vector is None or context is None
                else sum(a * b for a, b in zip(vector, context, strict=True))
            )
        metrics = benchmark.ranking_metrics(scores, (edge.context_key,), (1,))
        assert row["reciprocal_rank"] == metrics["reciprocal_rank"]
        assert row["recall_at_k"] == metrics["recall"]
        independent.append(metrics["reciprocal_rank"])
        if vector is not None:
            conditional.append(metrics["reciprocal_rank"])
    assert summary["all_gold_queries"]["mean_reciprocal_rank"] == math.fsum(independent) / 3
    assert summary["scorable_gold_queries"]["mean_reciprocal_rank"] == math.fsum(conditional) / 2


def test_public_direction_ties_and_multiple_positive_denominators_are_hand_calculated():
    query = ContextRecord("c", "query", "q")
    candidates = [
        ContextRecord("c", "a", "a"),
        ContextRecord("c", "b", "b"),
        ContextRecord("c", "z", "z"),
        ContextRecord("c", "missing", "oov"),
        query,
    ]
    edges = [ContextEdge("c", "query", item) for item in ("b", "z", "missing")]
    model = AuthoredVectors(
        {"a": (1.0, 0.0), "b": (1.0, 0.0), "z": (0.0, 1.0), "oov": None, "q": (1.0, 0.0)},
        {"q": (1.0, 0.0)},
        {"q": (0.0, 1.0)},
    )
    forward = evaluate_dual_context(
        model, [query], candidates, edges, direction="forward", ks=(1, 2, 9)
    )
    backward = evaluate_dual_context(
        model, [query], candidates, edges, direction="backward", ks=(1, 2, 9)
    )
    first, second = forward["rows"][0], backward["rows"][0]
    assert [item["utterance_id"] for item in first["top_candidates"]] == ["a", "b", "z"]
    assert [item["utterance_id"] for item in second["top_candidates"]] == ["z", "a", "b"]
    assert first["reciprocal_rank"] == 0.5 and second["reciprocal_rank"] == 1.0
    assert first["recall_at_k"] == {"1": 0.0, "2": 1 / 3, "9": 2 / 3}
    assert second["recall_at_k"] == {"1": 1 / 3, "2": 1 / 3, "9": 2 / 3}
    assert first["candidates"] == 4 and first["projectable_candidates"] == 3
    assert first["positives"] == 3 and first["projectable_positives"] == 2
    assert model.version == 0


def test_all_no_gold_rows_are_undefined_even_if_the_query_projects():
    model = AuthoredVectors({"a": (1.0,)}, {"q": (1.0,), "oov": None})
    report = evaluate_dual_context(
        model,
        [ContextRecord("c", "q", "q"), ContextRecord("c", "u", "oov")],
        [ContextRecord("c", "a", "a")],
        [],
        direction="forward",
        ks=(1,),
    )
    assert report["summary"]["query_coverage"] is None
    assert report["summary"]["queries_without_positives"] == 2
    for group in ("all_gold_queries", "scorable_gold_queries"):
        assert report["summary"][group] == {
            "queries": 0,
            "mean_reciprocal_rank": None,
            "mean_recall_at_k": {"1": None},
        }
    assert all(row["reciprocal_rank"] is None for row in report["rows"])


def test_pair_budget_rejects_before_relationship_iteration_or_model_access():
    model = AuthoredVectors({}, {})

    def untouched_edges():
        raise AssertionError("relationships must not be consumed after pair budget failure")
        yield

    with pytest.raises(ValueError, match="max_score_pairs"):
        evaluate_dual_context(
            model,
            [ContextRecord("c", "q", "q"), ContextRecord("c", "r", "r")],
            [ContextRecord("c", "a", "a"), ContextRecord("c", "b", "b")],
            untouched_edges(),
            direction="forward",
            max_score_pairs=3,
        )
    assert model.calls == [] and model.digest_reads == 0


def test_combined_utf8_text_budget_is_not_a_character_or_per_catalog_budget():
    model = AuthoredVectors({"😀": (1.0,)}, {"é": (1.0,)})
    queries = [ContextRecord("c", "q", "é")]
    candidates = [ContextRecord("c", "a", "😀")]
    edges = [ContextEdge("c", "q", "a")]
    with pytest.raises(ValueError, match="text byte limit"):
        evaluate_dual_context(
            model, queries, candidates, edges, direction="forward", max_text_bytes=5
        )
    assert model.calls == [] and model.digest_reads == 0
    report = evaluate_dual_context(
        model, queries, candidates, edges, direction="forward", max_text_bytes=6
    )
    assert report["summary"]["all_gold_queries"]["mean_reciprocal_rank"] == 1.0


def test_catalog_admission_reads_only_one_over_limit_and_never_projects():
    model = AuthoredVectors({}, {})
    consumed = []

    def queries():
        for name in ("q1", "q2"):
            consumed.append(name)
            yield ContextRecord("c", name, name)
        raise AssertionError("an over-limit input iterator must not be drained")

    with pytest.raises(ValueError, match="record limit"):
        evaluate_dual_context(
            model, queries(), [ContextRecord("c", "a", "a")], [], direction="forward", max_queries=1
        )
    assert consumed == ["q1", "q2"]
    assert model.calls == [] and model.digest_reads == 0


def test_shared_id_text_difference_is_rejected_before_model_access_without_echo():
    model = AuthoredVectors({}, {})
    with pytest.raises(ValueError) as failed:
        evaluate_dual_context(
            model,
            [ContextRecord("c", "q", "PRIVATE\r\nTEXT")],
            [ContextRecord("c", "q", "PRIVATE\nTEXT")],
            [],
            direction="forward",
        )
    assert "PRIVATE" not in str(failed.value)
    assert model.calls == [] and model.digest_reads == 0


def test_model_digest_mutation_cannot_produce_a_successful_evaluation_report():
    model = AuthoredVectors({"a": (1.0,)}, {"q": (1.0,)}, mutate=True)
    with pytest.raises(ValueError, match="frozen model changed"):
        evaluate_dual_context(
            model,
            [ContextRecord("c", "q", "q")],
            [ContextRecord("c", "a", "a")],
            [ContextEdge("c", "q", "a")],
            direction="forward",
        )
    # Detection is not a claim to roll back arbitrary subclass side effects.
    assert model.version == 1
