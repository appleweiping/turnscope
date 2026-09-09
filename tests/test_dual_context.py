"""Arithmetic oracles and hostile-artifact tests; no downloaded/model-generated fixtures."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import unicodedata
from collections import Counter
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from turnscope import _dual_context_artifact as artifact
from turnscope import _dual_context_numeric as numeric
from turnscope.dual_context import (
    TOKENIZER_VERSION,
    ContextEdge,
    ContextRecord,
    DualContextConfig,
    DualContextModel,
    _distance,
    _unit,
)


def training():
    records = [
        ContextRecord("c", str(index), text)
        for index, text in enumerate(
            ("ask ask red", "tell blue", "red blue", "yes red", "no blue", "isolated")
        )
    ]
    edges = [
        ContextEdge("c", "0", "3"),
        ContextEdge("c", "1", "4"),
        ContextEdge("c", "2", "3"),
        ContextEdge("c", "2", "4"),
    ]
    reverse = [ContextEdge(edge.conversation_id, edge.context_id, edge.source_id) for edge in edges]
    return records, edges, reverse


def trained(**kwargs):
    return DualContextModel(n_components=2, n_clusters=2, **kwargs).fit(*training())


def resign(value):
    body = {key: item for key, item in value.items() if key != "sha256"}
    value["sha256"] = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return value


def oracle(records, forward, backward, components=2, drop_first=False):
    """Independent dense matrix equations, not the production numeric helpers."""
    ordered = sorted(records, key=lambda row: row.key)
    counts = [Counter(row.text.casefold().split()) for row in ordered]
    vocab = sorted(set().union(*(row.keys() for row in counts)))
    df = [sum(term in row for row in counts) for term in vocab]
    idf = np.log((1 + len(records)) / (1 + np.array(df))) + 1
    a = np.array([[row[term] for term in vocab] for row in counts]) * idf
    a /= np.linalg.norm(a, axis=1)[:, None]
    columns = np.linalg.norm(a, axis=0)
    a /= columns
    lookup = {row.key: index for index, row in enumerate(ordered)}
    context_keys = sorted({edge.context_key for edge in forward + backward})
    context_index = {key: index for index, key in enumerate(context_keys)}
    c = a[[lookup[key] for key in context_keys]]
    u, sigma, vt = np.linalg.svd(c, full_matrices=False)
    rank = int(np.sum(sigma > max(1e-12, max(c.shape) * np.finfo(float).eps * sigma[0])))
    selected = slice(int(drop_first), min(rank, components + int(drop_first)))
    u, sigma, v = u[:, selected], sigma[selected], vt[selected].T
    for index in range(len(sigma)):
        if v[np.argmax(np.abs(v[:, index])), index] < 0:
            u[:, index] *= -1
            v[:, index] *= -1
    maps = []
    ranges = []
    for edges in (forward, backward):
        source_rows = np.array([a[lookup[edge.source_key]] for edge in edges])
        context_rows = np.array([u[context_index[edge.context_key]] for edge in edges])
        w = source_rows.T @ context_rows @ np.diag(1 / sigma)
        maps.append(w)
        values = []
        for column in range(len(vocab)):
            distances = []
            for row in range(len(edges)):
                if source_rows[row, column] and np.linalg.norm(context_rows[row]) > 1e-12:
                    distance = 1 - np.dot(w[column], context_rows[row]) / (
                        np.linalg.norm(w[column]) * np.linalg.norm(context_rows[row])
                    )
                    distances.append(min(1.0, max(0.0, float(distance))))
            values.append(
                sum(distances) / len(distances)
                if distances and np.linalg.norm(w[column]) > 1e-12
                else None
            )
        ranges.append(values)
    return a, columns, u, sigma, v, maps, ranges


@pytest.mark.parametrize("drop_first", [False, True])
def test_independent_full_equation_oracle(drop_first):
    records, forward, backward = training()
    a, columns, _, sigma, v, maps, ranges = oracle(
        records, forward, backward, drop_first=drop_first
    )
    model = trained(drop_first=drop_first)
    assert model.state.column_norms == pytest.approx(columns)
    assert np.array(model.state.basis) == pytest.approx(v)
    assert model.state.singular_values == pytest.approx(sigma)
    for index, direction in enumerate((model.state.forward, model.state.backward)):
        assert np.array(direction.term_map) == pytest.approx(maps[index])
        for actual, expected in zip(direction.term_ranges, ranges[index], strict=True):
            assert actual == pytest.approx(expected) if expected is not None else actual is None
        for row, record in enumerate(records):
            expected = a[row] @ maps[index] @ np.diag(1 / sigma)
            norm = np.linalg.norm(expected)
            prediction = model.predict(record.text)
            actual = (prediction.forward, prediction.backward)[index]
            if norm > 1e-12:
                assert actual.vector == pytest.approx(expected / norm)
            else:
                assert actual.vector is None
            eligible = [
                (weight, value)
                for weight, value in zip(a[row], ranges[index], strict=True)
                if value is not None
            ]
            denominator = sum(weight for weight, _ in eligible)
            expected_range = (
                sum(weight * value for weight, value in eligible) / denominator
                if denominator
                else None
            )
            assert (
                actual.range == pytest.approx(expected_range)
                if expected_range is not None
                else actual.range is None
            )
    for row, record in enumerate(records):
        expected = a[row] @ v @ np.diag(1 / sigma)
        norm = np.linalg.norm(expected)
        if norm > 1e-12:
            assert model.project_context(record.text).vector == pytest.approx(expected / norm)
    assert model.training_summary()["svd_fits"] == 1


def test_identity_relationship_requires_second_sigma_and_distinct_ids():
    texts = ["red red blue", "red green", "blue blue green", "green red blue"]
    records = [
        ContextRecord("c", f"{prefix}{index}", text)
        for prefix in ("s", "t")
        for index, text in enumerate(texts)
    ]
    edges = [ContextEdge("c", f"s{index}", f"t{index}") for index in range(len(texts))]
    model = DualContextModel(n_components=3, n_clusters=1).fit(records, edges, edges)
    assert np.array(model.state.forward.term_map) == pytest.approx(np.array(model.state.basis))
    assert len({round(value, 8) for value in model.state.singular_values}) > 1
    for text in texts:
        assert model.predict(text).forward.vector == pytest.approx(
            model.project_context(text).vector
        )
        assert model.predict(text).orientation == 0
        assert model.predict(text).shift == 0


def test_hand_balanced_orthogonal_context_range():
    records = [
        ContextRecord("c", "s", "red blue"),
        ContextRecord("c", "a", "red"),
        ContextRecord("c", "b", "blue"),
    ]
    edges = [ContextEdge("c", "s", "a"), ContextEdge("c", "s", "b")]
    model = DualContextModel(n_components=2, n_clusters=1).fit(records, edges, edges)
    assert model.predict("red blue").forward.range == pytest.approx(1 - 1 / math.sqrt(2))
    for term in model.term_statistics():
        assert term["forward"]["support_edges"] == term["forward"]["projectable_edges"] == 2
        assert term["orientation"] == term["shift"] == 0


def test_shared_orthogonal_coordinate_change_preserves_geometry():
    # Equal singular values allow an arbitrary orthogonal shared change of basis.
    records = [
        ContextRecord("c", "s0", "red"),
        ContextRecord("c", "s1", "blue"),
        ContextRecord("c", "t0", "red"),
        ContextRecord("c", "t1", "blue"),
    ]
    edges = [ContextEdge("c", "s0", "t0"), ContextEdge("c", "s1", "t1")]
    model = DualContextModel(n_components=2, n_clusters=1).fit(records, edges, edges)
    assert model.state.singular_values[0] == pytest.approx(model.state.singular_values[1])
    value = model.to_dict()
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    value["state"]["basis"] = (np.array(value["state"]["basis"]) @ rotation).tolist()
    for name in ("forward", "backward"):
        direction = value["state"][name]
        direction["term_map"] = (np.array(direction["term_map"]) @ rotation).tolist()
        direction["clustering"]["centers"] = (
            np.array(direction["clustering"]["centers"]) @ rotation
        ).tolist()
    rotated = DualContextModel.from_dict(resign(value))
    for text in ("red", "blue", "red blue"):
        original, changed = model.predict(text), rotated.predict(text)
        assert changed.forward.vector == pytest.approx(np.array(original.forward.vector) @ rotation)
        assert changed.forward.range == original.forward.range
        assert changed.forward.cluster_distance == pytest.approx(original.forward.cluster_distance)
        assert changed.shift == pytest.approx(original.shift)
    assert _distance(
        model.project_context("red").vector, model.project_context("blue").vector
    ) == pytest.approx(
        _distance(rotated.project_context("red").vector, rotated.project_context("blue").vector)
    )


def test_kmeans_arithmetic_centers_objective_and_degenerate_counts():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    result = numeric.fit_kmeans(vectors, DualContextConfig(n_clusters=1), np)
    assert result.centers[0] == pytest.approx([2 / 3, 1 / 3])
    assert result.objective == pytest.approx(4 / 3)
    assert result.counts == (3,)
    assert result.converged and result.iterations == 2
    assert math.hypot(*result.centers[0]) < 1
    limited = numeric.fit_kmeans(
        vectors, DualContextConfig(n_clusters=1, max_kmeans_iterations=1), np
    )
    assert limited.iterations == 1 and not limited.converged
    degenerate = numeric.fit_kmeans(vectors, DualContextConfig(n_clusters=5, seed=42), np)
    assert degenerate.effective_clusters == 2
    assert degenerate.reason == "distinct_vectors_below_requested"
    assert degenerate.objective == pytest.approx(0)
    assert degenerate == numeric.fit_kmeans(
        vectors[::-1], DualContextConfig(n_clusters=5, seed=42), np
    )
    empty = numeric.fit_kmeans(np.zeros((0, 2)), DualContextConfig(), np)
    assert empty.reason == "no_nonzero_vectors" and empty.centers == () and empty.objective == 0


def test_input_permutation_and_frozen_heldout_do_not_change_state():
    records, forward, backward = training()
    first = trained(seed=17)
    second = DualContextModel(n_components=2, n_clusters=2, seed=17).fit(
        reversed(records), reversed(forward), reversed(backward)
    )
    assert first.to_dict() == second.to_dict()
    before = first.to_dict()
    for text in ("", "never-seen heldout", "ask NEW", "isolated", "RED red"):
        prediction = first.predict(text)
        assert first.transform(text) == prediction
        assert prediction.to_dict()["tokens"] == prediction.tokens
        assert first.project_context(text).to_dict()["known_tokens"] <= prediction.tokens
    assert first.to_dict() == before
    with pytest.raises(FrozenInstanceError):
        first.state.catalog_records = 999
    with pytest.raises(TypeError):
        first.state.basis[0][0] = 999
    exported = first.to_dict()
    exported["state"]["basis"][0][0] = 999
    assert first.to_dict() == before


def test_oov_empty_unsupported_and_partial_coverage_are_not_scores():
    model = trained()
    assert model.predict("").forward.reason == "empty_text"
    assert model.predict("unknown").forward.reason == "out_of_vocabulary"
    isolated = model.predict("isolated")
    assert (
        isolated.forward.vector
        is isolated.forward.range
        is isolated.orientation
        is isolated.shift
        is None
    )
    assert isolated.forward.reason == "no_direction_support"
    partial = model.predict("ask isolated")
    assert 0 < partial.forward.range_weight_coverage < 1
    assert partial.forward.support_weight_coverage == partial.forward.range_weight_coverage
    assert model.project_context("isolated").reason == "zero_projection"
    assert model.project_context("").reason == "empty_text"
    assert model.project_context("unknown").reason == "out_of_vocabulary"
    assert _unit([0.0, 0.0]) is None
    with pytest.raises(ValueError, match="finite"):
        _unit([math.inf])


def test_rank_one_drop_failure_and_zero_training_preserve_prior_state():
    model = trained(drop_first=True)
    before = model.to_dict()
    records = [ContextRecord("c", "s", "same"), ContextRecord("c", "t", "same")]
    edge = [ContextEdge("c", "s", "t")]
    with pytest.raises(ValueError, match="rank"):
        model.fit(records, edge, edge)
    assert model.to_dict() == before
    constant = DualContextModel(n_components=4).fit(records, edge, edge)
    assert constant.state.dimensions == constant.state.numerical_rank == 1
    with pytest.raises(ValueError, match="vocabulary"):
        model.fit([replace(row, text="") for row in records], edge, edge)
    assert model.to_dict() == before
    zero_context = [ContextRecord("c", "s", "term"), ContextRecord("c", "t", "")]
    with pytest.raises(ValueError, match="rank"):
        model.fit(zero_context, edge, edge)
    assert model.to_dict() == before


def test_unicode_casefold_is_frozen_even_when_not_retokenization_fixed_point():
    records = [
        ContextRecord("c", "s", "\u0130 STRASSE"),
        ContextRecord("c", "t", "\u0130 stra\u00dfe"),
    ]
    edge = [ContextEdge("c", "s", "t")]
    model = DualContextModel().fit(records, edge, edge)
    assert model.state.vocabulary == ("i\u0307", "strasse")
    loaded = DualContextModel.from_dict(model.to_dict())
    assert loaded.predict("\u0130 STRASSE") == model.predict("\u0130 STRASSE")


def test_greek_casefold_combining_marks_are_valid_training_output():
    records = [ContextRecord("c", "s", "\u0390"), ContextRecord("c", "t", "\u03b0")]
    edge = [ContextEdge("c", "s", "t")]
    model = DualContextModel().fit(records, edge, edge)
    assert model.state.vocabulary == ("\u03b9\u0308\u0301", "\u03c5\u0308\u0301")
    assert DualContextModel.from_dict(model.to_dict()).predict("\u0390") == model.predict("\u0390")


def test_tokenizer_pins_unicode_database_and_rejects_incompatible_runtime(monkeypatch):
    model = trained()
    value = model.to_dict()
    assert value["tokenizer"] == TOKENIZER_VERSION
    assert value["tokenizer"].endswith(f"/ucd-{unicodedata.unidata_version}")
    monkeypatch.setattr(artifact, "TOKENIZER_VERSION", "turnscope.regex-casefold.v1/ucd-0.0.0")
    with pytest.raises(ValueError, match="Unicode database"):
        DualContextModel.from_dict(value)


def test_zero_context_is_excluded_from_ranges_not_silently_treated_as_distance_one():
    records = [
        ContextRecord("c", "s", "ask"),
        ContextRecord("c", "t", "red"),
        ContextRecord("c", "empty", ""),
    ]
    forward = [ContextEdge("c", "s", "t"), ContextEdge("c", "s", "empty")]
    backward = [ContextEdge("c", "empty", "t")]
    model = DualContextModel().fit(records, forward, backward)
    ask = next(row for row in model.term_statistics() if row["term"] == "ask")
    assert ask["forward"]["support_edges"] == 2
    assert ask["forward"]["projectable_edges"] == 1
    assert ask["forward"]["range"] == pytest.approx(0)
    assert model.state.backward.clustering.reason == "no_nonzero_vectors"
    assert model.predict("ask").backward.vector is None
    assert model.project_context("red").backward_cluster_id is None
    assert DualContextModel.from_dict(model.to_dict()).to_dict() == model.to_dict()


def test_clipped_range_and_signed_projection_cancellation_are_explicit():
    records = [
        ContextRecord("c", "s", "red blue"),
        ContextRecord("c", "a", "red"),
        ContextRecord("c", "b", "blue"),
    ]
    # A balanced first component is removed; opposing retained contexts cancel.
    edges = [ContextEdge("c", "s", "a"), ContextEdge("c", "s", "b")]
    model = DualContextModel(n_components=1, drop_first=True).fit(records, edges, edges)
    # This equal-sigma example's SVD may retain an axis, not the balanced direction;
    # a direct numeric fixture independently exercises the cancellation geometry.
    x = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    u = np.array([[1.0], [-1.0]]) / math.sqrt(2)
    direction = numeric._direction(np, x, u, np.array([1.0]), [(0, 0), (0, 1)], DualContextConfig())
    assert direction.term_ranges == (None, None)
    assert direction.support_edges == direction.projectable_edges == (2, 2)
    assert direction.clustering.training_vectors == 0
    assert model.training_summary()["drop_first"] is True
    unequal = np.array([[1.0], [0.5]])
    clipped = numeric._direction(
        np, unequal, u, np.array([1.0]), [(0, 0), (1, 1)], DualContextConfig()
    )
    assert clipped.term_ranges == pytest.approx((0.5,))  # Unclipped distances 0 and 2.


@pytest.mark.parametrize(
    "change",
    [
        lambda state: replace(state, context_records=state.catalog_records + 1),
        lambda state: replace(state, forward_edges=1, backward_edges=1),
        lambda state: replace(state, vocabulary=list(state.vocabulary)),
        lambda state: replace(state, vocabulary_candidates=state.training_tokens + 1),
        lambda state: replace(state, vocabulary=("UPPER", *state.vocabulary[1:])),
        lambda state: replace(state, vocabulary=tuple(reversed(state.vocabulary))),
        lambda state: replace(state, column_norms=list(state.column_norms)),
        lambda state: replace(state, document_frequencies=state.document_frequencies[:-1]),
        lambda state: replace(state, training_tokens=1, vocabulary_candidates=1),
        lambda state: replace(state, numerical_rank=1),
        lambda state: replace(state, singular_values=(state.svd_cutoff, state.svd_cutoff)),
        lambda state: replace(state, singular_values=tuple(reversed(state.singular_values))),
        lambda state: replace(state, basis=state.basis[:-1]),
        lambda state: replace(state, basis=(state.basis[0][:-1], *state.basis[1:])),
        lambda state: replace(state, forward={}),
        lambda state: replace(
            state, forward=replace(state.forward, term_map=list(state.forward.term_map))
        ),
        lambda state: replace(
            state, forward=replace(state.forward, term_map=((), *state.forward.term_map[1:]))
        ),
        lambda state: replace(state, forward=replace(state.forward, clustering={})),
        lambda state: replace(
            state,
            forward=replace(
                state.forward,
                clustering=replace(
                    state.forward.clustering, centers=list(state.forward.clustering.centers)
                ),
            ),
        ),
        lambda state: replace(
            state,
            forward=replace(
                state.forward, clustering=replace(state.forward.clustering, counts=(1, 1))
            ),
        ),
        lambda state: replace(
            state,
            forward=replace(
                state.forward,
                clustering=replace(state.forward.clustering, centers=((0.9, 0.9), (0.9, 0.9))),
            ),
        ),
        lambda state: replace(
            state,
            forward=replace(
                state.forward,
                clustering=replace(state.forward.clustering, centers=((0.0,), (0.0,))),
            ),
        ),
        lambda state: replace(
            state,
            forward=replace(
                state.forward,
                clustering=replace(
                    state.forward.clustering, reason="distinct_vectors_below_requested"
                ),
            ),
        ),
        lambda state: replace(
            state,
            forward=replace(
                state.forward,
                clustering=replace(state.forward.clustering, reason="empty_clusters_removed"),
            ),
        ),
    ],
)
def test_direct_state_export_validates_shape_immutability_and_semantics(change):
    model = trained()
    with pytest.raises(ValueError):
        artifact.encode_model(change(model.state), model.config)


def test_direction_zero_support_and_range_presence_cannot_be_forged():
    model = trained()
    value = model.to_dict()
    supported = next(
        index for index, support in enumerate(value["state"]["forward"]["support_edges"]) if support
    )
    value["state"]["forward"]["support_edges"][supported] = 0
    value["state"]["forward"]["projectable_edges"][supported] = 0
    with pytest.raises(ValueError, match="unsupported"):
        DualContextModel.from_dict(resign(value))
    value = model.to_dict()
    value["state"]["forward"]["term_ranges"][supported] = None
    with pytest.raises(ValueError, match="availability"):
        DualContextModel.from_dict(resign(value))
    with pytest.raises(ValueError, match="invalid fitted"):
        artifact.validate_state({}, model.config)


def test_json_pending_siblings_are_counted_before_stack_allocation(monkeypatch):
    # Ten pending siblings plus another full-width level must not each get a fresh
    # node allowance. A list subclass observes whether traversal was admitted.
    class ForbiddenIteration(list):
        def __iter__(self):
            raise AssertionError("over-budget nested array traversal was admitted")

    monkeypatch.setattr(artifact, "MAX_JSON_NODES", 20)
    over = [0] * 10 + [ForbiddenIteration([0] * 10)]
    with pytest.raises(ValueError, match="structural"):
        artifact._bounded_graph(over)
    with pytest.raises(ValueError, match="fields"):
        artifact._bounded_graph({str(index): 0 for index in range(11)})
    with pytest.raises(ValueError, match="JSON values"):
        artifact._bounded_graph([object()])
    with monkeypatch.context() as context:
        context.setattr(artifact, "MAX_ARTIFACT_BYTES", 10)
        with pytest.raises(ValueError, match="byte limit"):
            artifact._bounded_graph(["123456", "123456"])


def test_empty_cluster_summary_is_strict():
    records = [ContextRecord("c", "a", ""), ContextRecord("c", "b", "term")]
    edges = [ContextEdge("c", "a", "b")]
    model = DualContextModel().fit(records, edges, edges)
    value = model.to_dict()
    value["state"]["forward"]["clustering"]["converged"] = False
    with pytest.raises(ValueError, match="empty clustering"):
        DualContextModel.from_dict(resign(value))


def test_symbolic_output_is_rejected_before_writing(tmp_path, monkeypatch):
    model = trained()
    path = tmp_path / "output.json"
    monkeypatch.setattr(type(path), "is_symlink", lambda path: True)
    with pytest.raises(ValueError, match="symbolic"):
        model.save(path)
    assert not path.exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_components": True},
        {"drop_first": 1},
        {"seed": -1},
        {"max_features": 1025},
        {"max_dense_cells": 128000001},
        {"n_clusters": 0},
        {"max_training_tokens": 10**400},
    ],
)
def test_configuration_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        DualContextModel(**kwargs)


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "duplicate",
        "missing",
        "edge_duplicate",
        "cycle",
        "edge_type",
        "record_type",
        "no_edges",
    ],
)
def test_training_graph_rejections_preserve_model(kind):
    model = trained()
    before = model.to_dict()
    records, edges, backwards = training()
    if kind == "empty":
        records = []
    elif kind == "duplicate":
        records.append(records[0])
    elif kind == "missing":
        edges.append(ContextEdge("other", "0", "1"))
    elif kind == "edge_duplicate":
        edges.append(edges[0])
    elif kind == "cycle":
        edges.append(ContextEdge("c", "3", "0"))
    elif kind == "edge_type":
        edges.append({})
    elif kind == "record_type":
        records.append({})
    else:
        edges = []
    with pytest.raises(ValueError):
        model.fit(records, edges, backwards)
    assert model.to_dict() == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_catalog_records": 1},
        {"max_edges_per_direction": 1},
        {"max_training_text_bytes": 1},
        {"max_training_tokens": 1},
        {"max_vocabulary_candidates": 1},
        {"max_dense_cells": 1},
        {"min_document_frequency": 999},
    ],
)
def test_training_quotas_fail_before_numeric_allocations(monkeypatch, kwargs):
    def forbidden(*args):
        raise AssertionError("numeric allocation must not occur")

    monkeypatch.setattr(numeric, "fit_numeric_dual", forbidden)
    with pytest.raises(ValueError):
        DualContextModel(**kwargs).fit(*training())


@pytest.mark.parametrize(
    "record", [("", "a", "ok"), ("c", "\x00", "ok"), ("c", "a", "\ud800"), (1, "a", "ok")]
)
def test_record_id_and_unicode_validation(record):
    with pytest.raises(ValueError):
        ContextRecord(*record)


def test_edges_and_unfitted_and_query_contracts():
    with pytest.raises(ValueError, match="self"):
        ContextEdge("c", "a", "a")
    with pytest.raises(ValueError):
        ContextEdge("", "a", "b")
    with pytest.raises(ValueError, match="not been fitted"):
        DualContextModel().predict("x")
    model = trained()
    for text in (1, "x" * (1024 * 1024 + 1), "\ud800"):
        with pytest.raises(ValueError):
            model.predict(text)
    assert ContextRecord("c", "a", "x").to_dict()["utterance_id"] == "a"
    assert ContextEdge("c", "a", "b").to_dict()["context_id"] == "b"


def test_numpy_unavailable_and_svd_failure_are_controlled_and_atomic(monkeypatch):
    model = trained()
    before = model.to_dict()

    def failure(*args, **kwargs):
        raise np.linalg.LinAlgError("test failure")

    with monkeypatch.context() as context:
        context.setattr(np.linalg, "svd", failure)
        with pytest.raises(ValueError, match="converge"):
            model.fit(*training())
    assert model.to_dict() == before

    def missing(*args):
        raise ImportError("numpy absent")

    monkeypatch.setattr(numeric.importlib, "import_module", missing)
    with pytest.raises(ValueError, match="optional"):
        model.fit(*training())
    assert model.to_dict() == before


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("format",), "future.v99"),
        (("tokenizer",), "other"),
        (("config", "drop_first"), 1),
        (("config", "n_clusters"), False),
        (("state", "catalog_records"), True),
        (("state", "context_records"), 99),
        (("state", "training_tokens"), 999999999),
        (("state", "training_text_bytes"), 1),
        (("state", "vocabulary_candidates"), 1),
        (("state", "numerical_rank"), 999),
        (("state", "estimated_dense_cells"), 1),
        (("state", "catalog_digest"), "Z" * 64),
        (("state", "vocabulary"), ["a", "a"]),
        (("state", "column_norms"), [1e-308] * 8),
        (("state", "singular_values"), [1e-308, 1e-308]),
        (("state", "document_frequencies"), [0] * 8),
        (("state", "basis", 0), [1.0]),
        (("state", "basis", 0, 0), 2.0),
        (("state", "forward", "term_map", 0, 0), 1e308),
        (("state", "forward", "term_ranges", 0), 2.0),
        (("state", "forward", "support_edges", 0), True),
        (("state", "forward", "projectable_edges", 0), 999),
        (("state", "forward", "clustering", "converged"), 1),
        (("state", "forward", "clustering", "counts"), [0, 0]),
        (("state", "forward", "clustering", "iterations"), 0),
        (("state", "forward", "clustering", "objective"), -1),
        (("state", "forward", "clustering", "reason"), "invented"),
        (("state", "forward", "clustering", "centers", 0, 0), 10.0),
    ],
)
def test_rehashed_malformed_artifacts_reject_semantics(path, replacement):
    value = trained().to_dict()
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(ValueError):
        DualContextModel.from_dict(resign(value))


def test_artifact_unknown_fields_nonfinite_huge_integer_shape_and_checksum():
    model = trained()
    for field in ("extra", "sha256"):
        value = model.to_dict()
        value[field] = "bad"
        with pytest.raises(ValueError):
            DualContextModel.from_dict(value)
    value = model.to_dict()
    value["state"]["basis"][0][0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        DualContextModel.from_dict(value)
    value["state"]["basis"][0][0] = 10**400
    with pytest.raises(ValueError, match="integer"):
        DualContextModel.from_dict(value)
    value = model.to_dict()
    value["config"]["unrecognized"] = 0
    with pytest.raises(ValueError, match="fields"):
        DualContextModel.from_dict(resign(value))
    value = model.to_dict()
    value["state"]["forward"]["clustering"]["centers"] = "bad"
    with pytest.raises(ValueError, match="shape"):
        DualContextModel.from_dict(resign(value))
    value = model.to_dict()
    value["state"]["basis"][0][0] = 0.123
    with pytest.raises(ValueError, match="orthonormal"):
        DualContextModel.from_dict(resign(value))


def test_export_bounds_full_envelope_and_failed_refit(monkeypatch):
    model = trained()
    original = model.to_dict()
    body = {key: value for key, value in original.items() if key != "sha256"}
    cap = len(artifact._canonical(body)) + 1
    with monkeypatch.context() as context:
        context.setattr(artifact, "MAX_ARTIFACT_BYTES", cap)
        with pytest.raises(ValueError, match="byte limit"):
            model.fit(*training())
        with pytest.raises(ValueError, match="byte limit"):
            model.to_dict()
    assert model.to_dict() == original
    with monkeypatch.context() as context:
        context.setattr(artifact, "MAX_JSON_NODES", 10)
        with pytest.raises(ValueError):
            DualContextModel.from_dict(original)
    deep = []
    for _ in range(40):
        deep = [deep]
    with pytest.raises(ValueError, match="structural"):
        DualContextModel.from_dict(deep)
    with pytest.raises(ValueError):
        DualContextModel.from_dict({1: object()})


def test_roundtrip_exclusive_save_and_explicit_overwrite(tmp_path):
    model = trained()
    path = tmp_path / "model.json"
    assert model.save(path) is None
    assert DualContextModel.load(path).to_dict() == model.to_dict()
    with pytest.raises(FileExistsError):
        model.save(path)
    assert model.save(path, overwrite=True) is None
    assert not list(tmp_path.glob(".dual-context-*"))
    with pytest.raises(ValueError):
        model.save(path, overwrite=1)
    with pytest.raises(ValueError):
        model.save(tmp_path, overwrite=True)
    alias = tmp_path / "alias.json"
    os.link(path, alias)
    with pytest.raises(ValueError, match="unaliased"):
        model.save(path, overwrite=True)


def test_publication_failure_and_after_publication_cleanup_warning(tmp_path, monkeypatch):
    model = trained()
    path = tmp_path / "model.json"

    def cannot_link(*args):
        raise OSError("injected link failure")

    with monkeypatch.context() as context:
        context.setattr(artifact.os, "link", cannot_link)
        with pytest.raises(OSError, match="link failure"):
            model.save(path)
    assert not path.exists() and not list(tmp_path.glob(".dual-context-*"))
    original_unlink = type(path).unlink

    def cannot_cleanup(self, **kwargs):
        if self.name.startswith(".dual-context-"):
            raise OSError("private test detail")
        return original_unlink(self, **kwargs)

    monkeypatch.setattr(type(path), "unlink", cannot_cleanup)
    assert model.save(path) == "model published; private temporary-file cleanup failed"
    assert DualContextModel.load(path).digest == model.digest
    assert len(list(tmp_path.glob(".dual-context-*"))) == 1


@pytest.mark.parametrize(
    "raw", [b"\xff", b'{"a":1,"a":2}', b'{"a":NaN}', b"[" * 100 + b"]" * 100, b"{"]
)
def test_file_json_encoding_duplicates_and_depth(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        DualContextModel.load(path)


def test_file_size_rejected_before_parser(tmp_path, monkeypatch):
    path = tmp_path / "large.json"
    path.write_bytes(b" " * 65)
    monkeypatch.setattr(artifact, "MAX_ARTIFACT_BYTES", 64)
    with pytest.raises(ValueError, match="byte limit"):
        DualContextModel.load(path)


def test_frozen_load_and_prediction_do_not_import_numpy(tmp_path):
    path = tmp_path / "model.json"
    trained().save(path)
    script = """
import importlib.abc, sys
class NoNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numpy' or fullname.startswith('numpy.'):
            raise AssertionError('frozen inference imported numpy')
sys.meta_path.insert(0, NoNumpy())
from turnscope.dual_context import DualContextModel
model = DualContextModel.load(sys.argv[1])
assert model.predict('ask').forward.vector is not None
assert model.project_context('red').vector is not None
assert model.term_statistics()
assert 'numpy' not in sys.modules
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(path)], capture_output=True, text=True, check=False
    )
    assert process.returncode == 0, process.stderr
