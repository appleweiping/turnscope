"""Independent data/denominator checks without training or importing the model."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest

_BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmarks/benchmark_dual_context.py"
_SPEC = importlib.util.spec_from_file_location(
    "_dual_context_benchmark_test_module", _BENCHMARK_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
# Dataclasses resolve postponed annotations through their defining module. This
# registers only the explicitly loaded file, without changing import search paths.
sys.modules[_SPEC.name] = bench
_SPEC.loader.exec_module(bench)


def source_fixture():
    metadata, records = {}, []
    for index, split in enumerate(("train", "val", "test")):
        for suffix in ("a", "b"):
            conversation = f"{split}-{suffix}"
            partner = f"{split}-{'b' if suffix == 'a' else 'a'}"
            metadata[conversation] = {"split": split, "page_id": index + 1, "pair_id": partner}
            for node, parent, header in (
                ("h", None, True),
                ("q", "h", False),
                ("r", "q", False),
                ("s", "q", False),
            ):
                records.append(
                    {
                        "id": f"{conversation}:{node}",
                        "conversation_id": conversation,
                        "text": f"Original café🏮 {node}\r\n{{literal}}",
                        "reply-to": f"{conversation}:{parent}" if parent else None,
                        "meta": {"is_section_header": header, "toxicity": 0.99, "parsed": "unused"},
                    }
                )
    return metadata, records


def data_fixture():
    return bench.audit_cga(*source_fixture())


def write_archive(tmp_path, metadata, records, *, raw=None):
    path = tmp_path / "authored.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr(bench._PREFIX + "conversations.json", json.dumps(metadata).encode())
        archive.writestr(
            bench._PREFIX + "utterances.jsonl",
            raw
            if raw is not None
            else b"\n".join(json.dumps(row, ensure_ascii=False).encode() for row in records)
            + b"\n",
        )
    return path


def test_edge_inventory_and_original_reply_pointers_are_preserved():
    data = data_fixture()
    assert dict(data.audit) == {
        "nodes": 24,
        "header_nodes": 6,
        "null_parent": 6,
        "nonnull_edges": 18,
        "legacy_nan_parent": 0,
        "dangling_edges": 0,
        "self_edges": 0,
        "cross_conversation_edges": 0,
        "cycle_edges": 0,
        "header_edges": 6,
        "accepted_edges": 12,
    }
    assert all(len(data.catalog(split)) == 6 for split in ("train", "val", "test"))
    forward = data.edges("train")
    assert len(forward) == 4
    assert data.edges("train", "backward") == tuple(sorted((b, a) for a, b in forward))
    query = data.nodes[("train-a", "train-a:q")]
    assert query.reply_to == "train-a:h"  # Excluded header edge is not rewritten to null.
    assert "toxicity" not in vars(query)
    assert data.fold_audit["train"]["accepted_edges"] == 4


def test_source_legacy_nan_parent_is_quarantined_not_null_and_incoming_edges_survive():
    metadata, records = source_fixture()
    records[1]["reply-to"] = math.nan
    encoded = [json.dumps(row).encode() for row in records]
    parsed = [bench._source_record_json(raw) for raw in encoded]
    data = bench.audit_cga(metadata, parsed)
    assert data.audit["legacy_nan_parent"] == 1
    assert data.fold_audit["train"]["legacy_nan_parent"] == 1
    assert data.audit["null_parent"] == 6
    assert data.audit["nonnull_edges"] == 18
    assert data.audit["header_edges"] == 5
    assert data.audit["accepted_edges"] == 12
    node = data.nodes[("train-a", "train-a:q")]
    assert node.reply_to is bench._LEGACY_NAN_PARENT
    assert node in data.catalog("train")
    assert (("train-a", "train-a:q"), ("train-a", "train-a:r")) in data.forward_edges
    before = data.input_digest
    parsed[1]["reply-to"] = None
    assert bench.audit_cga(metadata, parsed).input_digest != before
    with pytest.raises(ValueError, match="nonfinite"):
        bench._json(encoded[1])


@pytest.mark.parametrize(
    "raw",
    [
        b'{"reply-to":Infinity}',
        b'{"reply-to":-Infinity}',
        b'{"reply-to":1e999}',
        b'{"meta":{"toxicity":NaN}}',
        b'{"meta":{"reply-to":NaN}}',
        b'{"reply-to":[NaN]}',
        b'{"reply-to":{"value":NaN}}',
        b'{"page_id":NaN}',
        b'{"other":[1e999]}',
        b'{"reply-to":NaN,"text":NaN}',
    ],
)
def test_source_nonfinite_exception_never_extends_to_other_locations_or_values(raw):
    with pytest.raises(ValueError, match="nonfinite"):
        bench._source_record_json(raw)


def test_finite_nontext_parent_and_fake_nan_markers_are_not_accepted():
    metadata, records = source_fixture()
    for invalid in (0.1, {"legacy_nonfinite": "NaN"}, bench._LegacyNonfinite("NaN"), math.nan):
        records[1]["reply-to"] = invalid
        with pytest.raises(ValueError, match="reply-to"):
            bench.audit_cga(metadata, records)


def test_source_archive_reports_legacy_parent_and_retained_incoming_relations(
    tmp_path, monkeypatch
):
    metadata, records = source_fixture()
    records[1]["reply-to"] = math.nan
    archive = write_archive(tmp_path, metadata, records)
    monkeypatch.setattr(bench, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    _, report = bench.load_cga_archive(archive)
    source_policy = report["source_json_policy"]
    assert source_policy["strict_json"] is False
    assert source_policy["legacy_nonfinite_count"] == 1
    assert source_policy["legacy_parent_nodes_with_nonheader_text_retained"] == 1
    assert source_policy["accepted_edges_from_legacy_parent_nodes"] == 2


def test_null_preserves_both_degrees_text_and_shared_context_basis_inputs():
    data, protocol = data_fixture(), bench.BenchmarkProtocol()
    null = bench.shuffled_training(data, protocol)
    again = bench.shuffled_training(data, protocol)
    assert null == again
    assert null.edges is not None
    restored = tuple(
        sorted((tuple(json.loads(a[1])), tuple(json.loads(b[1]))) for a, b in null.edges)
    )
    assert Counter(a for a, _ in restored) == Counter(a for a, _ in data.edges("train"))
    assert Counter(b for _, b in restored) == Counter(b for _, b in data.edges("train"))
    assert all(a[0] == b[0] == "synthetic-null" for a, b in null.edges)
    assert all(a != b for a, b in null.edges)
    assert not bench._cycle_nodes({b: a for a, b in null.edges})
    assert {tuple(json.loads(key[1])): text for key, text in null.records.items()} == {
        node.key: node.text for node in data.catalog("train")
    }
    assert null.report["same_text_and_context_inventory_verified"] is True
    assert null.report["degrees_preserved"] is True
    assert 0 <= null.report["unchanged_edge_fraction"] <= 1
    assert null.report["uniform_over_all_mappings"] is False


def test_null_stops_after_declared_structural_attempts_without_seed_search(monkeypatch):
    data = data_fixture()
    monkeypatch.setattr(bench, "_cycle_nodes", lambda parents: {next(iter(parents))})
    null = bench.shuffled_training(data, bench.BenchmarkProtocol(max_shuffle_attempts=3))
    assert null.edges is None
    assert null.report["status"] == "unavailable"
    assert null.report["attempts"] == 3
    assert null.report["rejected_attempts"] == {"cycles": 3}
    assert null.report["unchanged_edge_fraction"] is None


def test_lexical_baseline_fits_each_train_utterance_once_and_never_heldout():
    nodes = (
        bench.CgaNode(("train", "a"), "red red green", None, False),
        bench.CgaNode(("train", "b"), "green blue", None, False),
    )
    baseline = bench.LexicalBaseline(nodes, bench.BenchmarkProtocol(min_document_frequency=1))
    assert baseline.training_records == 2
    assert baseline.idf["green"] == 1
    assert baseline.idf["red"] == baseline.idf["blue"] == pytest.approx(math.log(3 / 2) + 1)
    before = baseline.digest
    assert baseline.project("heldout_only") is None
    assert baseline.project("") is None
    assert bench.cosine(baseline.project("red"), baseline.project("blue")) == 0
    assert bench.cosine(baseline.project("red"), baseline.project("red red")) == 1
    assert baseline.digest == before


def test_training_mean_is_normalized_and_tracks_undefined_or_cancellation():
    assert bench.mean_projection([None, (1.0, 0), (0, 1.0)]) == pytest.approx(
        (1 / math.sqrt(2), 1 / math.sqrt(2))
    )
    assert bench.mean_projection([None]) is None
    assert bench.mean_projection([(1.0, 0), (-1.0, 0)]) is None
    assert bench.cosine(None, (1.0, 0)) is None
    for left, right in (((1.0,), (1.0, 0)), ((2.0, 0), (1.0, 0)), ((math.nan, 0), (1.0, 0))):
        with pytest.raises(ValueError):
            bench.cosine(left, right)


def test_vector_scorer_freezes_vectors_and_matches_independent_cosine_oracle():
    query, a, b = ("scope", "q"), ("scope", "a"), ("scope", "b")
    queries = {query: (1 / math.sqrt(2), 1 / math.sqrt(2))}
    candidates = {a: (1.0, 0), b: None}
    expected = bench.cosine(queries[query], candidates[a])
    scorer = bench._vector_scorer(queries, candidates)
    queries[query] = (0, -1.0)
    candidates[a] = (0, 1.0)
    assert scorer(query, (a, b)) == {a: expected, b: None}
    with pytest.raises(ValueError, match="dimensions"):
        bench._vector_scorer({query: (1.0,)}, {a: (1.0, 0)})


def test_small_synthetic_model_pipeline_uses_exact_frozen_tasks_and_aggregate_only():
    pytest.importorskip("numpy")
    result = bench.evaluate_models(
        data_fixture(),
        bench.BenchmarkProtocol(n_components=2, n_clusters=2, min_document_frequency=1),
    )
    assert result["all_frozen_parameters_unchanged"] is True
    assert set(result["retrieval"]) == {
        "dual_context",
        "shuffled_relation_null",
        "lexical_tfidf",
        "training_context_mean",
    }
    for model in result["fitted_models"].values():
        assert model["training"]["catalog_records"] == 6
        assert model["training"]["forward_edges"] == model["training"]["backward_edges"] == 4
        assert model["artifact_roundtrip_verified"] is True
    tasks = result["retrieval"]["dual_context"]
    assert [(task["split"], task["direction"], task["selected_queries"]) for task in tasks] == [
        ("val", "forward", 2),
        ("val", "backward", 4),
        ("test", "forward", 2),
        ("test", "backward", 4),
    ]
    assert [task["positives"] for task in tasks] == [4, 4, 4, 4]
    for reports in result["retrieval"].values():
        assert [task["task_sha256"] for task in reports] == [task["task_sha256"] for task in tasks]
    output = json.dumps(result, ensure_ascii=False)
    for private in ("Original café", "train-a:q", "val-a", "{{literal}}"):
        assert private not in output


def test_main_refuses_existing_output_and_source_hardlink_before_expensive_work(
    tmp_path, monkeypatch
):
    source, output = tmp_path / "source.zip", tmp_path / "existing.json"
    source.write_bytes(b"private source")
    os.link(source, output)
    monkeypatch.setattr(bench, "build_report", lambda *args: pytest.fail("must not evaluate"))
    assert bench.main([str(source), "--output", str(output)]) == 1
    assert source.read_bytes() == output.read_bytes() == b"private source"


def test_atomic_publication_collision_preserves_prior_output_and_cleans_own_temporary(
    tmp_path, monkeypatch
):
    output = tmp_path / "report.json"

    def race(_source, destination):
        destination.write_bytes(b"other writer")
        raise FileExistsError("PRIVATE ERROR")

    monkeypatch.setattr(bench.os, "link", race)
    with pytest.raises(FileExistsError):
        bench._publish_report(output, b"new data")
    assert output.read_bytes() == b"other writer"
    assert tuple(tmp_path.iterdir()) == (output,)


def test_closed_stdout_does_not_misreport_successfully_published_artifact(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    monkeypatch.setattr(bench, "build_report", lambda *args: {"synthetic": True})
    closed = io.StringIO()
    closed.close()
    diagnostic = io.StringIO()
    monkeypatch.setattr(bench.sys, "stdout", closed)
    monkeypatch.setattr(bench.sys, "stderr", diagnostic)
    assert bench.main([str(tmp_path / "archive"), "--output", str(output)]) == 1
    assert json.loads(output.read_bytes()) == {"synthetic": True}
    assert json.loads(diagnostic.getvalue()) == {
        "status": "completed",
        "report_published": True,
        "error_type": "ValueError",
    }


def test_report_generation_failure_is_redacted_and_publishes_no_output(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "report.json"

    def fail(*_args):
        raise RuntimeError("PRIVATE RAW TEXT AND TOKEN")

    monkeypatch.setattr(bench, "build_report", fail)
    assert bench.main([str(tmp_path / "archive"), "--output", str(output)]) == 1
    assert not output.exists()
    assert "PRIVATE" not in capsys.readouterr().err


def test_report_rejects_runtime_source_changes_and_does_not_claim_reproducibility(monkeypatch):
    hashes = iter(({"module": "before"}, {"module": "after"}))
    monkeypatch.setattr(bench, "_source_hashes", lambda: next(hashes))
    monkeypatch.setattr(bench, "load_cga_archive", lambda path: (data_fixture(), {}))
    monkeypatch.setattr(bench, "evaluate_models", lambda *args: {})
    with pytest.raises(RuntimeError, match="source changed"):
        bench.build_report(None, bench.BenchmarkProtocol())


def test_invalid_relation_reasons_are_disjoint_and_cycle_tails_not_removed():
    metadata, records = source_fixture()
    for node, parent in (
        ("self", "self"),
        ("dangling", "absent"),
        ("cycle1", "cycle2"),
        ("cycle2", "cycle1"),
        ("tail", "cycle1"),
    ):
        records.append(
            {
                "id": node,
                "conversation_id": "train-a",
                "text": node,
                "reply-to": parent,
                "meta": {"is_section_header": False},
            }
        )
    records.append(
        {
            "id": "cross",
            "conversation_id": "train-a",
            "text": "cross",
            "reply-to": "train-b:q",
            "meta": {"is_section_header": False},
        }
    )
    data = bench.audit_cga(metadata, records)
    assert (
        data.audit["self_edges"]
        == data.audit["dangling_edges"]
        == data.audit["cross_conversation_edges"]
        == 1
    )
    assert data.audit["cycle_edges"] == 2
    assert data.audit["accepted_edges"] == 13
    assert (("train-a", "cycle1"), ("train-a", "tail")) in data.forward_edges
    assert not any(target[1] in ("cycle1", "cycle2") for _, target in data.forward_edges)
    assert data.audit["nonnull_edges"] == 24


@pytest.mark.parametrize(
    "mutation",
    [
        "pair_split",
        "page_split",
        "nonreciprocal",
        "missing_pair",
        "self_pair",
        "bool_page",
        "float_page",
    ],
)
def test_split_or_group_corruption_is_a_hard_error(mutation):
    metadata, records = source_fixture()
    if mutation == "pair_split":
        metadata["train-a"]["split"] = "test"
    elif mutation == "page_split":
        metadata["test-a"]["page_id"] = 1
    elif mutation == "nonreciprocal":
        metadata["train-b"]["pair_id"] = "test-b"
    elif mutation == "missing_pair":
        metadata["train-a"]["pair_id"] = "missing"
    elif mutation == "self_pair":
        metadata["train-a"]["pair_id"] = "train-a"
    else:
        metadata["train-a"]["page_id"] = True if mutation == "bool_page" else 1.0
    with pytest.raises(ValueError):
        bench.audit_cga(metadata, records)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "unknown_conversation",
        "missing_reply",
        "blank_parent",
        "bad_header",
        "surrogate",
        "missing_observed_conversation",
    ],
)
def test_ambiguous_or_missing_source_inventory_is_rejected(mutation):
    metadata, records = source_fixture()
    if mutation == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif mutation == "unknown_conversation":
        records[0]["conversation_id"] = "missing"
    elif mutation == "missing_reply":
        records[0]["reply_to"] = records[0].pop("reply-to")
    elif mutation == "blank_parent":
        records[0]["reply-to"] = " "
    elif mutation == "bad_header":
        records[0]["meta"]["is_section_header"] = 1
    elif mutation == "surrogate":
        records[0]["text"] = "\ud800"
    else:
        records = [row for row in records if row["conversation_id"] != "train-a"]
    with pytest.raises(ValueError):
        bench.audit_cga(metadata, records)


def test_labels_parses_timestamps_and_order_cannot_change_model_input():
    metadata, records = source_fixture()
    before = bench.audit_cga(metadata, records)
    for item in metadata.values():
        item["conversation_has_personal_attack"] = True
        item["page_title"] = "not a feature"
    for row in records:
        row["meta"].update(
            toxicity=-123.0, parsed={"must": "not enter model"}, comment_has_personal_attack=True
        )
        row["timestamp"] = -1
        row["speaker"] = "ignored identity"
    after = bench.audit_cga(metadata, reversed(records))
    assert before.input_digest == after.input_digest
    assert before.forward_edges == after.forward_edges
    records[1]["text"] += " changed source"
    assert before.input_digest != bench.audit_cga(metadata, records).input_digest


def test_loaded_jsonl_uses_actual_newlines_not_unicode_line_separators(tmp_path, monkeypatch):
    metadata, records = source_fixture()
    records[1]["text"] = "alpha\u2028beta\u0085gamma\u2029🏮"
    path = write_archive(tmp_path, metadata, records)
    monkeypatch.setattr(bench, "ARCHIVE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    data, report = bench.load_cga_archive(path)
    assert data.nodes[("train-a", "train-a:q")].text == records[1]["text"]
    assert report["audit"]["nodes"] == 24
    assert report["conversation_counts"] == {"train": 2, "val": 2, "test": 2}
    published = json.dumps(report)
    assert "alpha" not in published and "toxicity" not in published and "train-a:q" not in published


def test_archive_digest_checked_before_parsing(tmp_path, monkeypatch):
    path = write_archive(tmp_path, *source_fixture())
    monkeypatch.setattr(
        bench, "ZipFile", lambda *_args: pytest.fail("must not parse unpinned data")
    )
    with pytest.raises(ValueError, match="SHA-256"):
        bench.load_cga_archive(path)


@pytest.mark.parametrize("raw", [b'{"id":1,"id":2}\n', b'{"value":NaN}\n', b"\xff\n", b"\n"])
def test_strict_json_and_utf8_in_local_source(tmp_path, monkeypatch, raw):
    path = write_archive(tmp_path, *source_fixture(), raw=raw)
    monkeypatch.setattr(bench, "ARCHIVE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises((ValueError, UnicodeError)):
        bench.load_cga_archive(path)


@pytest.mark.parametrize(
    "bound", ["_MAX_ROW_BYTES", "_MAX_METADATA_BYTES", "_MAX_UTTERANCE_BYTES", "_MAX_ARCHIVE_BYTES"]
)
def test_archive_and_expanded_member_limits(tmp_path, monkeypatch, bound):
    path = write_archive(tmp_path, *source_fixture())
    monkeypatch.setattr(bench, "ARCHIVE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(bench, bound, 1)
    with pytest.raises(ValueError, match="bound"):
        bench.load_cga_archive(path)


def test_rank_oracle_multiple_positives_ties_and_undefined_contexts():
    a, b, c, d = (("c", value) for value in ("a", "b", "c", "d"))
    result = bench.ranking_metrics({d: None, c: 0.8, b: 0.8, a: 0.9}, (b, d), (1, 2, 4))
    assert result["reciprocal_rank"] == 0.5
    assert result["recall"] == {"1": 0, "2": 0.5, "4": 0.5}
    assert result["candidates"] == 4 and result["scored_candidates"] == 3
    assert result["positives"] == 2 and result["scored_positives"] == 1


@pytest.mark.parametrize("score", [True, math.nan, math.inf, 1.01, -1.01, 10**500, "0.3"])
def test_rank_rejects_invalid_numeric_scores(score):
    key = ("c", "a")
    with pytest.raises(ValueError):
        bench.ranking_metrics({key: score}, (key,), (1,))


@pytest.mark.parametrize(
    "positives,ks",
    [
        ((), (1,)),
        ((("c", "a"), ("c", "a")), (1,)),
        ((("c", "missing"),), (1,)),
        ((("c", "a"),), (True,)),
        ((("c", "a"),), (2, 1)),
    ],
)
def test_rank_rejects_ambiguous_denominators_or_cutoffs(positives, ks):
    with pytest.raises(ValueError):
        bench.ranking_metrics({("c", "a"): 0.0}, positives, ks)


def test_fixed_task_contains_every_positive_and_excludes_headers():
    data = data_fixture()
    config = bench.BenchmarkProtocol(max_queries=1, max_candidates=3)
    task = bench.make_retrieval_task(data, "test", "forward", config)
    assert len(task.queries) == 1 and task.eligible_queries == 2
    assert len(task.candidates) == 3 and task.available_candidates == 6
    assert set(task.queries[0].positives) <= set(task.candidates)
    assert len(task.queries[0].positives) == 2
    assert not any(key[1].endswith(":h") for key in task.candidates)
    assert task == bench.make_retrieval_task(data, "test", "forward", config)
    with pytest.raises(ValueError, match="no positives truncated"):
        bench.make_retrieval_task(data, "test", "forward", replace(config, max_candidates=1))


def test_task_inventory_does_not_use_text_to_pick_queries_or_candidates():
    metadata, records = source_fixture()
    config = bench.BenchmarkProtocol(max_queries=1, max_candidates=3)
    before = bench.make_retrieval_task(
        bench.audit_cga(metadata, records), "test", "forward", config
    )
    for row in records:
        row["text"] = "changed " + row["id"]
    after = bench.make_retrieval_task(bench.audit_cga(metadata, records), "test", "forward", config)
    assert before == after


def test_undefined_queries_are_zero_credit_not_silent_success_or_dropped_denominator():
    data = data_fixture()
    config = bench.BenchmarkProtocol()
    task = bench.make_retrieval_task(data, "test", "forward", config)
    first = task.queries[0].key
    true_targets = set(task.queries[0].positives)
    observed = []

    def score(key, candidates):
        observed.append((key, candidates))
        assert key not in candidates
        if key != first:
            return None
        return {target: 1.0 if target in true_targets else 0.0 for target in candidates}

    result = bench.evaluate_retrieval(data, task, score, config)
    assert len(observed) == 2
    assert result["selected_queries"] == 2 and result["scored_queries"] == 1
    assert result["query_coverage"] == result["zero_credit_mrr"] == 0.5
    assert result["scorable_mrr"] == 1
    assert result["zero_credit_recall"] == {"1": 0.25, "5": 0.5, "10": 0.5}
    assert result["page_macro_zero_credit_mrr"] == result["pair_macro_zero_credit_mrr"] == 0.5
    assert result["page_groups"] == result["pair_groups"] == 1
    assert "test-a" not in json.dumps(result)


def test_forged_tasks_rejected_before_scorer_and_missing_score_keys_rejected():
    data = data_fixture()
    config = bench.BenchmarkProtocol()
    task = bench.make_retrieval_task(data, "test", "forward", config)
    with pytest.raises(ValueError, match="frozen"):
        bench.evaluate_retrieval(
            data,
            replace(task, queries=task.queries[:1]),
            lambda *_args: pytest.fail("not admitted"),
            config,
        )
    with pytest.raises(ValueError, match="exact fixed candidate"):
        bench.evaluate_retrieval(data, task, lambda *_args: {}, config)


def test_group_macro_weights_groups_not_query_count():
    metadata, records = source_fixture()
    # Add a second test matched pair on its own page with one edge per conversation.
    for suffix in ("a", "b"):
        conversation = f"extra-{suffix}"
        metadata[conversation] = {
            "split": "test",
            "page_id": 99,
            "pair_id": f"extra-{'b' if suffix == 'a' else 'a'}",
        }
        for node, parent in (("q", None), ("r", "q")):
            records.append(
                {
                    "id": f"{conversation}:{node}",
                    "conversation_id": conversation,
                    "text": "original",
                    "reply-to": f"{conversation}:{parent}" if parent else None,
                    "meta": {"is_section_header": False},
                }
            )
    data = bench.audit_cga(metadata, records)
    config = bench.BenchmarkProtocol()
    task = bench.make_retrieval_task(data, "test", "backward", config)
    positives = {query.key: query.positives for query in task.queries}

    def score(key, candidates):
        if key[0].startswith("test-"):
            return None
        return {candidate: 1.0 if candidate in positives[key] else 0.0 for candidate in candidates}

    result = bench.evaluate_retrieval(data, task, score, config)
    assert result["selected_queries"] == 6
    assert result["zero_credit_mrr"] == pytest.approx(1 / 3)
    assert result["page_macro_zero_credit_mrr"] == result["pair_macro_zero_credit_mrr"] == 0.5
    assert result["page_macro_zero_credit_recall"]["1"] == 0.5
