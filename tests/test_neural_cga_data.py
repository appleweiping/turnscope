"""Independent, hand-authored source shapes; no corpus download or training."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from datetime import timezone
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from turnscope.neural_forecast_data import prepare_sequence_forecasts
from turnscope.neural_token_data import fit_sequence_vocabulary

_SOURCE = Path(__file__).resolve().parents[1] / "benchmarks/neural_cga_data.py"
_SPEC = importlib.util.spec_from_file_location("turnscope_neural_cga_data_test", _SOURCE)
assert _SPEC and _SPEC.loader
bench = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bench
_SPEC.loader.exec_module(bench)


def native_fixture():
    metadata = {}
    for a, b, split, page_a, page_b in (
        ("ta", "tb", "train", 1, 1),
        ("xa", "xb", "test", 2, 2),
        ("va", "vb", "val", 10, 20),
        ("vc", "vd", "val", 20, 30),
        ("ve", "vf", "val", 40, 40),
        ("vg", "vh", "val", 50, 50),
    ):
        for identifier, partner, page, label in ((a, b, page_a, True), (b, a, page_b, False)):
            metadata[identifier] = {
                "page_title": "PRIVATE_PAGE_TITLE",
                "page_id": page,
                "pair_id": partner,
                "conversation_has_personal_attack": label,
                "verified": True,
                "pair_verified": True,
                "annotation_year": "2018",
                "split": split,
            }
    records = []
    for identifier, meta in metadata.items():
        # A later-excluded conversation still bridges validation page/pair groups.
        total = 1 if identifier == "vb" else 4
        for index in range(total):
            event = meta["conversation_has_personal_attack"] and index == total - 1
            records.append(
                {
                    "id": f"{identifier}-{index}",
                    "conversation_id": identifier,
                    "text": (
                        "EVENT_SECRET"
                        if event
                        else "CENSOR_SECRET"
                        if index == total - 1
                        else "alpha beta"
                        if meta["split"] == "train"
                        else "HELDOUT_SECRET"
                    ),
                    "speaker": "PRIVATE_PERSON",
                    "reply-to": None if index == 0 else f"{identifier}-{index - 1}",
                    "timestamp": float(10 + index),
                    "meta": {
                        "is_section_header": False,
                        "comment_has_personal_attack": event,
                        "parsed": [{"ignored": "PARSE_SECRET"}],
                        "toxicity": 0.75,
                    },
                }
            )
    header = copy.deepcopy(records[0])
    header.update(id="header", text="HEADER_SECRET", timestamp=9.0)
    header["meta"].update(is_section_header=True, comment_has_personal_attack=True)
    records.append(header)
    return metadata, records


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def archive(tmp_path, monkeypatch, metadata=None, records=None, *, raw_rows=None, ignored=b"x"):
    if metadata is None:
        metadata, records = native_fixture()
    metadata_raw = encode(metadata)
    utterances = (
        raw_rows if raw_rows is not None else b"".join(encode(row) + b"\n" for row in records)
    )
    path = tmp_path / "authored.zip"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as value:
        value.writestr(bench.PREFIX + "conversations.json", metadata_raw)
        value.writestr(bench.PREFIX + "utterances.jsonl", utterances)
        value.writestr(bench.PREFIX + "info.parsed.jsonl", ignored)
    monkeypatch.setattr(bench, "ARCHIVE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return path, metadata_raw, utterances


def memberships(result):
    return {
        conversation.id: name for name in bench._OUTPUTS for conversation in getattr(result, name)
    }


def test_exact_component_formula_and_excluded_bridge_remain_in_hash_input():
    metadata, records = native_fixture()
    result = bench.partition_neural_cga(metadata, records)
    assignment = memberships(result)
    expected_components = (("va", "vb", "vc", "vd"), ("ve", "vf"), ("vg", "vh"))
    for ids in expected_components:
        canonical = json.dumps(
            {"seed": "turnscope-neural-v1/2026-09-12", "conversation_ids": list(ids)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        bucket = int(hashlib.sha256(canonical).hexdigest(), 16) % 5
        expected = "policy_validation" if bucket in (0, 1) else "validation"
        assert {assignment[key] for key in ids} == {expected}
    assert assignment["ta"] == assignment["tb"] == "training"
    assert assignment["xa"] == assignment["xb"] == "test"
    prepared = result.prepared_audit()
    assert sum(value["audit"]["excluded_conversations"] for value in prepared.values()) == 1
    assert "vb" in assignment
    assert result.audit["validation_components"]["components"] == 3
    assert result.audit["validation_components"]["largest_component_conversations"] == 4
    assert result.audit["cross_partition_page_pair_groups"] == 0


def test_labels_and_input_iteration_order_cannot_choose_partitions():
    metadata, records = native_fixture()
    before_meta, before_rows = copy.deepcopy(metadata), copy.deepcopy(records)
    first = bench.partition_neural_cga(metadata, records)
    second = bench.partition_neural_cga(dict(reversed(list(metadata.items()))), reversed(records))
    assert first.audit == second.audit
    assert memberships(first) == memberships(second)
    assert metadata == before_meta and records == before_rows
    for value in metadata.values():
        value["conversation_has_personal_attack"] = False
    for row in records:
        row["meta"]["comment_has_personal_attack"] = False
    third = bench.partition_neural_cga(metadata, records)
    assert memberships(third) == memberships(first)
    assert third.audit["assignment_sha256"] == first.audit["assignment_sha256"]
    assert third.audit["partition_label_counts"] != first.audit["partition_label_counts"]


def test_exact_utc_time_id_order_and_text_only_causal_vocabulary():
    metadata, records = native_fixture()
    result = bench.partition_neural_cga(metadata, reversed(records))
    train = prepare_sequence_forecasts(result.training)
    vocabulary = fit_sequence_vocabulary(train, max_features=100)
    assert vocabulary.tokens == ("alpha", "beta")
    for conversation in result.training:
        assert conversation.utterances == tuple(
            sorted(conversation.utterances, key=lambda turn: (turn.timestamp, turn.id))
        )
        assert all(turn.timestamp.tzinfo is timezone.utc for turn in conversation.utterances)
        assert all(
            turn.role == "speaker" and turn.reply_to is None for turn in conversation.utterances
        )
        assert set(conversation.metadata) == {"forecast_groups"}
        assert all(
            set(turn.metadata) == {"event", "is_section_header"} for turn in conversation.utterances
        )
    raw = json.dumps(result.audit)
    for marker in ("PRIVATE", "SECRET", '"ta"', '"vb"'):
        assert marker not in raw
    result.audit["raw_conversations"] = 0
    assert result.audit["raw_conversations"] == len(metadata)


def test_equal_timestamp_tie_blocks_not_split_and_id_breaks_storage_ties():
    metadata, records = native_fixture()
    for row in records:
        if row["id"] in ("ta-0", "ta-1"):
            row["timestamp"] = 10.5
    result = bench.partition_neural_cga(metadata, reversed(records))
    conversation = next(item for item in result.training if item.id == "ta")
    assert [turn.id for turn in conversation.utterances] == [
        "header",
        "ta-0",
        "ta-1",
        "ta-2",
        "ta-3",
    ]
    data = prepare_sequence_forecasts((conversation,))
    assert [example.endpoint for example in data.examples] == [1, 2]


@pytest.mark.parametrize("parent", ["missing-parent", "ta-0", "xa-0", "ta-1"])
def test_irrelevant_invalid_reply_relationships_do_not_drop_or_relabel(parent):
    metadata, records = native_fixture()
    before = bench.partition_neural_cga(metadata, records)
    records[0]["reply-to"] = parent
    after = bench.partition_neural_cga(metadata, records)
    assert (
        after.audit["normalized_partition_input_sha256"]
        == before.audit["normalized_partition_input_sha256"]
    )
    assert after.audit["raw_utterances"] == len(records)


def test_legacy_nan_only_original_top_level_reply_to_without_dropping_text(tmp_path, monkeypatch):
    metadata, records = native_fixture()
    records[0]["reply-to"] = float("nan")
    path, _, _ = archive(tmp_path, monkeypatch, metadata, records)
    result = bench.load_neural_cga(path)
    assert result.audit["legacy_reply_to_nan"] == 1
    assert result.audit["raw_utterances"] == len(records)
    assert result.training[0].utterances[1].text == "alpha beta"
    assert not result.audit["reply_edge_filtering"]


@pytest.mark.parametrize(
    "raw",
    [
        b'{"reply-to":Infinity}',
        b'{"reply-to":-Infinity}',
        b'{"timestamp":NaN}',
        b'{"meta":{"reply-to":NaN}}',
        b'{"reply-to":[NaN]}',
        b'{"meta":{"toxicity":NaN}}',
        b'{"reply-to":1e999}',
        b'{"id":"x","id":"y"}',
        b'{"meta":{"x":1,"x":2}}',
        b'{"text":"\\ud800"}',
        b'{"timestamp":9223372036854775808}',
        b"\xff",
    ],
)
def test_source_json_rejects_other_nonfinite_duplicate_or_unrepresentable_values(raw):
    with pytest.raises(ValueError):
        bench._json(raw, bench.NeuralCgaLimits())


def test_metadata_has_no_legacy_nan_exception():
    with pytest.raises(ValueError, match="NaN"):
        bench._json(b'{"reply-to":NaN}', bench.NeuralCgaLimits(), metadata=True)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m, r: m["ta"].update(pair_id="ta"),
        lambda m, r: m["ta"].update(pair_id="missing"),
        lambda m, r: m["tb"].update(pair_id="va"),
        lambda m, r: m["tb"].update(split="test"),
        lambda m, r: m["xa"].update(page_id=1),
        lambda m, r: m["ta"].update(page_id=True),
        lambda m, r: m["ta"].update(page_id=-1),
        lambda m, r: m["ta"].update(split="validation"),
        lambda m, r: m["ta"].update(conversation_has_personal_attack=1),
        lambda m, r: m["ta"].update(conversation_has_personal_attack=False),
        lambda m, r: m["ta"].update(verified=1),
        lambda m, r: m["ta"].update(unknown=True),
        lambda m, r: r[0].update(unknown=True),
        lambda m, r: r[0].update(conversation_id="unknown"),
        lambda m, r: r[1].update(id=r[0]["id"]),
        lambda m, r: r[0]["meta"].update(is_section_header=1),
        lambda m, r: r[0]["meta"].update(comment_has_personal_attack=1),
        lambda m, r: r[0]["meta"].update(parsed={}),
        lambda m, r: r[0]["meta"].update(toxicity=True),
        lambda m, r: r[0]["meta"].update(unknown=False),
        lambda m, r: r[0].update(timestamp=True),
        lambda m, r: r[0].update(timestamp=math.inf),
        lambda m, r: r[0].update(timestamp=10**400),
        lambda m, r: r[0].update(timestamp=253402300800),
        lambda m, r: r[0].update(text=1),
        lambda m, r: r[0].update(**{"reply-to": 123}),
        lambda m, r: r[0].update(speaker=None),
        lambda m, r: r.clear(),
    ],
)
def test_native_source_schema_and_label_failures_are_not_silently_repaired(mutate):
    metadata, records = native_fixture()
    mutate(metadata, records)
    with pytest.raises(ValueError):
        bench.partition_neural_cga(metadata, records)


def test_fields_nodes_depth_and_bytes_rejected_before_decoder(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("decoder ran before admission"))
    monkeypatch.setattr(bench.json, "loads", forbidden)
    for raw, limits in (
        (b'{"a":0,"b":1}', bench.NeuralCgaLimits(max_object_fields=1)),
        (b"[0,0,0]", bench.NeuralCgaLimits(max_json_nodes=3)),
        (b"[[[0]]]", bench.NeuralCgaLimits(max_json_depth=2)),
        (b'"12345"', bench.NeuralCgaLimits(max_row_bytes=5)),
    ):
        with pytest.raises(ValueError):
            bench._json(raw, limits)
    forbidden.assert_not_called()


def test_native_queued_nodes_and_full_encoded_bytes_are_bounded():
    for value, limits in (
        ([[0, 0], [0, 0]], bench.NeuralCgaLimits(max_json_nodes=5)),
        ({"a": "\0" * 5}, bench.NeuralCgaLimits(max_row_bytes=20)),
        ({"a": ["é" * 5, "é" * 5]}, bench.NeuralCgaLimits(max_row_bytes=20)),
        ({1: None}, bench.NeuralCgaLimits()),
        ({"a": object()}, bench.NeuralCgaLimits()),
    ):
        with pytest.raises(ValueError):
            bench._tree(value, limits, metadata=False)


def test_native_inert_nonfinite_still_checked_and_metadata_is_snapshotted():
    metadata, records = native_fixture()
    expected = bench.partition_neural_cga(metadata, records).audit

    def rows():
        metadata["ta"]["conversation_has_personal_attack"] = False
        metadata["ta"]["split"] = "test"
        yield from records

    assert bench.partition_neural_cga(metadata, rows()).audit == expected
    metadata, records = native_fixture()
    records[0]["meta"]["parsed"] = [{"inert": float("nan")}]
    with pytest.raises(ValueError, match="nonfinite"):
        bench.partition_neural_cga(metadata, records)


def test_file_loader_validates_inert_tree_once_not_never_or_twice(tmp_path, monkeypatch):
    path, _, _ = archive(tmp_path, monkeypatch)
    original = bench._tree
    counts = {"metadata": 0, "row": 0}

    def tracked(value, limits, *, metadata):
        counts["metadata" if metadata else "row"] += 1
        return original(value, limits, metadata=metadata)

    monkeypatch.setattr(bench, "_tree", tracked)
    result = bench.load_neural_cga(path)
    assert counts == {"metadata": 1, "row": result.audit["raw_utterances"]}


@pytest.mark.parametrize(
    "limits",
    [
        bench.NeuralCgaLimits(max_conversations=1),
        bench.NeuralCgaLimits(max_utterances=1),
        bench.NeuralCgaLimits(max_text_bytes=2),
        bench.NeuralCgaLimits(max_total_text_bytes=20),
        bench.NeuralCgaLimits(max_identifier_bytes=2),
    ],
)
def test_inventory_and_feature_budgets(limits):
    with pytest.raises(ValueError):
        bench.partition_neural_cga(*native_fixture(), limits=limits)


def test_archive_reads_only_required_members_not_unrelated_aggregate(tmp_path, monkeypatch):
    path, meta_raw, rows_raw = archive(tmp_path, monkeypatch, ignored=b"IGNORED" * 10000)
    opened = []
    original = ZipFile.open

    def tracked(self, name, *args, **kwargs):
        opened.append(name.filename if hasattr(name, "filename") else name)
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "open", tracked)
    result = bench.load_neural_cga(
        path, limits=bench.NeuralCgaLimits(max_utterance_bytes=len(rows_raw))
    )
    assert opened == [bench.PREFIX + "conversations.json", bench.PREFIX + "utterances.jsonl"]
    assert result.audit["metadata_sha256"] == hashlib.sha256(meta_raw).hexdigest()
    assert result.audit["utterances_sha256"] == hashlib.sha256(rows_raw).hexdigest()
    assert result.audit["read_members"] == 2
    assert result.audit["ignored_members"] == 1
    assert result.audit["source_archive_verified"] is True


def test_wrong_pin_and_archive_directory_budget_fail_before_zip_construction(tmp_path, monkeypatch):
    path, _, _ = archive(tmp_path, monkeypatch)
    forbidden = Mock(side_effect=AssertionError("ZipFile constructed before admission"))
    monkeypatch.setattr(bench, "ZipFile", forbidden)
    with pytest.raises(ValueError, match="member/directory"):
        bench.load_neural_cga(path, limits=bench.NeuralCgaLimits(max_members=1))
    with pytest.raises(ValueError, match="member/directory"):
        bench.load_neural_cga(path, limits=bench.NeuralCgaLimits(max_directory_bytes=1))
    monkeypatch.setattr(bench, "ARCHIVE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        bench.load_neural_cga(path)
    forbidden.assert_not_called()


def test_declared_member_and_actual_row_budgets(tmp_path, monkeypatch):
    path, meta_raw, rows_raw = archive(tmp_path, monkeypatch)
    for limits in (
        bench.NeuralCgaLimits(max_archive_bytes=1),
        bench.NeuralCgaLimits(max_metadata_bytes=len(meta_raw) - 1),
        bench.NeuralCgaLimits(max_utterance_bytes=len(rows_raw) - 1),
        bench.NeuralCgaLimits(max_row_bytes=20),
    ):
        with pytest.raises(ValueError):
            bench.load_neural_cga(path, limits=limits)


def test_duplicate_member_and_malformed_end_record(tmp_path, monkeypatch):
    path, meta_raw, _ = archive(tmp_path, monkeypatch)
    with ZipFile(path, "a") as value, pytest.warns(UserWarning):
        value.writestr(bench.PREFIX + "conversations.json", meta_raw)
    monkeypatch.setattr(bench, "ARCHIVE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="duplicate"):
        bench.load_neural_cga(path)
    data = path.read_bytes() + b"TRAILER"
    path.write_bytes(data)
    monkeypatch.setattr(bench, "ARCHIVE_SHA256", hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError, match="ZIP framing"):
        bench.load_neural_cga(path)


def test_required_member_incomplete_inventory_and_blank_row_rejected(tmp_path, monkeypatch):
    metadata, records = native_fixture()
    for raw in (
        b"\n",
        b"".join(encode(row) + b"\n" for row in records if row["conversation_id"] != "ta"),
    ):
        path, _, _ = archive(tmp_path, monkeypatch, metadata, records, raw_rows=raw)
        with pytest.raises(ValueError):
            bench.load_neural_cga(path)


def test_limits_closed_positive_and_no_accidental_boolean_counts():
    for changes in (
        {"max_members": True},
        {"max_archive_bytes": 129 * 1024 * 1024},
        {"max_json_nodes": 0},
        {"max_row_bytes": 4 * 1024 * 1024 + 1},
    ):
        with pytest.raises(ValueError):
            bench.NeuralCgaLimits(**changes)
    with pytest.raises(ValueError):
        bench.partition_neural_cga(*native_fixture(), limits={})


def test_source_is_readonly_and_no_corpus_write_or_training_dependencies(tmp_path, monkeypatch):
    path, _, _ = archive(tmp_path, monkeypatch)
    before = path.read_bytes()
    before_files = sorted(tmp_path.iterdir())
    torch_already_imported = "torch" in sys.modules
    result = bench.load_neural_cga(path)
    result.prepared_audit()
    assert path.read_bytes() == before
    assert sorted(tmp_path.iterdir()) == before_files
    assert ("torch" in sys.modules) == torch_already_imported


def test_static_protocol_hash_and_metadata_only_partition_identity():
    first = bench.protocol()
    expected = hashlib.sha256(
        json.dumps(first, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert bench.protocol_digest() == expected
    assert first["seed"] == "turnscope-neural-v1/2026-09-12"
    assert first["policy_validation_buckets"] == [0, 1]
    assert first["model_validation_buckets"] == [2, 3, 4]
    first["seed"] = "changed"
    assert bench.protocol()["seed"] == "turnscope-neural-v1/2026-09-12"
