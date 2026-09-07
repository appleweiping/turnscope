from datetime import datetime, timezone

import pytest

from turnscope import Conversation, ConversationSearchIndex, Utterance


def conversation(identifier: str, rows: list[tuple[str, str]]) -> Conversation:
    return Conversation(
        identifier,
        [
            Utterance(row_id, "user", text, datetime(2024, 1, 1, tzinfo=timezone.utc))
            for row_id, text in rows
        ],
    )


def test_query_is_ranked_and_reports_terms() -> None:
    index = ConversationSearchIndex(
        [
            conversation("c1", [("u1", "power grid stability")]),
            conversation("c2", [("u2", "power grid power")]),
        ]
    )
    hits = index.query("power stability")
    assert hits[0].utterance_id == "u1"
    assert hits[0].matched_terms == ("power", "stability")
    assert hits[0].score > hits[1].score


def test_replace_and_remove_recompute_statistics() -> None:
    index = ConversationSearchIndex([conversation("c1", [("u1", "alpha beta")])])
    assert index.documents == 1 and index.average_length == 2
    index.add(conversation("c1", [("u2", "gamma")]))
    assert index.documents == 1 and index.query("alpha") == ()
    assert index.query("gamma")[0].utterance_id == "u2"
    assert index.remove("c1") and not index.remove("c1") and index.documents == 0


def test_filters_and_validation() -> None:
    index = ConversationSearchIndex([conversation("c1", [("u1", "alpha")])])
    assert index.query("alpha", conversation_id="missing") == ()
    assert index.query("unknown") == ()
    with pytest.raises(ValueError, match="limit"):
        index.query("alpha", limit=0)
    with pytest.raises(ValueError, match="b"):
        index.query("alpha", b=2)
    with pytest.raises(TypeError, match="query"):
        index.query(1)  # type: ignore[arg-type]


def test_snapshot_round_trip_is_deterministic(tmp_path) -> None:
    index = ConversationSearchIndex(
        [conversation("c1", [("u1", "alpha beta")]), conversation("c2", [("u2", "beta")])]
    )
    first = tmp_path / "one.json"
    second = tmp_path / "nested" / "two.json"
    digest = index.save(first)
    assert len(digest) == 64
    restored = ConversationSearchIndex.load(first)
    assert restored.query("alpha beta") == index.query("alpha beta")
    assert restored.terms() == index.terms()
    assert restored.save(second) == digest
    assert second.read_bytes() == first.read_bytes()


def test_snapshot_rejects_duplicate_or_malformed_documents(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"documents":[{"conversation_id":"c","utterance_id":"u","tokens":["x"]},'
        '{"conversation_id":"c","utterance_id":"u","tokens":["x"]}],"format":1}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        ConversationSearchIndex.load(path)
