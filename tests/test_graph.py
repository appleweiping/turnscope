from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from turnscope import Conversation, Utterance, interaction_edges, reply_forest


def message(id: str, parent: str | None = None, seconds: int = 0) -> Utterance:
    return Utterance(
        id,
        "user",
        "text",
        datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds),
        reply_to=parent,
    )


def test_forest_unsorted_multiple_roots() -> None:
    forest = reply_forest(
        Conversation(
            "c",
            [message("d", "b"), message("b", "a"), message("c", "a"), message("a"), message("e")],
        )
    )
    assert forest.roots == ("a", "e")
    assert forest.traversal == ("a", "e", "b", "c", "d")
    assert dict(forest.depths) == {"a": 0, "e": 0, "b": 1, "c": 1, "d": 2}
    assert dict(forest.descendants) == {"a": 3, "b": 1, "c": 0, "d": 0, "e": 0}
    assert forest.ancestors("d") == ("b", "a")
    assert forest.ancestors("a") == ()
    assert forest.subtree("b") == ("b", "d")
    with pytest.raises(KeyError):
        forest.ancestors("missing")
    with pytest.raises(KeyError):
        forest.subtree("missing")
    with pytest.raises(TypeError):
        forest.depths["a"] = 3  # type: ignore[index]


@pytest.mark.parametrize(
    "items, reason",
    [
        ([message("a"), message("a")], "duplicate"),
        ([message("a", "absent")], "unknown reply"),
        ([message("a", "a")], "cycle"),
        ([message("a", "b"), message("b", "a"), message("c", "b")], "cycle"),
    ],
)
def test_invalid_forest(items: list[Utterance], reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        reply_forest(Conversation("c", items))


def test_deep_forest_does_not_recurse() -> None:
    items = [message(str(i), str(i - 1) if i else None) for i in range(5000)]
    forest = reply_forest(Conversation("deep", items))
    assert forest.depths["4999"] == 4999
    assert forest.descendants["0"] == 4999
    assert len(forest.ancestors("4999")) == 4999
    assert len(forest.subtree("0")) == 5000


def test_empty_forest() -> None:
    conversation = Conversation("empty", [])
    assert reply_forest(conversation).traversal == ()
    assert interaction_edges(conversation) == ()


def test_interactions_keep_negative_latencies_and_self_replies() -> None:
    conversation = Conversation(
        "c", [message("a", seconds=10), message("b", "a", 5), message("c", "b", 20)]
    )
    (edge,) = interaction_edges(conversation)
    assert (edge.sender, edge.recipient, edge.replies) == ("user", "user", 2)
    assert edge.mean_latency_seconds == 5
    assert edge.negative_latencies == 1


def test_explicit_speaker_identity(conversation: Conversation) -> None:
    with pytest.raises(ValueError, match="speaker"):
        interaction_edges(conversation, speaker_field="speaker")
    conversation = Conversation(
        "speakers",
        [
            replace(message("a"), metadata={"speaker": "alice"}),
            replace(message("b", "a", 10), metadata={"speaker": "bob"}),
        ],
    )
    (edge,) = interaction_edges(conversation, speaker_field="speaker")
    assert (edge.sender, edge.recipient, edge.mean_latency_seconds) == ("bob", "alice", 10)
