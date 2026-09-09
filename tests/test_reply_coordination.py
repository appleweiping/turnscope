from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from turnscope import Conversation, Utterance, linguistic_coordination, reply_coordination
from turnscope.cli import main
from turnscope.io import conversation_to_dict


def utterance(identifier: str, speaker: str, text: str, parent: str | None = None) -> Utterance:
    return Utterance(
        identifier,
        "member",
        text,
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        parent,
        metadata={"speaker": speaker},
    )


def branch() -> Conversation:
    return Conversation(
        "branch",
        [
            utterance("a", "alice", "the plan"),
            utterance("noise", "other", "the interruption"),
            utterance("b", "bob", "the answer", "a"),
            utterance("c", "bob", "reply", "a"),
            utterance("d", "alice", "different"),
            utterance("e", "bob", "none", "d"),
        ],
    )


def test_branching_reply_counts_and_independent_probability_arithmetic() -> None:
    result = reply_coordination([branch()], {"article": ["the"]}, speaker_field="speaker")
    assert len(result) == 1
    score = result[0]
    assert (score.source, score.target) == ("alice", "bob")
    assert score.replies == 3
    assert score.conditioned_replies == 2
    assert score.response_category_replies == 1
    assert score.coordinated_replies == 1
    assert score.conditional_rate == pytest.approx(1 / 2)
    assert score.baseline_rate == pytest.approx(1 / 3)
    assert score.score == pytest.approx(1 / 6)
    assert score.support_failures == ()
    assert score.to_dict()["support_failures"] == []


def test_order_is_irrelevant_and_baseline_is_pair_specific() -> None:
    first = branch()
    second = Conversation(
        "partner",
        [utterance("root", "carol", "the"), utterance("reply", "bob", "the", "root")],
    )
    original = reply_coordination([first, second], {"article": ["the"]}, speaker_field="speaker")
    permuted = reply_coordination(
        [
            Conversation(second.id, tuple(reversed(second.utterances))),
            Conversation(first.id, tuple(reversed(first.utterances))),
        ],
        {"article": ["the"]},
        speaker_field="speaker",
    )
    assert original == permuted
    assert original[0].baseline_rate == pytest.approx(1 / 3)
    assert original[1].baseline_rate == 1
    assert original[1].score == 0


def test_always_used_category_has_zero_coordination() -> None:
    conversation = Conversation(
        "always",
        [
            utterance("a", "a", "the"),
            utterance("b", "b", "the", "a"),
            utterance("c", "a", "absent"),
            utterance("d", "b", "the", "c"),
        ],
    )
    score = reply_coordination([conversation], {"article": ["the"]}, speaker_field="speaker")[0]
    assert score.conditional_rate == score.baseline_rate == 1
    assert score.score == 0


def test_negative_coordination_and_zero_response_rate() -> None:
    conversation = Conversation(
        "avoid",
        [
            utterance("a", "a", "the"),
            utterance("b", "b", "nothing", "a"),
            utterance("c", "a", "none"),
            utterance("d", "b", "the", "c"),
        ],
    )
    score = reply_coordination([conversation], {"article": ["the"]}, speaker_field="speaker")[0]
    assert score.conditional_rate == 0
    assert score.baseline_rate == 0.5
    assert score.score == -0.5
    zero = Conversation("zero", [utterance("a", "a", "the"), utterance("b", "b", "none", "a")])
    assert reply_coordination([zero], {"article": ["the"]}, speaker_field="speaker")[0].score == 0


def test_support_thresholds_preserve_counts_rates_and_failure_reasons() -> None:
    scores = reply_coordination(
        [branch()],
        {"article": ["the"], "missing": ["absent"]},
        speaker_field="speaker",
        min_replies=4,
        min_conditioned_replies=3,
        min_response_category_replies=2,
    )
    article, missing = scores
    assert article.replies == 3 and article.conditional_rate == 0.5
    assert article.baseline_rate == pytest.approx(1 / 3)
    assert article.score is None
    assert article.support_failures == (
        "min_replies",
        "min_conditioned_replies",
        "min_response_category_replies",
    )
    assert missing.conditional_rate is None and missing.baseline_rate == 0
    assert missing.score is None
    assert reply_coordination(
        [branch()],
        {"article": ["the"]},
        speaker_field="speaker",
        min_replies=3,
        min_conditioned_replies=2,
        min_response_category_replies=1,
    )[0].score == pytest.approx(1 / 6)


def test_no_invented_adjacency_same_speaker_or_quadratic_unobserved_pairs() -> None:
    roots = [utterance(str(index), str(index), "the") for index in range(1000)]
    roots.append(utterance("self", "0", "the", "0"))
    assert reply_coordination([Conversation("isolated", roots)], speaker_field="speaker") == ()
    assert reply_coordination([]) == ()
    # Roles remain the default identity; all member-role edges are self replies.
    assert reply_coordination([branch()]) == ()


def test_repeated_message_ids_are_scoped_to_conversation_and_corpus_counts_add() -> None:
    first = branch()
    second = Conversation("second", first.utterances)
    score = reply_coordination(
        iter([first, second]), {"article": ["the"]}, speaker_field="speaker"
    )[0]
    assert (score.replies, score.conditioned_replies, score.response_category_replies) == (6, 4, 2)
    assert score.score == pytest.approx(1 / 6)


def test_missing_parent_duplicate_message_conversation_and_cycles_fail() -> None:
    malformed = (
        (Conversation("dangling", [utterance("x", "a", "the", "unknown")]), "unknown reply parent"),
        (
            Conversation("duplicate", [utterance("x", "a", "the"), utterance("x", "b", "the")]),
            "duplicate utterance",
        ),
        (
            Conversation(
                "cycle", [utterance("x", "a", "the", "y"), utterance("y", "b", "the", "x")]
            ),
            "cycle",
        ),
        (Conversation("self-link", [utterance("x", "a", "the", "x")]), "cycle"),
    )
    for conversation, message in malformed:
        with pytest.raises(ValueError, match=message):
            reply_coordination([conversation], speaker_field="speaker")
    with pytest.raises(ValueError, match="duplicate conversation"):
        reply_coordination([branch(), branch()])
    with pytest.raises(TypeError, match="Conversation"):
        reply_coordination([None])  # type: ignore[list-item]


@pytest.mark.parametrize("identity", [None, 5, "", " "])
def test_missing_or_invalid_identity_fails_even_on_root(identity) -> None:
    item = utterance("root", "a", "the")
    item.metadata["speaker"] = identity  # type: ignore[index]
    with pytest.raises(ValueError, match="missing speaker identity"):
        reply_coordination([Conversation("one", [item])], speaker_field="speaker")


@pytest.mark.parametrize(
    "options",
    [
        {"speaker_field": ""},
        {"speaker_field": False},
        {"min_replies": 0},
        {"min_replies": True},
        {"min_conditioned_replies": 0},
        {"min_conditioned_replies": 1.5},
        {"min_response_category_replies": -1},
        {"min_response_category_replies": True},
    ],
)
def test_configuration_is_checked_without_data(options) -> None:
    with pytest.raises(ValueError):
        reply_coordination([], **options)


@pytest.mark.parametrize(
    "categories",
    [
        {},
        [],
        {"": ["the"]},
        {"x": "the"},
        {"x": b"the"},
        {"x": {"the": 1}},
        {"x": 3},
        {"x": []},
        {"x": [True]},
        {"x": [""]},
        {"x": ["two words"]},
        {"x": ["the!"]},
    ],
)
def test_invalid_lexicons_are_not_silently_dropped(categories) -> None:
    with pytest.raises(ValueError):
        reply_coordination([], categories)


def test_lexicon_casefold_generators_and_overlapping_categories() -> None:
    scores = reply_coordination(
        [branch()], {"z": iter(["THE", "the"]), "a": ["the"]}, speaker_field="speaker"
    )
    assert [score.category for score in scores] == ["a", "z"]
    assert scores[0].score == scores[1].score == pytest.approx(1 / 6)


def test_legacy_conditional_rate_keeps_existing_adjacent_semantics() -> None:
    items = [
        Utterance("a", "a", "the", datetime.now(timezone.utc)),
        Utterance("b", "b", "the", datetime.now(timezone.utc)),
    ]
    conversation = Conversation("legacy", items)
    assert linguistic_coordination(conversation, {"article": ["the"]})[0].score == 1
    assert reply_coordination([conversation], {"article": ["the"]}) == ()


def test_cli_outputs_explicit_counts_baseline_and_strict_inputs(tmp_path: Path, capsys) -> None:
    source, lexicon, output = [
        tmp_path / name for name in ("input.jsonl", "lexicon.json", "result.json")
    ]
    source.write_text(json.dumps(conversation_to_dict(branch())), encoding="utf-8")
    lexicon.write_text('{"article": ["the"]}', encoding="utf-8")
    base = [
        "reply-coordination",
        str(source),
        "--speaker-field",
        "speaker",
        "--categories",
        str(lexicon),
    ]
    assert main(base) == 0
    score = json.loads(capsys.readouterr().out)[0]
    assert score["replies"] == 3 and score["score"] == pytest.approx(1 / 6)
    assert main([*base, "--min-replies", "4", "--output", str(output)]) == 0
    assert json.loads(output.read_text())[0]["support_failures"] == ["min_replies"]
    previous = output.read_bytes()
    for args in (
        [*base, "--output", str(source)],
        [*base, "--output", str(lexicon)],
        [*base, "--min-replies", "0", "--output", str(output)],
    ):
        assert main(args) == 2
    assert output.read_bytes() == previous
    lexicon.write_text('{"article": ["the"], "article": ["a"]}', encoding="utf-8")
    assert main(base) == 2
    assert "duplicate" in capsys.readouterr().err


def test_cli_default_lexicon_and_dangling_parent_error(tmp_path: Path, capsys) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(conversation_to_dict(branch())), encoding="utf-8")
    assert main(["reply-coordination", str(source), "--speaker-field", "speaker"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 5
    source.write_text(
        json.dumps(
            conversation_to_dict(Conversation("bad", [utterance("x", "a", "the", "unknown")]))
        ),
        encoding="utf-8",
    )
    assert main(["reply-coordination", str(source)]) == 2
    assert "unknown reply parent" in capsys.readouterr().err
