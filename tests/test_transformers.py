from datetime import datetime, timezone

import pytest

from turnscope import (
    Conversation,
    TfidfVectorizer,
    Utterance,
    conversation_features,
    speaker_profiles,
)


def convo(identifier: str, texts: list[str], roles: list[str] | None = None) -> Conversation:
    roles = roles or ["user"] * len(texts)
    return Conversation(
        identifier,
        [
            Utterance(f"{identifier}-{i}", role, text, datetime(2026, 1, 1, tzinfo=timezone.utc))
            for i, (role, text) in enumerate(zip(roles, texts, strict=True))
        ],
    )


def test_tfidf_fit_transform_is_sparse_and_deterministic() -> None:
    first, second = (
        convo("a", ["Cats chase cats", "shared word"]),
        convo("b", ["Dogs chase", "shared word"]),
    )
    vectorizer = TfidfVectorizer(min_document_frequency=2)
    output = vectorizer.fit_transform((first, second))
    assert vectorizer.state.vocabulary == ("chase", "shared", "word")
    assert output[0]["a-0"]["chase"] > 0
    assert "cats" not in output[0]["a-0"]
    assert output[0]["a-1"] == output[1]["b-1"]
    with pytest.raises(TypeError):
        output[0]["a-0"]["new"] = 1  # type: ignore[index]


def test_fit_requires_data_and_transform_requires_fit() -> None:
    vectorizer = TfidfVectorizer()
    with pytest.raises(ValueError, match="fitted"):
        vectorizer.transform(convo("a", ["text"]))
    with pytest.raises(ValueError, match="required"):
        vectorizer.fit([])
    with pytest.raises(ValueError, match="positive"):
        TfidfVectorizer(min_document_frequency=0)
    with pytest.raises(ValueError, match="positive"):
        TfidfVectorizer(max_features=0)


def test_structural_and_speaker_features() -> None:
    conversation = convo("c", ["hello there", "answer", "again"], ["alice", "bob", "alice"])
    features = conversation_features(conversation)
    assert features.utterances == 3 and features.tokens == 4
    assert dict(features.roles) == {"alice": 2, "bob": 1}
    assert features.roots == 3 and features.mean_turn_tokens == pytest.approx(4 / 3)
    profiles = speaker_profiles(conversation)
    assert profiles[0].speaker == "alice" and profiles[0].tokens == 3
    with pytest.raises(ValueError, match="speaker"):
        speaker_profiles(conversation, field="speaker")


def test_reply_profiles_use_metadata_field() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    items = [
        Utterance("a", "role", "a", start, metadata={"speaker": "alice"}),
        Utterance("b", "role", "b", start, "a", metadata={"speaker": "bob"}),
    ]
    profiles = speaker_profiles(Conversation("c", items), field="speaker")
    assert profiles[0].replies_received == 1 and profiles[1].replies_sent == 1
