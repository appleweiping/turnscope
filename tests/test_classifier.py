from __future__ import annotations

from datetime import datetime, timezone

import pytest

from turnscope import Conversation, ConversationClassifier, Utterance


def _conversation(identifier: str, text: str) -> Conversation:
    return Conversation(
        identifier,
        [Utterance(f"{identifier}-u", "user", text, datetime(2026, 1, 1, tzinfo=timezone.utc))],
    )


def test_classifier_fit_predict_probability_and_round_trip(tmp_path) -> None:
    conversations = (_conversation("a", "refund payment"), _conversation("b", "ship package"))
    model = ConversationClassifier().fit(conversations, {"a": "billing", "b": "delivery"})
    assert model.predict(_conversation("c", "payment refund")) == "billing"
    assert sum(model.predict_proba(conversations[0]).values()) == pytest.approx(1.0)
    path = tmp_path / "model.json"
    model.save(path)
    restored = ConversationClassifier.load(path)
    assert restored.digest() == model.digest()
    assert restored.predict(_conversation("c", "payment refund")) == "billing"


def test_classifier_rejects_bad_labels_and_unfitted_calls() -> None:
    conversation = _conversation("a", "hello")
    with pytest.raises(ValueError, match="not been fitted"):
        ConversationClassifier().predict(conversation)
    with pytest.raises(ValueError, match="missing labels"):
        ConversationClassifier().fit((conversation,), {})
