from datetime import datetime, timedelta, timezone

import pytest

from turnscope.models import Conversation, Utterance


@pytest.fixture
def conversation() -> Conversation:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Conversation(
        "demo",
        [
            Utterance("u1", "user", "hello there", start, token_count=2),
            Utterance("a1", "assistant", "how can I help", start + timedelta(seconds=10), "u1", 4),
            Utterance("u2", "user", "explain this", start + timedelta(seconds=20), "a1", 2),
            Utterance("a2", "assistant", "a useful answer", start + timedelta(seconds=30), "u2", 3),
        ],
    )
