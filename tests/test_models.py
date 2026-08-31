from datetime import datetime, timezone

import pytest

from turnscope.models import AuditReport, Conversation, Issue, Severity, Utterance


def test_utterance_normalizes_timezone_and_copies_metadata() -> None:
    metadata = {"source": "test"}
    item = Utterance(
        "id", "user", "text", datetime.fromisoformat("2026-01-01T08:00:00+08:00"), metadata=metadata
    )
    metadata["source"] = "changed"
    assert item.timestamp == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert item.metadata == {"source": "test"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"id": "", "role": "user", "token_count": None}, "id"),
        ({"id": "x", "role": "", "token_count": None}, "role"),
        ({"id": "x", "role": "user", "token_count": -1}, "token_count"),
        ({"id": "x", "role": "user", "token_count": True}, "token_count"),
        ({"id": "x", "role": "user", "token_count": 1.5}, "token_count"),
    ],
)
def test_utterance_rejects_invalid_fields(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Utterance(text="x", timestamp=datetime.now(timezone.utc), **kwargs)  # type: ignore[arg-type]


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValueError, match="timezone"):
        Utterance("x", "user", "text", datetime(2026, 1, 1))


def test_timestamp_utc_overflow_is_a_clear_value_error() -> None:
    timestamp = datetime.fromisoformat("9999-12-31T23:59:59-23:59")
    with pytest.raises(ValueError, match="supported UTC range"):
        Utterance("x", "user", "text", timestamp)


def test_conversation_and_report_helpers(conversation: Conversation) -> None:
    assert conversation.by_id()["u1"].text == "hello there"
    issue = Issue("test", Severity.WARNING, "message", conversation.id, ("u1",))
    report = AuditReport((issue,), 1, 4)
    assert report.counts() == {"info": 0, "warning": 1, "error": 0}
    assert report.failing(Severity.WARNING)
    assert not report.failing(Severity.ERROR)
    assert report.as_dict()["issues"][0]["severity"] == "warning"


def test_invalid_conversation_and_severity() -> None:
    with pytest.raises(ValueError, match="conversation id"):
        Conversation("", [])
    assert Severity.parse("Warning") is Severity.WARNING
    with pytest.raises(ValueError, match="unknown severity"):
        Severity.parse("fatal")
