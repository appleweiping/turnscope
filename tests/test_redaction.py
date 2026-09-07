from datetime import datetime, timezone

from turnscope import Conversation, RedactionPolicy, Utterance, redact_conversations


def _conversation() -> Conversation:
    return Conversation(
        "c1",
        [
            Utterance(
                "u1",
                "user",
                "Email A@Example.com or call +1 (555) 123-4567; key sk-abcdefghijklmnop.",
                datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            Utterance(
                "u2",
                "assistant",
                "Email A@Example.com again.",
                datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
            ),
        ],
    )


def test_redaction_is_deterministic_and_linkable() -> None:
    conversations, report = redact_conversations([_conversation()])
    first, second = conversations[0].utterances
    assert "A@Example.com" not in first.text
    assert "555" not in first.text
    assert "sk-" not in first.text
    assert "[REDACTED:email:" in first.text
    assert (
        first.text.split("[REDACTED:email:")[1][:10]
        == second.text.split("[REDACTED:email:")[1][:10]
    )
    assert report.replacements == 4
    assert dict(report.by_kind) == {"api_key": 1, "email": 2, "phone": 1}
    assert report.to_dict()["policy"]["url"] is False


def test_policy_can_disable_pattern_families() -> None:
    conversations, report = redact_conversations(
        [_conversation()], policy=RedactionPolicy(email=False, phone=False, api_key=False)
    )
    assert conversations[0].utterances[0].text == _conversation().utterances[0].text
    assert report.replacements == 0


def test_urls_are_opt_in() -> None:
    item = _conversation()
    utterance = item.utterances[0]
    item = Conversation(
        item.id,
        [
            Utterance(
                utterance.id, utterance.role, "visit https://example.com/a", utterance.timestamp
            )
        ],
    )
    conversations, report = redact_conversations([item], policy=RedactionPolicy(url=True))
    assert "https://example.com" not in conversations[0].utterances[0].text
    assert dict(report.by_kind)["url"] == 1
