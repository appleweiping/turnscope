from datetime import datetime, timedelta, timezone

import pytest

from turnscope.audit import Auditor, default_auditor
from turnscope.models import Conversation, Severity, Utterance
from turnscope.rules import (
    ConversationBudgetRule,
    DuplicateIdRule,
    RoleTransitionRule,
    TokenCountRule,
)


def broken_conversation() -> Conversation:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Conversation(
        "broken",
        [
            Utterance("dup", "user", "two words", start, reply_to="future", token_count=1),
            Utterance("dup", "user", "again", start - timedelta(seconds=1)),
            Utterance(
                "future",
                "assistant",
                "later",
                start + timedelta(seconds=20),
                reply_to="missing",
            ),
        ],
    )


def test_default_audit_finds_all_reliability_classes() -> None:
    report = default_auditor(token_budget=2).audit([broken_conversation()])
    assert {issue.rule for issue in report.issues} == {
        "chronology",
        "duplicate-id",
        "orphan-reply",
        "future-leakage",
        "role-transition",
        "token-count",
        "token-budget",
    }
    assert report.conversations == 1
    assert report.utterances == 3
    assert report.failing()


def test_rules_support_configuration(conversation: Conversation) -> None:
    rule = RoleTransitionRule(exempt_roles=frozenset({"user", "assistant"}))
    assert rule.check(conversation) == ()
    assert TokenCountRule(tolerance=10).check(conversation) == ()
    assert ConversationBudgetRule(11).check(conversation) == ()
    assert ConversationBudgetRule(10).check(conversation)[0].details["tokens"] == 11


def test_rule_validation_and_unique_names() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        TokenCountRule(-1)
    with pytest.raises(ValueError, match="budget"):
        ConversationBudgetRule(-1)
    with pytest.raises(ValueError, match="unique"):
        Auditor([RoleTransitionRule(), RoleTransitionRule()])
    assert Auditor([]).audit([]).issues == ()


@pytest.mark.parametrize("invalid", [True, 1.5])
def test_token_rule_configuration_requires_non_negative_integers(invalid: object) -> None:
    with pytest.raises(ValueError, match="integer"):
        TokenCountRule(invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer"):
        ConversationBudgetRule(invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [-1, True, 1.5])
def test_token_rules_validate_custom_counter_results(invalid: object) -> None:
    conversation = Conversation(
        "uncounted",
        [
            Utterance(
                "declared",
                "user",
                "text",
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                token_count=1,
            ),
            Utterance(
                "computed",
                "assistant",
                "text",
                datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ],
    )

    def counter(text: str) -> int:
        del text
        return invalid  # type: ignore[return-value]

    with pytest.raises(ValueError, match="non-negative integer"):
        TokenCountRule(token_counter=counter).check(conversation)
    with pytest.raises(ValueError, match="non-negative integer"):
        ConversationBudgetRule(10, token_counter=counter).check(conversation)


def test_duplicate_id_rule_scans_the_conversation_once() -> None:
    class CountingTuple(tuple[Utterance, ...]):
        iterations = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            self.iterations += 1
            return super().__iter__()

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    items = CountingTuple(Utterance(f"id-{index // 2}", "user", "x", start) for index in range(200))
    conversation = Conversation("duplicates", items)
    object.__setattr__(conversation, "utterances", items)
    items.iterations = 0
    assert len(DuplicateIdRule().check(conversation)) == 100
    assert items.iterations == 1


def test_custom_severity_is_preserved() -> None:
    report = Auditor([RoleTransitionRule(severity=Severity.INFO)]).audit([broken_conversation()])
    assert report.issues[0].severity is Severity.INFO


def test_auditor_consumes_conversations_in_one_pass(conversation: Conversation) -> None:
    consumed: list[str] = []

    def conversations():  # type: ignore[no-untyped-def]
        consumed.append(conversation.id)
        yield conversation

    report = Auditor([]).audit(conversations())
    assert consumed == [conversation.id]
    assert report.conversations == 1
    assert report.utterances == len(conversation.utterances)
