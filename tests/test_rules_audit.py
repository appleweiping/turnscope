from datetime import datetime, timedelta, timezone

import pytest

from turnscope.audit import Auditor, default_auditor
from turnscope.models import Conversation, Severity, Utterance
from turnscope.rules import ConversationBudgetRule, RoleTransitionRule, TokenCountRule


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


def test_custom_severity_is_preserved() -> None:
    report = Auditor([RoleTransitionRule(severity=Severity.INFO)]).audit([broken_conversation()])
    assert report.issues[0].severity is Severity.INFO
