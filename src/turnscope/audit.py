"""Audit orchestration with configurable, independently testable rules."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .models import AuditReport, Conversation, Issue
from .rules import (
    AuditRule,
    ChronologyRule,
    ConversationBudgetRule,
    DuplicateIdRule,
    FutureReplyRule,
    OrphanReplyRule,
    RoleTransitionRule,
    TokenCountRule,
)


class Auditor:
    """Run rules in declaration order and preserve each rule's finding order."""

    def __init__(self, rules: Sequence[AuditRule]) -> None:
        names = [rule.name for rule in rules]
        if len(names) != len(set(names)):
            raise ValueError("audit rule names must be unique")
        self.rules = tuple(rules)

    def audit(self, conversations: Iterable[Conversation]) -> AuditReport:
        items = tuple(conversations)
        issues: list[Issue] = []
        for conversation in items:
            for rule in self.rules:
                issues.extend(rule.check(conversation))
        return AuditReport(
            issues=tuple(issues),
            conversations=len(items),
            utterances=sum(len(item.utterances) for item in items),
        )


def default_auditor(*, token_budget: int | None = None) -> Auditor:
    """Return the stable default rule set, optionally adding a total-token budget."""
    rules: list[AuditRule] = [
        ChronologyRule(),
        DuplicateIdRule(),
        OrphanReplyRule(),
        FutureReplyRule(),
        RoleTransitionRule(),
        TokenCountRule(),
    ]
    if token_budget is not None:
        rules.append(ConversationBudgetRule(token_budget))
    return Auditor(rules)
