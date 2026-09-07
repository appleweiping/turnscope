"""Audit orchestration with configurable, independently testable rules."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .models import AuditReport, Conversation, Issue
from .policies import TokenCounter, whitespace_tokens
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
        issues: list[Issue] = []
        conversation_count = 0
        utterance_count = 0
        for conversation in conversations:
            conversation_count += 1
            utterance_count += len(conversation.utterances)
            for rule in self.rules:
                issues.extend(rule.check(conversation))
        return AuditReport(
            issues=tuple(issues),
            conversations=conversation_count,
            utterances=utterance_count,
        )


def default_auditor(
    *, token_budget: int | None = None, token_counter: TokenCounter | None = None
) -> Auditor:
    """Return the stable default rule set, optionally adding a total-token budget."""
    counter = whitespace_tokens if token_counter is None else token_counter
    rules: list[AuditRule] = [
        ChronologyRule(),
        DuplicateIdRule(),
        OrphanReplyRule(),
        FutureReplyRule(),
        RoleTransitionRule(),
        TokenCountRule(token_counter=counter),
    ]
    if token_budget is not None:
        rules.append(ConversationBudgetRule(token_budget, token_counter=counter))
    return Auditor(rules)
