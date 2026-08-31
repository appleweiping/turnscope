"""Built-in audit rules for common conversation-data reliability failures."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from .models import Conversation, Issue, Severity
from .policies import TokenCounter, count_text_tokens, count_tokens, whitespace_tokens


class AuditRule(Protocol):
    """A stateless rule that returns deterministic findings in input order."""

    @property
    def name(self) -> str: ...

    def check(self, conversation: Conversation) -> tuple[Issue, ...]: ...


@dataclass(frozen=True, slots=True)
class ChronologyRule:
    severity: Severity = Severity.ERROR
    name: str = "chronology"

    def check(self, conversation: Conversation) -> tuple[Issue, ...]:
        issues: list[Issue] = []
        for previous, current in zip(
            conversation.utterances, conversation.utterances[1:], strict=False
        ):
            if current.timestamp < previous.timestamp:
                issues.append(
                    Issue(
                        self.name,
                        self.severity,
                        f"{current.id!r} occurs before the preceding utterance {previous.id!r}",
                        conversation.id,
                        (previous.id, current.id),
                    )
                )
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class DuplicateIdRule:
    severity: Severity = Severity.ERROR
    name: str = "duplicate-id"

    def check(self, conversation: Conversation) -> tuple[Issue, ...]:
        counts = Counter(item.id for item in conversation.utterances)
        return tuple(
            Issue(
                self.name,
                self.severity,
                f"utterance ID {item_id!r} appears {count} times",
                conversation.id,
                (item_id,) * count,
                {"count": count},
            )
            for item_id, count in counts.items()
            if count > 1
        )


@dataclass(frozen=True, slots=True)
class OrphanReplyRule:
    severity: Severity = Severity.ERROR
    name: str = "orphan-reply"

    def check(self, conversation: Conversation) -> tuple[Issue, ...]:
        known = {item.id for item in conversation.utterances}
        return tuple(
            Issue(
                self.name,
                self.severity,
                f"{item.id!r} replies to missing utterance {item.reply_to!r}",
                conversation.id,
                (item.id,),
                {"reply_to": item.reply_to},
            )
            for item in conversation.utterances
            if item.reply_to is not None and item.reply_to not in known
        )


@dataclass(frozen=True, slots=True)
class FutureReplyRule:
    """Detect references to messages positioned later or timestamped in the future."""

    severity: Severity = Severity.ERROR
    name: str = "future-leakage"

    def check(self, conversation: Conversation) -> tuple[Issue, ...]:
        positions = {item.id: index for index, item in enumerate(conversation.utterances)}
        by_id = conversation.by_id()
        issues: list[Issue] = []
        for index, item in enumerate(conversation.utterances):
            if item.reply_to is None or item.reply_to not in by_id:
                continue
            referenced = by_id[item.reply_to]
            if positions[item.reply_to] >= index or referenced.timestamp > item.timestamp:
                issues.append(
                    Issue(
                        self.name,
                        self.severity,
                        f"{item.id!r} references reply context unavailable at its position",
                        conversation.id,
                        (item.id, item.reply_to),
                    )
                )
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class RoleTransitionRule:
    """Flag adjacent same-role turns unless the role is explicitly exempted."""

    exempt_roles: frozenset[str] = frozenset({"system", "tool"})
    severity: Severity = Severity.WARNING
    name: str = "role-transition"

    def check(self, conversation: Conversation) -> tuple[Issue, ...]:
        return tuple(
            Issue(
                self.name,
                self.severity,
                f"adjacent utterances use the same role {current.role!r}",
                conversation.id,
                (previous.id, current.id),
                {"role": current.role},
            )
            for previous, current in zip(
                conversation.utterances, conversation.utterances[1:], strict=False
            )
            if previous.role == current.role and current.role not in self.exempt_roles
        )


@dataclass(frozen=True, slots=True)
class TokenCountRule:
    """Compare declared token counts with a deterministic counter."""

    tolerance: int = 0
    token_counter: TokenCounter = whitespace_tokens
    severity: Severity = Severity.WARNING
    name: str = "token-count"

    def __post_init__(self) -> None:
        if (
            isinstance(self.tolerance, bool)
            or not isinstance(self.tolerance, int)
            or self.tolerance < 0
        ):
            raise ValueError("tolerance must be a non-negative integer")

    def check(self, conversation: Conversation) -> tuple[Issue, ...]:
        issues: list[Issue] = []
        for item in conversation.utterances:
            if item.token_count is None:
                continue
            observed = count_text_tokens(item.text, self.token_counter)
            if abs(observed - item.token_count) > self.tolerance:
                issues.append(
                    Issue(
                        self.name,
                        self.severity,
                        f"declared token count {item.token_count} differs from observed {observed}",
                        conversation.id,
                        (item.id,),
                        {"declared": item.token_count, "observed": observed},
                    )
                )
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ConversationBudgetRule:
    """Flag conversations whose total declared or estimated tokens exceed a budget."""

    budget: int
    token_counter: TokenCounter = whitespace_tokens
    severity: Severity = Severity.WARNING
    name: str = "token-budget"

    def __post_init__(self) -> None:
        if isinstance(self.budget, bool) or not isinstance(self.budget, int) or self.budget < 0:
            raise ValueError("budget must be a non-negative integer")

    def check(self, conversation: Conversation) -> tuple[Issue, ...]:
        total = sum(count_tokens(item, self.token_counter) for item in conversation.utterances)
        if total <= self.budget:
            return ()
        return (
            Issue(
                self.name,
                self.severity,
                f"conversation uses {total} tokens, exceeding budget {self.budget}",
                conversation.id,
                details={"tokens": total, "budget": self.budget},
            ),
        )
