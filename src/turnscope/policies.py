"""Composable policies that select prior utterances for a target."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

from .models import Utterance


class TokenCounter(Protocol):
    def __call__(self, text: str) -> int: ...


def whitespace_tokens(text: str) -> int:
    """A deterministic dependency-free token estimate."""
    return len(text.split())


class WindowPolicy(Protocol):
    """Structural protocol implemented by all selection policies."""

    @property
    def name(self) -> str: ...

    def select(
        self, prior: Sequence[Utterance], target: Utterance, *, token_counter: TokenCounter
    ) -> tuple[Utterance, ...]: ...


@dataclass(frozen=True, slots=True)
class TurnWindowPolicy:
    """Keep at most ``turns`` immediately preceding utterances."""

    turns: int

    def __post_init__(self) -> None:
        if self.turns < 0:
            raise ValueError("turns must be non-negative")

    @property
    def name(self) -> str:
        return f"turns:{self.turns}"

    def select(
        self, prior: Sequence[Utterance], target: Utterance, *, token_counter: TokenCounter
    ) -> tuple[Utterance, ...]:
        del target, token_counter
        return tuple(prior[-self.turns :]) if self.turns else ()


@dataclass(frozen=True, slots=True)
class TokenBudgetPolicy:
    """Select the newest complete utterances that fit the token budget."""

    budget: int
    include_target: bool = False

    def __post_init__(self) -> None:
        if self.budget < 0:
            raise ValueError("budget must be non-negative")

    @property
    def name(self) -> str:
        return f"tokens:{self.budget}"

    def select(
        self, prior: Sequence[Utterance], target: Utterance, *, token_counter: TokenCounter
    ) -> tuple[Utterance, ...]:
        remaining = self.budget - (
            count_tokens(target, token_counter) if self.include_target else 0
        )
        if remaining < 0:
            return ()
        selected: list[Utterance] = []
        for item in reversed(prior):
            cost = count_tokens(item, token_counter)
            if cost > remaining:
                break
            selected.append(item)
            remaining -= cost
        return tuple(reversed(selected))


@dataclass(frozen=True, slots=True)
class TimeWindowPolicy:
    """Keep preceding messages no older than a duration relative to target."""

    duration: timedelta

    def __post_init__(self) -> None:
        if self.duration < timedelta(0):
            raise ValueError("duration must be non-negative")

    @property
    def name(self) -> str:
        return f"time:{self.duration.total_seconds():g}s"

    def select(
        self, prior: Sequence[Utterance], target: Utterance, *, token_counter: TokenCounter
    ) -> tuple[Utterance, ...]:
        del token_counter
        cutoff = target.timestamp - self.duration
        return tuple(item for item in prior if cutoff <= item.timestamp <= target.timestamp)


@dataclass(frozen=True, slots=True)
class ReplyChainPolicy:
    """Follow ``reply_to`` ancestors, returning them in chronological chain order."""

    max_depth: int | None = None

    def __post_init__(self) -> None:
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")

    @property
    def name(self) -> str:
        depth = "all" if self.max_depth is None else str(self.max_depth)
        return f"reply-chain:{depth}"

    def select(
        self, prior: Sequence[Utterance], target: Utterance, *, token_counter: TokenCounter
    ) -> tuple[Utterance, ...]:
        del token_counter
        by_id = {item.id: item for item in prior}
        cursor = target.reply_to
        selected: list[Utterance] = []
        seen: set[str] = set()
        while cursor is not None and cursor in by_id and cursor not in seen:
            if self.max_depth is not None and len(selected) >= self.max_depth:
                break
            seen.add(cursor)
            item = by_id[cursor]
            selected.append(item)
            cursor = item.reply_to
        return tuple(reversed(selected))


def count_tokens(item: Utterance, counter: TokenCounter) -> int:
    """Return a validated declared or computed token count."""
    value = item.token_count if item.token_count is not None else counter(item.text)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("token counter must return a non-negative integer")
    return value
