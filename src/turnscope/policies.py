"""Composable policies that select prior utterances for a target."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol, runtime_checkable

from .models import Utterance


@runtime_checkable
class TokenCounter(Protocol):
    """Callable protocol for deterministic text-to-token estimates.

    Implementations must be pure for a given string. ``ContextBuilder`` may
    memoize calls while constructing many overlapping windows.
    """

    def __call__(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class WhitespaceTokenCounter:
    """Count non-empty runs separated by Unicode whitespace."""

    def __call__(self, text: str) -> int:
        return len(text.split())


@dataclass(frozen=True, slots=True)
class Utf8ByteTokenCounter:
    """Estimate tokens from UTF-8 bytes using a fixed, documented divisor.

    This is not a model tokenizer. It is useful when a stable multilingual
    approximation is preferable to whitespace splitting.
    """

    bytes_per_token: int = 4

    def __post_init__(self) -> None:
        if (
            isinstance(self.bytes_per_token, bool)
            or not isinstance(self.bytes_per_token, int)
            or self.bytes_per_token <= 0
        ):
            raise ValueError("bytes_per_token must be a positive integer")

    def __call__(self, text: str) -> int:
        byte_count = len(text.encode("utf-8"))
        return (byte_count + self.bytes_per_token - 1) // self.bytes_per_token


_WHITESPACE_COUNTER = WhitespaceTokenCounter()


def whitespace_tokens(text: str) -> int:
    """Compatibility function for the default whitespace counter."""
    return _WHITESPACE_COUNTER(text)


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
        if isinstance(self.turns, bool) or not isinstance(self.turns, int) or self.turns < 0:
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
        if isinstance(self.budget, bool) or not isinstance(self.budget, int) or self.budget < 0:
            raise ValueError("budget must be non-negative")
        if not isinstance(self.include_target, bool):
            raise ValueError("include_target must be a boolean")

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
        if not isinstance(self.duration, timedelta) or self.duration < timedelta(0):
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
        if self.max_depth is not None and (
            isinstance(self.max_depth, bool)
            or not isinstance(self.max_depth, int)
            or self.max_depth < 0
        ):
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
    if item.token_count is not None:
        return item.token_count
    return count_text_tokens(item.text, counter)


def count_text_tokens(text: str, counter: TokenCounter) -> int:
    """Return one validated counter result without consulting a declared count."""
    value = counter(text)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("token counter must return a non-negative integer")
    return value
