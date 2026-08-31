"""Context-window construction with streaming, output-sensitive fast paths."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Protocol

from .models import ContextWindow, Conversation, Utterance
from .policies import (
    ReplyChainPolicy,
    TimeWindowPolicy,
    TokenBudgetPolicy,
    TokenCounter,
    TurnWindowPolicy,
    WindowPolicy,
    count_tokens,
    whitespace_tokens,
)


class _CountedUtterance:
    """Compute and retain one utterance's token cost only while policy state needs it."""

    def __init__(self, item: Utterance, counter: TokenCounter) -> None:
        self.item = item
        self._counter = counter
        self._cost: int | None = None

    @property
    def cost(self) -> int:
        if self._cost is None:
            value = count_tokens(self.item, self._counter)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("token counter must return a non-negative integer")
            self._cost = value
        return self._cost


class _PolicyState(Protocol):
    def selection_for(self, target: _CountedUtterance) -> tuple[tuple[Utterance, ...], int]: ...

    def append(self, item: _CountedUtterance) -> None: ...


class _TurnState:
    def __init__(self, policy: TurnWindowPolicy) -> None:
        self._prior: deque[_CountedUtterance] = deque(maxlen=policy.turns or None)
        self._turns = policy.turns

    def selection_for(self, target: _CountedUtterance) -> tuple[tuple[Utterance, ...], int]:
        del target
        if not self._turns:
            return (), 0
        return tuple(entry.item for entry in self._prior), sum(entry.cost for entry in self._prior)

    def append(self, item: _CountedUtterance) -> None:
        if self._turns:
            self._prior.append(item)


class _TokenState:
    def __init__(self, policy: TokenBudgetPolicy) -> None:
        self._policy = policy
        self._prior: deque[_CountedUtterance] = deque()
        self._total = 0

    def selection_for(self, target: _CountedUtterance) -> tuple[tuple[Utterance, ...], int]:
        remaining = self._policy.budget
        if self._policy.include_target:
            remaining -= target.cost
        if remaining < 0:
            return (), 0

        selected: list[_CountedUtterance] = []
        token_total = 0
        for entry in reversed(self._prior):
            if entry.cost > remaining:
                break
            selected.append(entry)
            remaining -= entry.cost
            token_total += entry.cost
        return tuple(entry.item for entry in reversed(selected)), token_total

    def append(self, item: _CountedUtterance) -> None:
        if item.cost > self._policy.budget:
            # This item is a permanent barrier: no earlier item can appear in a
            # future contiguous suffix, so retaining either side is unnecessary.
            self._prior.clear()
            self._total = 0
            return
        self._prior.append(item)
        self._total += item.cost
        while self._total > self._policy.budget:
            self._total -= self._prior.popleft().cost


class _TimeState:
    def __init__(self, policy: TimeWindowPolicy) -> None:
        self._duration = policy.duration
        self._prior: list[_CountedUtterance] = []
        self._start = 0
        self._last_timestamp: datetime | None = None
        self._monotonic = True

    def selection_for(self, target: _CountedUtterance) -> tuple[tuple[Utterance, ...], int]:
        if self._last_timestamp is not None and target.item.timestamp < self._last_timestamp:
            self._monotonic = False
        cutoff = target.item.timestamp - self._duration
        if not self._monotonic:
            # Preserve v0.1 behavior for malformed, out-of-order conversations.
            fallback_selected = tuple(
                entry
                for entry in self._prior
                if cutoff <= entry.item.timestamp <= target.item.timestamp
            )
            return tuple(entry.item for entry in fallback_selected), sum(
                entry.cost for entry in fallback_selected
            )
        while self._start < len(self._prior) and self._prior[self._start].item.timestamp < cutoff:
            self._start += 1
        selected_entries = self._prior[self._start :]
        return tuple(entry.item for entry in selected_entries), sum(
            entry.cost for entry in selected_entries
        )

    def append(self, item: _CountedUtterance) -> None:
        if self._last_timestamp is not None and item.item.timestamp < self._last_timestamp:
            self._monotonic = False
        self._prior.append(item)
        self._last_timestamp = item.item.timestamp


class _ReplyState:
    def __init__(self, policy: ReplyChainPolicy) -> None:
        self._max_depth = policy.max_depth
        self._by_id: dict[str, _CountedUtterance] = {}

    def selection_for(self, target: _CountedUtterance) -> tuple[tuple[Utterance, ...], int]:
        cursor = target.item.reply_to
        selected: list[_CountedUtterance] = []
        seen: set[str] = set()
        while cursor is not None and cursor in self._by_id and cursor not in seen:
            if self._max_depth is not None and len(selected) >= self._max_depth:
                break
            seen.add(cursor)
            item = self._by_id[cursor]
            selected.append(item)
            cursor = item.item.reply_to
        return tuple(item.item for item in reversed(selected)), sum(item.cost for item in selected)

    def append(self, item: _CountedUtterance) -> None:
        # Match ``{item.id: item for item in prior}``: the last duplicate wins.
        self._by_id[item.item.id] = item


class _CompatibilityState:
    """Adapter for third-party v0.1 policies using the original protocol."""

    def __init__(self, policy: WindowPolicy, counter: TokenCounter) -> None:
        self._policy = policy
        self._counter = counter
        self._prior: list[Utterance] = []
        self._counted: dict[int, _CountedUtterance] = {}

    def selection_for(self, target: _CountedUtterance) -> tuple[tuple[Utterance, ...], int]:
        selected = self._policy.select(self._prior, target.item, token_counter=self._counter)
        token_total = 0
        for item in selected:
            counted = self._counted.get(id(item))
            token_total += (
                counted.cost if counted is not None else _CountedUtterance(item, self._counter).cost
            )
        return selected, token_total

    def append(self, item: _CountedUtterance) -> None:
        self._prior.append(item.item)
        self._counted[id(item.item)] = item


def _policy_state(policy: WindowPolicy, counter: TokenCounter) -> _PolicyState:
    if isinstance(policy, TurnWindowPolicy):
        return _TurnState(policy)
    if isinstance(policy, TokenBudgetPolicy):
        return _TokenState(policy)
    if isinstance(policy, TimeWindowPolicy):
        return _TimeState(policy)
    if isinstance(policy, ReplyChainPolicy):
        return _ReplyState(policy)
    return _CompatibilityState(policy, counter)


class ContextBuilder:
    """Build deterministic context windows from materialized or streamed input."""

    def __init__(
        self, policy: WindowPolicy, token_counter: TokenCounter = whitespace_tokens
    ) -> None:
        self.policy = policy
        self.token_counter = token_counter

    def iter_build(
        self,
        conversation_id: str,
        utterances: Iterable[Utterance],
        *,
        target_ids: Iterable[str] | None = None,
    ) -> Iterator[ContextWindow]:
        """Yield windows in one pass without materializing the conversation.

        If target IDs are supplied, an unknown ID is reported when the input
        iterator is exhausted. Windows yielded before that terminal validation
        remain valid; use :meth:`build` when all-or-nothing behavior is needed.
        """
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ValueError("conversation id must not be empty")
        selected_ids = set(target_ids) if target_ids is not None else None
        found_ids: set[str] = set()
        state = _policy_state(self.policy, self.token_counter)

        for target in utterances:
            counted_target = _CountedUtterance(target, self.token_counter)
            if selected_ids is None or target.id in selected_ids:
                context, token_total = state.selection_for(counted_target)
                found_ids.add(target.id)
                yield ContextWindow(
                    conversation_id=conversation_id,
                    target=target,
                    context=context,
                    policy=self.policy.name,
                    token_total=token_total,
                )
            state.append(counted_target)

        if selected_ids is not None:
            missing = selected_ids - found_ids
            if missing:
                raise KeyError(f"unknown target utterance IDs: {', '.join(sorted(missing))}")

    def build(
        self, conversation: Conversation, *, target_ids: Iterable[str] | None = None
    ) -> tuple[ContextWindow, ...]:
        """Return all windows, preserving the v0.1 materialized API."""
        return tuple(
            self.iter_build(conversation.id, conversation.utterances, target_ids=target_ids)
        )
