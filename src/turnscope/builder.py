"""Context-window construction that preserves input positions."""

from __future__ import annotations

from collections.abc import Iterable

from .models import ContextWindow, Conversation, Utterance
from .policies import TokenCounter, WindowPolicy, count_tokens, whitespace_tokens


class ContextBuilder:
    """Build one deterministic context window per selected target."""

    def __init__(
        self, policy: WindowPolicy, token_counter: TokenCounter = whitespace_tokens
    ) -> None:
        self.policy = policy
        self.token_counter = token_counter

    def build(
        self, conversation: Conversation, *, target_ids: Iterable[str] | None = None
    ) -> tuple[ContextWindow, ...]:
        selected_ids = set(target_ids) if target_ids is not None else None
        known_ids = {item.id for item in conversation.utterances}
        if selected_ids is not None:
            missing = selected_ids - known_ids
            if missing:
                raise KeyError(f"unknown target utterance IDs: {', '.join(sorted(missing))}")
        windows: list[ContextWindow] = []
        prior: list[Utterance] = []
        for target in conversation.utterances:
            if selected_ids is None or target.id in selected_ids:
                context = self.policy.select(prior, target, token_counter=self.token_counter)
                windows.append(
                    ContextWindow(
                        conversation_id=conversation.id,
                        target=target,
                        context=context,
                        policy=self.policy.name,
                        token_total=sum(self._tokens(item) for item in context),
                    )
                )
            prior.append(target)
        return tuple(windows)

    def _tokens(self, item: Utterance) -> int:
        return count_tokens(item, self.token_counter)
