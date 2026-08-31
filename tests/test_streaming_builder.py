from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import pytest

from turnscope.builder import ContextBuilder
from turnscope.models import Conversation, Utterance
from turnscope.policies import (
    ReplyChainPolicy,
    TimeWindowPolicy,
    TokenBudgetPolicy,
    TokenCounter,
    TurnWindowPolicy,
    Utf8ByteTokenCounter,
    WhitespaceTokenCounter,
    WindowPolicy,
    count_tokens,
)


def _reference(
    conversation: Conversation, policy: WindowPolicy, counter: TokenCounter
) -> list[tuple[tuple[str, ...], int]]:
    prior: list[Utterance] = []
    result: list[tuple[tuple[str, ...], int]] = []
    for target in conversation.utterances:
        selected = policy.select(prior, target, token_counter=counter)
        result.append(
            (
                tuple(item.id for item in selected),
                sum(count_tokens(item, counter) for item in selected),
            )
        )
        prior.append(target)
    return result


def _random_conversation(seed: int, *, monotonic: bool) -> Conversation:
    generator = random.Random(seed)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    items: list[Utterance] = []
    elapsed = 0
    for index in range(80):
        elapsed += generator.randint(0, 4)
        if not monotonic and index % 13 == 0:
            elapsed -= 8
        reply_to = None if index == 0 else f"m{generator.randrange(index)}"
        token_count = generator.randrange(5)
        items.append(
            Utterance(
                f"m{index}",
                "user" if index % 2 == 0 else "assistant",
                f"message {index}",
                start + timedelta(seconds=elapsed),
                reply_to=reply_to,
                token_count=token_count,
            )
        )
    return Conversation(f"random-{seed}", items)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("monotonic", [True, False])
@pytest.mark.parametrize(
    "policy",
    [
        TurnWindowPolicy(7),
        TokenBudgetPolicy(12),
        TokenBudgetPolicy(12, include_target=True),
        TimeWindowPolicy(timedelta(seconds=10)),
        ReplyChainPolicy(6),
    ],
)
def test_optimized_builder_matches_v01_policy_semantics(
    seed: int, monotonic: bool, policy: WindowPolicy
) -> None:
    conversation = _random_conversation(seed, monotonic=monotonic)
    expected = _reference(conversation, policy, WhitespaceTokenCounter())
    actual = ContextBuilder(policy, WhitespaceTokenCounter()).build(conversation)
    assert [(window.utterance_ids, window.token_total) for window in actual] == expected


def test_iter_build_is_lazy_and_reports_unknown_targets_at_exhaustion() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    consumed: list[str] = []

    def utterances():  # type: ignore[no-untyped-def]
        for index in range(3):
            item_id = f"m{index}"
            consumed.append(item_id)
            yield Utterance(item_id, "user", "text", start + timedelta(seconds=index))

    windows = ContextBuilder(TurnWindowPolicy(1)).iter_build(
        "stream", utterances(), target_ids={"m0", "missing"}
    )
    assert consumed == []
    assert next(windows).target.id == "m0"
    assert consumed == ["m0"]
    with pytest.raises(KeyError, match="missing"):
        list(windows)
    assert consumed == ["m0", "m1", "m2"]


def test_iter_build_validates_conversation_id() -> None:
    with pytest.raises(ValueError, match="conversation id"):
        next(ContextBuilder(TurnWindowPolicy(1)).iter_build("", []))


def test_custom_v01_policy_uses_compatibility_path(conversation: Conversation) -> None:
    class FirstPolicy:
        @property
        def name(self) -> str:
            return "first"

        def select(
            self,
            prior: Sequence[Utterance],
            target: Utterance,
            *,
            token_counter: TokenCounter,
        ) -> tuple[Utterance, ...]:
            del target, token_counter
            return tuple(prior[:1])

    windows = ContextBuilder(FirstPolicy()).build(conversation)
    assert windows[0].context == ()
    assert windows[-1].utterance_ids == ("u1",)


def test_counter_is_memoized_across_overlapping_windows() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    conversation = Conversation(
        "memo",
        [Utterance(f"m{i}", "user", f"unique {i}", start) for i in range(20)],
    )
    calls: list[str] = []

    def counter(text: str) -> int:
        calls.append(text)
        return 2

    ContextBuilder(TokenBudgetPolicy(100), counter).build(conversation)
    assert calls == [f"unique {index}" for index in range(20)]


def test_turn_counter_cache_is_bounded_to_retained_utterances() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    conversation = Conversation(
        "repeated",
        [Utterance(f"m{i}", "user", "same text", start) for i in range(20)],
    )
    calls = 0

    def counter(text: str) -> int:
        nonlocal calls
        assert text == "same text"
        calls += 1
        return 2

    ContextBuilder(TurnWindowPolicy(1), counter).build(conversation)
    # Each prior utterance is counted once, but evicted texts are not retained in
    # a conversation-wide cache. The final target never becomes context.
    assert calls == len(conversation.utterances) - 1


@pytest.mark.parametrize("budget", range(6))
@pytest.mark.parametrize("include_target", [False, True])
def test_token_suffix_boundaries_match_reference(budget: int, include_target: bool) -> None:
    conversation = _random_conversation(29, monotonic=True)
    policy = TokenBudgetPolicy(budget, include_target=include_target)
    expected = _reference(conversation, policy, WhitespaceTokenCounter())
    actual = ContextBuilder(policy).build(conversation)
    assert [(window.utterance_ids, window.token_total) for window in actual] == expected


def test_reply_index_preserves_last_duplicate_id_semantics() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    conversation = Conversation(
        "duplicates",
        [
            Utterance("same", "user", "first", start),
            Utterance("same", "assistant", "second", start),
            Utterance("target", "user", "third", start, reply_to="same"),
        ],
    )
    window = ContextBuilder(ReplyChainPolicy()).build(conversation)[-1]
    assert [item.text for item in window.context] == ["second"]


def test_time_state_detects_chronology_violation_on_unselected_target() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    conversation = Conversation(
        "selected",
        [
            Utterance("m0", "user", "zero", start),
            Utterance("m1", "assistant", "hundred", start + timedelta(seconds=100)),
            Utterance("m2", "user", "ten", start + timedelta(seconds=10)),
            Utterance("m3", "assistant", "twenty", start + timedelta(seconds=20)),
        ],
    )
    windows = ContextBuilder(TimeWindowPolicy(timedelta(seconds=30))).build(
        conversation, target_ids={"m1", "m3"}
    )
    assert windows[1].utterance_ids == ("m0", "m2")


def test_dependency_free_counter_implementations_and_validation() -> None:
    whitespace = WhitespaceTokenCounter()
    assert isinstance(whitespace, TokenCounter)
    assert whitespace("  hello\t世界 \n") == 2
    assert Utf8ByteTokenCounter()("abcd世界") == 3
    assert Utf8ByteTokenCounter(1)("世界") == 6
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            Utf8ByteTokenCounter(invalid)  # type: ignore[arg-type]
