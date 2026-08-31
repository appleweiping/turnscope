from datetime import timedelta

import pytest

from turnscope.builder import ContextBuilder
from turnscope.models import Conversation
from turnscope.policies import (
    ReplyChainPolicy,
    TimeWindowPolicy,
    TokenBudgetPolicy,
    TurnWindowPolicy,
)


def test_turn_policy_and_target_selection(conversation: Conversation) -> None:
    windows = ContextBuilder(TurnWindowPolicy(2)).build(conversation, target_ids=["a2"])
    assert len(windows) == 1
    assert windows[0].utterance_ids == ("a1", "u2")
    assert windows[0].token_total == 6
    assert windows[0].policy == "turns:2"
    assert ContextBuilder(TurnWindowPolicy(0)).build(conversation)[2].context == ()


def test_token_budget_stops_at_first_message_that_does_not_fit(conversation: Conversation) -> None:
    windows = ContextBuilder(TokenBudgetPolicy(6)).build(conversation)
    assert windows[-1].utterance_ids == ("a1", "u2")
    window = ContextBuilder(TokenBudgetPolicy(2, include_target=True)).build(conversation)[-1]
    assert window.context == ()


def test_time_and_reply_chain_policies(conversation: Conversation) -> None:
    time_window = ContextBuilder(TimeWindowPolicy(timedelta(seconds=15))).build(conversation)[-1]
    assert time_window.utterance_ids == ("u2",)
    chain = ContextBuilder(ReplyChainPolicy(max_depth=2)).build(conversation)[-1]
    assert chain.utterance_ids == ("a1", "u2")
    assert ContextBuilder(ReplyChainPolicy(0)).build(conversation)[-1].context == ()


def test_reply_chain_handles_cycles_without_looping(conversation: Conversation) -> None:
    items = list(conversation.utterances)
    items[0] = type(items[0])(
        items[0].id, items[0].role, items[0].text, items[0].timestamp, reply_to="a1"
    )
    cyclic = Conversation("cyclic", items)
    assert ContextBuilder(ReplyChainPolicy()).build(cyclic)[2].utterance_ids == ("u1", "a1")


def test_builder_rejects_unknown_targets(conversation: Conversation) -> None:
    with pytest.raises(KeyError, match="missing"):
        ContextBuilder(TurnWindowPolicy(1)).build(conversation, target_ids=["missing"])


@pytest.mark.parametrize("invalid", [-1, True, 1.5])
def test_custom_token_counter_must_return_non_negative_integer(
    conversation: Conversation, invalid: object
) -> None:
    def counter(text: str) -> int:
        del text
        return invalid  # type: ignore[return-value]

    without_declared_counts = Conversation(
        "uncounted",
        [
            type(item)(item.id, item.role, item.text, item.timestamp, item.reply_to)
            for item in conversation.utterances
        ],
    )
    with pytest.raises(ValueError, match="non-negative integer"):
        ContextBuilder(TokenBudgetPolicy(5), counter).build(without_declared_counts)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TurnWindowPolicy(-1),
        lambda: TokenBudgetPolicy(-1),
        lambda: TimeWindowPolicy(timedelta(seconds=-1)),
        lambda: ReplyChainPolicy(-1),
    ],
)
def test_invalid_policy_values(factory) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        factory()
