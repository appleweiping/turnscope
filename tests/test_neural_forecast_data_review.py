"""Independent causal trace, typed admission and aggregate-budget checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from turnscope import neural_forecast_data as data
from turnscope import neural_token_data as token_data
from turnscope.models import Conversation, Utterance

START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def message(identifier, text, time, *, event=False, header=False):
    return Utterance(
        identifier,
        "not-a-feature",
        text,
        START + timedelta(seconds=time),
        metadata={"event": event, "is_section_header": header},
    )


def authored_sources(*, future_marker="future-marker"):
    positive = Conversation(
        "positive",
        (
            message("heading", "header-secret", 0, event="ignored", header=True),
            message("p1", "shared alpha", 1),
            message("p2", "shared beta", 1),
            message("p3", "shared gamma", 2),
            message("event", future_marker, 3, event=True),
            message("after", "post-event-secret", 4),
        ),
        {"forecast_groups": ["group-positive"]},
    )
    negative = Conversation(
        "negative",
        (
            message("n1", "shared delta", 1),
            message("n2", "shared epsilon", 2),
            message("censored", "censored-last-secret", 3),
        ),
        {"forecast_groups": ["group-negative"]},
    )
    return positive, negative


def test_native_prefix_admits_identity_bytes_before_accepting_the_catalog(monkeypatch):
    turn = data.ObservedTurn("def", START, "123456")
    monkeypatch.setitem(data._HARD, "max_source_bytes", 10)
    # Six text bytes plus six identity bytes exceed the ten-byte source cap.
    with pytest.raises(data.SequenceLimitError, match="byte"):
        data.ObservedPrefix("abc", (turn,))


def test_known_wrong_object_cardinality_is_rejected_without_expanding_mapping_keys():
    class WrongShape(Mapping):
        def __len__(self):
            return 4  # ObservedTurn has exactly three fields.

        def __iter__(self):
            pytest.fail("known invalid object cardinality was expanded before rejection")

        def __getitem__(self, key):
            pytest.fail("known invalid object field was accessed before rejection")

    with pytest.raises(data.SequenceDataError, match="fields"):
        data.ObservedTurn.from_dict(WrongShape())


def test_equal_time_and_first_future_event_have_an_independent_hand_counted_trace():
    dataset = data.prepare_sequence_forecasts(authored_sources())
    assert tuple(item.conversation_id for item in dataset.observations) == (
        "negative",
        "positive",
    )
    assert tuple(
        (row.conversation_index, row.endpoint, row.label, row.lead_turns)
        for row in dataset.examples
    ) == ((0, 1, False, None), (1, 1, True, 2), (1, 2, True, 1))
    assert dataset.weights == (1.0, 0.5, 0.5)
    assert tuple(
        tuple(turn.id for turn in dataset.prefix(row).turns) for row in dataset.examples
    ) == (
        ("n1", "n2"),
        ("p1", "p2"),
        ("p1", "p2", "p3"),
    )
    assert dataset.audit.source_turns == 9
    assert dataset.audit.header_turns == 1
    assert dataset.audit.observed_turns == 5
    assert dataset.audit.prefixes == 3
    assert dataset.audit.raw_tokens == 10


def test_future_text_cannot_enter_observations_or_training_vocabulary():
    first = data.prepare_sequence_forecasts(authored_sources())
    changed = data.prepare_sequence_forecasts(
        authored_sources(future_marker="different future text")
    )
    assert first.observation_digest == changed.observation_digest
    # Full source audits legitimately observe different source bytes; model inputs do not.
    assert first.audit.source_text_bytes != changed.audit.source_text_bytes
    vocabulary = token_data.fit_sequence_vocabulary(first)
    assert vocabulary.digest == token_data.fit_sequence_vocabulary(changed).digest
    assert vocabulary.documents == 2
    assert vocabulary.tokens == ("alpha", "beta", "delta", "epsilon", "gamma", "shared")
    assert vocabulary.document_frequencies == (1, 1, 1, 1, 1, 2)


def test_unsupervised_observations_ignore_outcome_groups_role_and_reply_metadata():
    source = Conversation(
        "observed",
        (
            replace(message("first", "alpha", 1), reply_to="private-reply", token_count=999),
            replace(message("second", "beta", 2), role="outcome-looking-role"),
        ),
        {"forecast_groups": "invalid if inspected", "event": "private-future-label"},
    )
    for turn in source.utterances:
        turn.metadata["event"] = "not-a-valid-supervised-label"
        turn.metadata["private-label"] = "private-value"
    prefix = data.observed_prefix(source)
    assert prefix.to_dict() == {
        "conversation_id": "observed",
        "turns": [
            {
                "id": "first",
                "timestamp": (START + timedelta(seconds=1)).isoformat(),
                "text": "alpha",
            },
            {
                "id": "second",
                "timestamp": (START + timedelta(seconds=2)).isoformat(),
                "text": "beta",
            },
        ],
    }
    with pytest.raises(data.SequenceDataError, match="groups"):
        data.prepare_sequence_forecasts((source,))


def test_pinned_truncation_counts_all_raw_tokens_without_fitting_the_tail():
    dataset = data.prepare_sequence_forecasts(
        (
            Conversation(
                "one",
                (
                    message("one", "alpha beta hidden", 1),
                    message("two", "alpha new unseen", 2),
                    message("last", "future", 3),
                ),
                {"forecast_groups": ["one"]},
            ),
        )
    )
    vocabulary = token_data.fit_sequence_vocabulary(
        dataset, max_turn_tokens=2, long_turn_policy="head"
    )
    assert vocabulary.tokens == ("alpha", "beta", "new")
    encoded = token_data.encode_observed_prefix(dataset.prefix(dataset.examples[0]), vocabulary)
    assert encoded.turns == ((3, 4, 2), (3, 5, 2))
    assert (
        encoded.raw_tokens,
        encoded.retained_tokens,
        encoded.known_tokens,
        encoded.truncated_turns,
    ) == (6, 4, 4, 2)
    tighter = replace(data.SequenceLimits(), max_tokens=5)
    with pytest.raises(data.SequenceLimitError, match="token"):
        token_data.encode_observed_prefix(
            dataset.prefix(dataset.examples[0]), vocabulary, limits=tighter
        )
    with pytest.raises(data.SequenceLimitError, match="pinned"):
        token_data.fit_sequence_vocabulary(dataset, max_turn_tokens=2, long_turn_policy="reject")


def test_source_budget_counts_excluded_future_text_instead_of_silently_discarding_it():
    dataset = data.prepare_sequence_forecasts(authored_sources())
    limits = replace(
        data.SequenceLimits(),
        max_source_bytes=dataset.audit.source_identity_bytes + dataset.audit.source_text_bytes - 1,
    )
    with pytest.raises(data.SequenceLimitError, match="source-byte"):
        data.prepare_sequence_forecasts(authored_sources(), limits=limits)
