"""Explicit first-event prefix examples; future annotations never enter model features."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import cast

from .models import Conversation
from .transformers import _tokens


@dataclass(frozen=True, slots=True)
class ForecastPrefix:
    """A text-only cumulative feature snapshot, with no future or annotation fields."""

    conversation_id: str
    utterance_id: str
    turns: int
    timestamp: datetime
    counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.counts, Mapping) or any(
            not isinstance(term, str) or not term or type(count) is not int or count < 0
            for term, count in self.counts.items()
        ):
            raise ValueError("prefix counts require non-empty tokens and non-negative integers")
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))


@dataclass(frozen=True, slots=True)
class ForecastExample:
    """Supervision outside the feature snapshot; lead_turns includes the event turn."""

    prefix: ForecastPrefix
    label: bool
    lead_turns: int | None


@dataclass(frozen=True, slots=True)
class ForecastDataset:
    examples: tuple[ForecastExample, ...]
    group_digests: frozenset[str]
    conversations: int
    excluded_conversations: int
    prefix_cells: int


def group_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prepare_forecast_examples(
    conversations: Iterable[Conversation],
    *,
    event_field: str = "event",
    skip_field: str = "is_section_header",
    groups_field: str = "forecast_groups",
    min_turns: int = 2,
    max_prefixes: int = 30_000,
    max_prefix_cells: int = 4_000_000,
    max_tokens: int = 2_000_000,
) -> ForecastDataset:
    """Build bounded snapshots strictly before the first annotated event.

    All timestamps must be nondecreasing. Equal-time observations are atomic:
    no prediction boundary can split them or call a simultaneous event future.
    Negatives mean no event in the observed record, not an uncensored guarantee.
    Explicit page/pair/etc. groups must be supplied by the dataset producer.
    """
    for name, field in (
        ("event_field", event_field),
        ("skip_field", skip_field),
        ("groups_field", groups_field),
    ):
        if not isinstance(field, str) or not field.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if event_field == skip_field:
        raise ValueError("event_field and skip_field must differ")
    for name, value in (
        ("min_turns", min_turns),
        ("max_prefixes", max_prefixes),
        ("max_prefix_cells", max_prefix_cells),
        ("max_tokens", max_tokens),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    seen: set[str] = set()
    groups: set[str] = set()
    examples: list[ForecastExample] = []
    cells = total_tokens = eligible_conversations = 0
    for conversation in conversations:
        if not isinstance(conversation, Conversation):
            raise TypeError("forecast input must contain Conversation values")
        if conversation.id in seen:
            raise ValueError("forecast conversation IDs must be unique")
        if len(seen) >= 100_000:
            raise ValueError("forecast conversation count exceeds 100000")
        seen.add(conversation.id)
        raw_groups = conversation.metadata.get(groups_field)
        if (
            not isinstance(raw_groups, list)
            or not raw_groups
            or not all(isinstance(value, str) and value.strip() for value in raw_groups)
            or len(set(raw_groups)) != len(raw_groups)
        ):
            raise ValueError(f"{groups_field} must contain distinct non-empty group strings")
        groups.add(group_digest("conversation\0" + conversation.id))
        groups.update(group_digest("group\0" + value) for value in cast(list[str], raw_groups))
        if len(groups) > 400_000:
            raise ValueError("forecast group count exceeds 400000")
        identifiers: set[str] = set()
        observed = []
        previous: datetime | None = None
        for utterance in conversation.utterances:
            if utterance.id in identifiers:
                raise ValueError("forecast utterance IDs must be unique within a conversation")
            identifiers.add(utterance.id)
            if previous is not None and utterance.timestamp < previous:
                raise ValueError("forecast timestamps must be nondecreasing")
            previous = utterance.timestamp
            skipped = utterance.metadata.get(skip_field, False)
            if type(skipped) is not bool:
                raise ValueError(f"{skip_field} must be boolean when supplied")
            if skipped:
                continue
            event = utterance.metadata.get(event_field)
            if type(event) is not bool:
                raise ValueError(f"every observed utterance needs boolean {event_field}")
            observed.append(utterance)
        first_event = next(
            (i for i, item in enumerate(observed) if item.metadata[event_field]), None
        )
        stop = first_event if first_event is not None else len(observed) - 1
        counter: Counter[str] = Counter()
        conversation_examples = 0
        for index in range(max(stop, 0)):
            item = observed[index]
            tokens = _tokens(item.text)
            total_tokens += len(tokens)
            if total_tokens > max_tokens:
                raise ValueError("forecast token budget exceeded")
            counter.update(tokens)
            if index + 1 < min_turns or item.timestamp == observed[index + 1].timestamp:
                continue
            if first_event is not None and item.timestamp >= observed[first_event].timestamp:
                continue
            if len(examples) >= max_prefixes or cells + len(counter) > max_prefix_cells:
                raise ValueError("forecast prefix count or sparse-cell budget exceeded")
            cells += len(counter)
            examples.append(
                ForecastExample(
                    ForecastPrefix(conversation.id, item.id, index + 1, item.timestamp, counter),
                    first_event is not None,
                    first_event - index if first_event is not None else None,
                )
            )
            conversation_examples += 1
        eligible_conversations += conversation_examples > 0
    return ForecastDataset(
        tuple(examples),
        frozenset(groups),
        eligible_conversations,
        len(seen) - eligible_conversations,
        cells,
    )
