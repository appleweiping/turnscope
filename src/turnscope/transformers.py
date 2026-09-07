"""Fit/transform text and conversation feature primitives."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .models import Conversation, Utterance

_TOKEN = re.compile(r"[\w]+(?:['-][\w]+)*", re.UNICODE)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _TOKEN.finditer(text))


@dataclass(frozen=True, slots=True)
class TfidfState:
    """Fitted vocabulary and inverse-document-frequency weights."""

    vocabulary: tuple[str, ...]
    idf: Mapping[str, float]
    documents: int


class TfidfVectorizer:
    """Sparse per-utterance TF-IDF vectors with deterministic vocabulary order."""

    def __init__(self, *, min_document_frequency: int = 1, max_features: int | None = None) -> None:
        if (
            isinstance(min_document_frequency, bool)
            or not isinstance(min_document_frequency, int)
            or min_document_frequency < 1
        ):
            raise ValueError("min_document_frequency must be a positive integer")
        if max_features is not None and (
            isinstance(max_features, bool) or not isinstance(max_features, int) or max_features < 1
        ):
            raise ValueError("max_features must be a positive integer or None")
        self._min_df = min_document_frequency
        self._max_features = max_features
        self._state: TfidfState | None = None

    @property
    def state(self) -> TfidfState:
        if self._state is None:
            raise ValueError("vectorizer has not been fitted")
        return self._state

    def fit(self, conversations: Iterable[Conversation]) -> TfidfVectorizer:
        """Fit on conversation documents, counting each conversation once."""
        materialized = tuple(conversations)
        if not materialized:
            raise ValueError("at least one conversation is required")
        document_frequency: Counter[str] = Counter()
        for conversation in materialized:
            document_frequency.update(
                {token for item in conversation.utterances for token in _tokens(item.text)}
            )
        candidates = [
            (token, frequency)
            for token, frequency in document_frequency.items()
            if frequency >= self._min_df
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        if self._max_features is not None:
            candidates = candidates[: self._max_features]
        vocabulary = tuple(token for token, _ in candidates)
        documents = len(materialized)
        idf = MappingProxyType(
            {
                token: math.log((1 + documents) / (1 + document_frequency[token])) + 1.0
                for token in vocabulary
            }
        )
        self._state = TfidfState(vocabulary, idf, documents)
        return self

    def transform(self, conversation: Conversation) -> Mapping[str, Mapping[str, float]]:
        """Return sparse normalized vectors keyed by utterance ID."""
        state = self.state
        vocabulary = set(state.vocabulary)
        vectors: dict[str, Mapping[str, float]] = {}
        for item in conversation.utterances:
            counts = Counter(token for token in _tokens(item.text) if token in vocabulary)
            total = sum(counts.values())
            values = (
                {token: (count / total) * state.idf[token] for token, count in counts.items()}
                if total
                else {}
            )
            vectors[item.id] = MappingProxyType(dict(sorted(values.items())))
        return MappingProxyType(vectors)

    def fit_transform(
        self, conversations: Sequence[Conversation]
    ) -> tuple[Mapping[str, Mapping[str, float]], ...]:
        """Fit once and transform the supplied sequence in its original order."""
        self.fit(conversations)
        return tuple(self.transform(conversation) for conversation in conversations)


@dataclass(frozen=True, slots=True)
class ConversationFeatures:
    """Output-size-independent structural and lexical conversation features."""

    conversation_id: str
    utterances: int
    roles: Mapping[str, int]
    tokens: int
    unique_tokens: int
    reply_edges: int
    roots: int
    mean_turn_tokens: float


def conversation_features(conversation: Conversation) -> ConversationFeatures:
    """Compute deterministic counts without requiring a fitted model."""
    role_counts = Counter(item.role for item in conversation.utterances)
    tokens = [token for item in conversation.utterances for token in _tokens(item.text)]
    return ConversationFeatures(
        conversation.id,
        len(conversation.utterances),
        MappingProxyType(dict(sorted(role_counts.items()))),
        len(tokens),
        len(set(tokens)),
        sum(item.reply_to is not None for item in conversation.utterances),
        sum(item.reply_to is None for item in conversation.utterances),
        len(tokens) / len(conversation.utterances) if conversation.utterances else 0.0,
    )


@dataclass(frozen=True, slots=True)
class SpeakerProfile:
    """Role/metadata speaker aggregate for one conversation."""

    speaker: str
    utterances: int
    tokens: int
    unique_tokens: int
    replies_sent: int
    replies_received: int


def speaker_profiles(
    conversation: Conversation, *, field: str | None = None
) -> tuple[SpeakerProfile, ...]:
    """Aggregate by role or a required string metadata field."""
    names: dict[str, list[Utterance]] = defaultdict(list)
    for item in conversation.utterances:
        value = item.role if field is None else item.metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing speaker value for utterance {item.id!r}")
        names[value].append(item)
    by_id = conversation.by_id()
    result: list[SpeakerProfile] = []
    for speaker, items in sorted(names.items()):
        tokens = [token for item in items for token in _tokens(item.text)]
        sent = sum(item.reply_to is not None for item in items)
        received_count = sum(
            (
                by_id[item.reply_to].role
                if field is None
                else by_id[item.reply_to].metadata.get(field)
            )
            == speaker
            for item in conversation.utterances
            if item.reply_to in by_id
        )
        result.append(
            SpeakerProfile(speaker, len(items), len(tokens), len(set(tokens)), sent, received_count)
        )
    return tuple(result)
