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


@dataclass(frozen=True, slots=True)
class CorpusSpeakerProfile:
    """Aggregate one speaker identity across a conversation collection."""

    speaker: str
    conversations: int
    utterances: int
    tokens: int
    unique_tokens: int
    roles: Mapping[str, int]
    replies_sent: int
    replies_received: int


@dataclass(frozen=True, slots=True)
class DiversityProfile:
    """Lexical diversity summary for one speaker-like grouping."""

    speaker: str
    tokens: int
    unique_tokens: int
    type_token_ratio: float
    lexical_entropy: float


def linguistic_diversity(
    conversation: Conversation, *, field: str | None = None
) -> tuple[DiversityProfile, ...]:
    """Compute type-token ratio and Shannon lexical entropy per speaker.

    ``field=None`` groups by utterance role; otherwise a non-empty string
    metadata field supplies the grouping key. Empty groups cannot occur because
    every group is created from at least one utterance, while groups with no
    lexical tokens receive zero ratio and entropy rather than a division error.
    """

    groups: dict[str, list[str]] = defaultdict(list)
    for item in conversation.utterances:
        value = item.role if field is None else item.metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing speaker value for utterance {item.id!r}")
        groups[value].extend(_tokens(item.text))
    result: list[DiversityProfile] = []
    for speaker, tokens in sorted(groups.items()):
        counts = Counter(tokens)
        total = len(tokens)
        entropy = (
            -sum((count / total) * math.log2(count / total) for count in counts.values())
            if total
            else 0.0
        )
        result.append(
            DiversityProfile(
                speaker,
                total,
                len(counts),
                len(counts) / total if total else 0.0,
                entropy,
            )
        )
    return tuple(result)


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


def corpus_speaker_profiles(
    conversations: Iterable[Conversation], *, field: str | None = None
) -> tuple[CorpusSpeakerProfile, ...]:
    """Aggregate speaker identity, lexical counts, and reply edges globally.

    By default the utterance role is the speaker key. A metadata ``field`` can
    provide a stable speaker ID across conversations; missing or non-string
    values fail rather than silently merging records under a placeholder.
    Conversation IDs must be unique because they are part of the aggregate's
    denominator.
    """

    materialized = tuple(conversations)
    if not materialized:
        raise ValueError("at least one conversation is required")
    if not all(isinstance(item, Conversation) for item in materialized):
        raise TypeError("conversations must contain Conversation values")
    if len({item.id for item in materialized}) != len(materialized):
        raise ValueError("conversation IDs must be unique")
    conversation_ids: dict[str, set[str]] = defaultdict(set)
    utterance_counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    token_values: dict[str, set[str]] = defaultdict(set)
    role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    sent_counts: Counter[str] = Counter()
    received_counts: Counter[str] = Counter()

    def speaker(item: Utterance) -> str:
        value = item.role if field is None else item.metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing speaker value for utterance {item.id!r}")
        return value

    for conversation in materialized:
        by_id = conversation.by_id()
        speakers = {item.id: speaker(item) for item in conversation.utterances}
        for item in conversation.utterances:
            current = speakers[item.id]
            tokens = _tokens(item.text)
            conversation_ids[current].add(conversation.id)
            utterance_counts[current] += 1
            token_counts[current] += len(tokens)
            token_values[current].update(tokens)
            role_counts[current][item.role] += 1
            if item.reply_to in by_id:
                sent_counts[current] += 1
                received_counts[speakers[item.reply_to]] += 1

    return tuple(
        CorpusSpeakerProfile(
            name,
            len(conversation_ids[name]),
            utterance_counts[name],
            token_counts[name],
            len(token_values[name]),
            MappingProxyType(dict(sorted(role_counts[name].items()))),
            sent_counts[name],
            received_counts[name],
        )
        for name in sorted(conversation_ids)
    )


@dataclass(frozen=True, slots=True)
class CoordinationScore:
    """Directional function-word coordination between two speaker roles.

    ``score`` is the conditional response rate: among source turns that use a
    category, the fraction of immediately following target turns that also use
    that category. Counts are retained so small samples are not mistaken for a
    reliable population estimate.
    """

    source: str
    target: str
    category: str
    score: float | None
    conditioned_turns: int
    coordinated_turns: int


def linguistic_coordination(
    conversation: Conversation,
    categories: Mapping[str, Iterable[str]],
) -> tuple[CoordinationScore, ...]:
    """Measure deterministic adjacent-turn linguistic coordination.

    Categories map names to function words (case-insensitive). Only adjacent
    utterances by different roles are considered; reply-tree traversal is not
    inferred from missing ``reply_to`` links. A ``None`` score means that no
    source turn used the category, preserving the distinction between no
    evidence and zero coordination.
    """

    if not isinstance(categories, Mapping) or not categories:
        raise ValueError("categories must be a non-empty mapping")
    normalized: dict[str, frozenset[str]] = {}
    for name, words in categories.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("category names must be non-empty strings")
        if name in normalized:
            raise ValueError(f"duplicate category {name!r}")
        terms = frozenset(
            token.casefold() for token in words if isinstance(token, str) and token.strip()
        )
        if not terms:
            raise ValueError(f"category {name!r} must contain at least one word")
        normalized[name] = terms

    roles = tuple(sorted({item.role for item in conversation.utterances}))
    counters: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for source_item, target_item in zip(
        conversation.utterances, conversation.utterances[1:], strict=False
    ):
        if source_item.role == target_item.role:
            continue
        source_tokens = set(_tokens(source_item.text))
        target_tokens = set(_tokens(target_item.text))
        for category, words in normalized.items():
            if not source_tokens & words:
                continue
            bucket = counters[(source_item.role, target_item.role, category)]
            bucket[0] += 1
            bucket[1] += int(bool(target_tokens & words))

    scores: list[CoordinationScore] = []
    for source_role in roles:
        for target_role in roles:
            if source_role == target_role:
                continue
            for category in normalized:
                conditioned, coordinated = counters[(source_role, target_role, category)]
                scores.append(
                    CoordinationScore(
                        source_role,
                        target_role,
                        category,
                        None if conditioned == 0 else coordinated / conditioned,
                        conditioned,
                        coordinated,
                    )
                )
    return tuple(scores)
