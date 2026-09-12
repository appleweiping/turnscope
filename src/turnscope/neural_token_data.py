"""Training-only, policy-pinned vocabulary and ordered integer sequence encoding."""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .neural_forecast_data import (
    _HARD,
    _HASH,
    _TOKEN,
    TOKENIZER_VERSION,
    ObservedPrefix,
    SequenceDataError,
    SequenceForecastDataset,
    SequenceLimitError,
    SequenceLimits,
    _digest,
    _integer,
    _iter_tokens,
    _limits,
    _object,
    _utf8_size,
)

PAD_ID = 0
UNK_ID = 1
EOS_ID = 2
VOCABULARY_FORMAT = "turnscope.sequence-vocabulary.v1"
ENCODING_FORMAT = "turnscope.encoded-prefix.v1"
MAX_TURN_TOKENS = 4096


def _token_policy(max_turn_tokens: Any, long_turn_policy: Any) -> None:
    _integer(max_turn_tokens, "max_turn_tokens", MAX_TURN_TOKENS, minimum=1)
    if type(long_turn_policy) is not str or long_turn_policy not in ("head", "reject"):
        raise SequenceDataError("long_turn_policy must be 'head' or 'reject'")


def _vocabulary_tokens(tokens: Any, frequencies: Any, documents: int) -> None:
    if (
        type(tokens) is not tuple
        or type(frequencies) is not tuple
        or len(tokens) != len(frequencies)
        or len(tokens) > _HARD["max_candidate_tokens"]
    ):
        raise SequenceDataError("vocabulary needs equally sized bounded tuples")
    size = 0
    previous: str | None = None
    for token, frequency in zip(tokens, frequencies, strict=True):
        size += _utf8_size(token, _HARD["max_token_bytes"], name="vocabulary token", empty=False)
        if size > _HARD["max_source_bytes"]:
            raise SequenceLimitError("vocabulary byte budget exceeded")
        # Casefold can introduce combining marks absent from the matched source
        # (e.g. U+0130). Removing only these marks here checks the token shape;
        # it does not alter the stored token or inference tokenization.
        skeleton = "".join(char for char in token if not unicodedata.category(char).startswith("M"))
        if token.casefold() != token or not _TOKEN.fullmatch(skeleton):
            raise SequenceDataError("vocabulary tokens must be casefolded tokenizer words")
        if previous is not None and token <= previous:
            raise SequenceDataError("vocabulary tokens must be sorted and distinct")
        previous = token
        _integer(frequency, "document frequency", documents, minimum=1)


@dataclass(frozen=True, slots=True)
class SequenceVocabulary:
    """Ordinary tokens only; three reserved IDs precede the lexical rows."""

    tokens: tuple[str, ...]
    document_frequencies: tuple[int, ...]
    documents: int
    max_turn_tokens: int = 128
    long_turn_policy: str = "head"
    tokenizer_version: str = TOKENIZER_VERSION
    _indices: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _integer(self.documents, "documents", _HARD["max_conversations"], minimum=1)
        _token_policy(self.max_turn_tokens, self.long_turn_policy)
        if type(self.tokenizer_version) is not str or self.tokenizer_version != TOKENIZER_VERSION:
            raise SequenceDataError("tokenizer/Unicode database version mismatch")
        _vocabulary_tokens(self.tokens, self.document_frequencies, self.documents)
        object.__setattr__(
            self,
            "_indices",
            MappingProxyType({token: index + 3 for index, token in enumerate(self.tokens)}),
        )

    @property
    def size(self) -> int:
        return len(self.tokens) + 3

    def token_id(self, token: str) -> int:
        _utf8_size(token, _HARD["max_token_bytes"], name="token")
        return self._indices.get(token, UNK_ID)

    def validate_limits(self, limits: SequenceLimits | None = None) -> None:
        bounds = _limits(limits)
        if (
            self.documents > bounds.max_conversations
            or len(self.tokens) > bounds.max_candidate_tokens
        ):
            raise SequenceLimitError("vocabulary inventory budget exceeded")
        size = 0
        for token in self.tokens:
            size += _utf8_size(token, bounds.max_token_bytes, name="vocabulary token")
            if size > bounds.max_source_bytes:
                raise SequenceLimitError("vocabulary byte budget exceeded")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": VOCABULARY_FORMAT,
            "tokens": list(self.tokens),
            "document_frequencies": list(self.document_frequencies),
            "documents": self.documents,
            "max_turn_tokens": self.max_turn_tokens,
            "long_turn_policy": self.long_turn_policy,
            "tokenizer_version": self.tokenizer_version,
        }

    @classmethod
    def from_dict(cls, value: Any) -> SequenceVocabulary:
        data = _object(
            value,
            {
                "format",
                "tokens",
                "document_frequencies",
                "documents",
                "max_turn_tokens",
                "long_turn_policy",
                "tokenizer_version",
            },
        )
        if type(data["format"]) is not str or data["format"] != VOCABULARY_FORMAT:
            raise SequenceDataError("unsupported vocabulary format")
        for name in ("tokens", "document_frequencies"):
            if type(data[name]) is not list or len(data[name]) > _HARD["max_candidate_tokens"]:
                raise SequenceDataError("serialized vocabulary requires bounded arrays")
        if len(data["tokens"]) != len(data["document_frequencies"]):
            raise SequenceDataError("vocabulary array sizes disagree")
        size = 0
        for token in data["tokens"]:
            size += _utf8_size(
                token, _HARD["max_token_bytes"], name="vocabulary token", empty=False
            )
            if size > _HARD["max_source_bytes"]:
                raise SequenceLimitError("vocabulary byte budget exceeded")
        return cls(
            tuple(data["tokens"]),
            tuple(data["document_frequencies"]),
            data["documents"],
            data["max_turn_tokens"],
            data["long_turn_policy"],
            data["tokenizer_version"],
        )

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())


def fit_sequence_vocabulary(
    dataset: SequenceForecastDataset,
    *,
    min_document_frequency: int = 1,
    max_features: int = 10_000,
    max_turn_tokens: int = 128,
    long_turn_policy: str = "head",
    limits: SequenceLimits | None = None,
) -> SequenceVocabulary:
    """Fit once on eligible training observations, never on individual prefix copies."""
    bounds = _limits(limits)
    _integer(
        min_document_frequency, "min_document_frequency", _HARD["max_conversations"], minimum=1
    )
    _integer(max_features, "max_features", bounds.max_candidate_tokens, minimum=1)
    _token_policy(max_turn_tokens, long_turn_policy)
    if not isinstance(dataset, SequenceForecastDataset):
        raise SequenceDataError("vocabulary fitting needs a SequenceForecastDataset")
    dataset.validate_limits(bounds)
    if not dataset.observations:
        raise SequenceDataError("vocabulary fitting requires an eligible conversation")
    frequencies: Counter[str] = Counter()
    candidates: set[str] = set()
    raw_tokens = 0
    for observation in dataset.observations:
        seen: set[str] = set()
        for turn in observation.turns:
            for position, token in enumerate(_iter_tokens(turn.text, bounds.max_token_bytes)):
                raw_tokens += 1
                if raw_tokens > bounds.max_tokens:
                    raise SequenceLimitError("sequence token budget exceeded")
                if position >= max_turn_tokens:
                    if long_turn_policy == "reject":
                        raise SequenceLimitError("turn exceeds the pinned token limit")
                    continue
                if token not in candidates:
                    if len(candidates) >= bounds.max_candidate_tokens:
                        raise SequenceLimitError("candidate vocabulary budget exceeded")
                    candidates.add(token)
                seen.add(token)
        frequencies.update(seen)
    selected = sorted(
        (token for token, frequency in frequencies.items() if frequency >= min_document_frequency),
        key=lambda token: (-frequencies[token], token),
    )[:max_features]
    tokens = tuple(sorted(selected))
    result = SequenceVocabulary(
        tokens,
        tuple(frequencies[token] for token in tokens),
        len(dataset.observations),
        max_turn_tokens,
        long_turn_policy,
    )
    result.validate_limits(bounds)
    return result


@dataclass(frozen=True, slots=True)
class EncodedPrefix:
    """Ordered lexical IDs plus EOS; padding is left to an explicit model batcher."""

    turns: tuple[tuple[int, ...], ...]
    source_digest: str
    vocabulary_digest: str
    vocabulary_size: int
    raw_tokens: int
    retained_tokens: int
    known_tokens: int
    truncated_turns: int

    def __post_init__(self) -> None:
        _integer(
            self.vocabulary_size, "vocabulary_size", _HARD["max_candidate_tokens"] + 3, minimum=3
        )
        for digest in (self.source_digest, self.vocabulary_digest):
            if type(digest) is not str or not _HASH.fullmatch(digest):
                raise SequenceDataError("encoding identities require lowercase SHA-256")
        if type(self.turns) is not tuple or not 1 <= len(self.turns) <= _HARD["max_observed_turns"]:
            raise SequenceDataError("encoded turns require a bounded non-empty tuple")
        for name in ("raw_tokens", "retained_tokens", "known_tokens"):
            _integer(getattr(self, name), name, _HARD["max_tokens"])
        _integer(self.truncated_turns, "truncated_turns", len(self.turns))
        if self.raw_tokens < self.retained_tokens or self.retained_tokens < self.known_tokens:
            raise SequenceDataError("encoding token accounting mismatch")
        if self.raw_tokens - self.retained_tokens < self.truncated_turns or (
            self.truncated_turns == 0 and self.raw_tokens != self.retained_tokens
        ):
            raise SequenceDataError("encoding truncation accounting mismatch")
        total = known = 0
        for turn in self.turns:
            if type(turn) is not tuple or not 1 <= len(turn) <= MAX_TURN_TOKENS + 1:
                raise SequenceDataError("each encoded turn needs bounded IDs and EOS")
            total += len(turn) - 1
            if total > self.retained_tokens:
                raise SequenceDataError("encoding token accounting mismatch")
            for index, token in enumerate(turn):
                _integer(token, "token ID", self.vocabulary_size - 1, minimum=1)
                if (token == EOS_ID) != (index == len(turn) - 1):
                    raise SequenceDataError("EOS must occur exactly at the end of each turn")
                known += token >= 3
        if (total, known) != (self.retained_tokens, self.known_tokens):
            raise SequenceDataError("encoding token accounting mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": ENCODING_FORMAT,
            "turns": [list(turn) for turn in self.turns],
            "source_digest": self.source_digest,
            "vocabulary_digest": self.vocabulary_digest,
            "vocabulary_size": self.vocabulary_size,
            "raw_tokens": self.raw_tokens,
            "retained_tokens": self.retained_tokens,
            "known_tokens": self.known_tokens,
            "truncated_turns": self.truncated_turns,
        }

    @classmethod
    def from_dict(cls, value: Any) -> EncodedPrefix:
        names = {
            "turns",
            "source_digest",
            "vocabulary_digest",
            "vocabulary_size",
            "raw_tokens",
            "retained_tokens",
            "known_tokens",
            "truncated_turns",
        }
        data = dict(_object(value, names | {"format"}))
        format_value = data.pop("format")
        if type(format_value) is not str or format_value != ENCODING_FORMAT:
            raise SequenceDataError("unsupported encoding format")
        turns = data["turns"]
        if type(turns) is not list or not 1 <= len(turns) <= _HARD["max_observed_turns"]:
            raise SequenceDataError("serialized encoding needs a bounded turn list")
        total = 0
        for turn in turns:
            if type(turn) is not list or not 1 <= len(turn) <= MAX_TURN_TOKENS + 1:
                raise SequenceDataError("serialized turn IDs must be a bounded list")
            total += len(turn) - 1
            if total > _HARD["max_tokens"]:
                raise SequenceLimitError("serialized encoding token budget exceeded")
        data["turns"] = tuple(tuple(turn) for turn in turns)
        return cls(**data)


def encode_observed_prefix(
    prefix: ObservedPrefix,
    vocabulary: SequenceVocabulary,
    *,
    limits: SequenceLimits | None = None,
) -> EncodedPrefix:
    """Encode complete observations with the frozen training token policy."""
    bounds = _limits(limits)
    if not isinstance(prefix, ObservedPrefix) or not isinstance(vocabulary, SequenceVocabulary):
        raise SequenceDataError("encoding needs ObservedPrefix and SequenceVocabulary")
    prefix.validate_limits(bounds)
    vocabulary.validate_limits(bounds)
    raw = retained = known = truncated = 0
    turns: list[tuple[int, ...]] = []
    for turn in prefix.turns:
        ids: list[int] = []
        count = 0
        for token in _iter_tokens(turn.text, bounds.max_token_bytes):
            count += 1
            raw += 1
            if raw > bounds.max_tokens:
                raise SequenceLimitError("sequence token budget exceeded")
            if count > vocabulary.max_turn_tokens:
                if vocabulary.long_turn_policy == "reject":
                    raise SequenceLimitError("turn exceeds the pinned token limit")
                continue
            token_id = vocabulary._indices.get(token, UNK_ID)
            ids.append(token_id)
            retained += 1
            known += token_id >= 3
        truncated += count > vocabulary.max_turn_tokens
        ids.append(EOS_ID)
        turns.append(tuple(ids))
    return EncodedPrefix(
        tuple(turns),
        prefix.digest,
        vocabulary.digest,
        vocabulary.size,
        raw,
        retained,
        known,
        truncated,
    )
