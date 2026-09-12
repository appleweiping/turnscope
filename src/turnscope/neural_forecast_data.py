"""Bounded ordered first-future-event data, without a numerical dependency."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Any

from .models import Conversation, Utterance

TOKENIZER_VERSION = f"turnscope.regex-casefold.v1/ucd-{unicodedata.unidata_version}"
SEQUENCE_DATA_FORMAT = "turnscope.sequence-forecast-data.v1"
_TOKEN = re.compile(r"[\w]+(?:['-][\w]+)*", re.UNICODE)
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN_BYTE_HARD_LIMIT = 4096
_HARD = {
    "max_conversations": 100_000,
    "max_source_turns": 1_000_000,
    "max_observed_turns": 4096,
    "max_prefixes": 100_000,
    "max_source_bytes": 128 * 1024 * 1024,
    "max_turn_bytes": 8 * 1024 * 1024,
    "max_identifier_bytes": 4096,
    "max_groups": 400_000,
    "max_groups_per_conversation": 1024,
    "max_tokens": 10_000_000,
    "max_candidate_tokens": 500_000,
    "max_token_bytes": _TOKEN_BYTE_HARD_LIMIT,
}


class SequenceDataError(ValueError):
    """An invalid ordered sequence or declared supervision was supplied."""


class SequenceLimitError(SequenceDataError):
    """A declared source or processing budget was exceeded, not a model score."""


def _integer(value: Any, name: str, maximum: int, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise SequenceDataError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _utf8_size(value: Any, maximum: int, *, name: str, empty: bool = True) -> int:
    if type(value) is not str:
        raise SequenceDataError(f"{name} must be Unicode text")
    if len(value) > maximum:
        raise SequenceLimitError(f"{name} exceeds its byte limit")
    if not empty and not value.strip():
        raise SequenceDataError(f"{name} must not be blank")
    total = 0
    try:
        for offset in range(0, len(value), 4096):
            total += len(value[offset : offset + 4096].encode("utf-8"))
            if total > maximum:
                raise SequenceLimitError(f"{name} exceeds its byte limit")
    except UnicodeError as error:
        raise SequenceDataError(f"{name} contains invalid Unicode") from error
    return total


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise SequenceDataError("observation timestamps must be timezone-aware datetimes")
    try:
        if value.utcoffset() is None:
            raise SequenceDataError("observation timestamps must have a UTC offset")
        return value.astimezone(timezone.utc)
    except (OverflowError, OSError, ValueError) as error:
        raise SequenceDataError("observation timestamp is outside the UTC range") from error


def _object(value: Any, names: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or len(value) != len(names) or set(value) != names:
        raise SequenceDataError("sequence object has missing or unknown fields")
    return value


def _digest(value: Any) -> str:
    """Hash a validated bounded value without joining the complete JSON encoding."""
    result = hashlib.sha256()
    encoder = json.JSONEncoder(
        ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    for fragment in encoder.iterencode(value):
        for offset in range(0, len(fragment), 4096):
            result.update(fragment[offset : offset + 4096].encode("utf-8"))
    return result.hexdigest()


def _group_digest(kind: str, identifier: str) -> str:
    # Match the existing lexical workflow's namespace, never hash bare IDs.
    return hashlib.sha256((kind + "\0" + identifier).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SequencePolicy:
    event_field: str = "event"
    skip_field: str = "is_section_header"
    groups_field: str = "forecast_groups"
    min_turns: int = 2

    def __post_init__(self) -> None:
        for name in ("event_field", "skip_field", "groups_field"):
            _utf8_size(getattr(self, name), 256, name=name, empty=False)
        if self.event_field == self.skip_field:
            raise SequenceDataError("event_field and skip_field must differ")
        _integer(self.min_turns, "min_turns", _HARD["max_observed_turns"], minimum=1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> SequencePolicy:
        return cls(**dict(_object(value, {field.name for field in fields(cls)})))


@dataclass(frozen=True, slots=True)
class SequenceLimits:
    max_conversations: int = 10_000
    max_source_turns: int = 100_000
    max_observed_turns: int = 64
    max_prefixes: int = 30_000
    max_source_bytes: int = 32 * 1024 * 1024
    max_turn_bytes: int = 4 * 1024 * 1024
    max_identifier_bytes: int = 1024
    max_groups: int = 40_000
    max_groups_per_conversation: int = 128
    max_tokens: int = 2_000_000
    max_candidate_tokens: int = 100_000
    max_token_bytes: int = 1024

    def __post_init__(self) -> None:
        for name, maximum in _HARD.items():
            _integer(getattr(self, name), name, maximum, minimum=1)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> SequenceLimits:
        return cls(**dict(_object(value, set(_HARD))))


def _limits(value: SequenceLimits | None) -> SequenceLimits:
    if value is None:
        return SequenceLimits()
    if not isinstance(value, SequenceLimits):
        raise SequenceDataError("limits must be SequenceLimits")
    return value


def _policy(value: SequencePolicy | None, limits: SequenceLimits) -> SequencePolicy:
    if value is None:
        value = SequencePolicy()
    if not isinstance(value, SequencePolicy):
        raise SequenceDataError("policy must be SequencePolicy")
    if value.min_turns > limits.max_observed_turns:
        raise SequenceLimitError("min_turns exceeds the observed-turn budget")
    return value


def _iter_tokens(text: str, maximum: int) -> Iterator[str]:
    for match in _TOKEN.finditer(text):
        if match.end() - match.start() > maximum:
            raise SequenceLimitError("sequence token exceeds its byte limit")
        token = match.group().casefold()
        _utf8_size(token, maximum, name="sequence token", empty=False)
        yield token


@dataclass(frozen=True, slots=True)
class ObservedTurn:
    """One complete observation; no role, reply, outcome or arbitrary metadata."""

    id: str
    timestamp: datetime
    text: str

    def __post_init__(self) -> None:
        _utf8_size(self.id, _HARD["max_identifier_bytes"], name="utterance ID", empty=False)
        _utf8_size(self.text, _HARD["max_turn_bytes"], name="utterance text")
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp))

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "timestamp": self.timestamp.isoformat(), "text": self.text}

    @classmethod
    def from_dict(cls, value: Any) -> ObservedTurn:
        data = _object(value, {"id", "timestamp", "text"})
        raw = data["timestamp"]
        if type(raw) is not str or len(raw) > 64:
            raise SequenceDataError("serialized timestamp must be ISO-8601 text")
        try:
            stamp = datetime.fromisoformat(raw)
        except ValueError as error:
            raise SequenceDataError("serialized timestamp must be ISO-8601 text") from error
        return cls(data["id"], stamp, data["text"])


def _prefix_counts(prefix: ObservedPrefix, limits: SequenceLimits) -> tuple[int, int, int]:
    if len(prefix.turns) > limits.max_observed_turns:
        raise SequenceLimitError("observed-turn budget exceeded")
    identities = _utf8_size(
        prefix.conversation_id, limits.max_identifier_bytes, name="conversation ID", empty=False
    )
    text_bytes = tokens = 0
    for turn in prefix.turns:
        identities += _utf8_size(
            turn.id, limits.max_identifier_bytes, name="utterance ID", empty=False
        )
        text_bytes += _utf8_size(turn.text, limits.max_turn_bytes, name="utterance text")
        if text_bytes + identities > limits.max_source_bytes:
            raise SequenceLimitError("observation source-byte budget exceeded")
        for _token in _iter_tokens(turn.text, limits.max_token_bytes):
            tokens += 1
            if tokens > limits.max_tokens:
                raise SequenceLimitError("sequence token budget exceeded")
    return text_bytes, identities, tokens


def _raw_observation_admission(values: list[Any]) -> None:
    """Check aggregate raw inventories before constructing copied turn catalogs."""
    turns = size = 0
    for value in values:
        data = _object(value, {"conversation_id", "turns"})
        size += _utf8_size(
            data["conversation_id"],
            _HARD["max_identifier_bytes"],
            name="conversation ID",
            empty=False,
        )
        items = data["turns"]
        if type(items) is not list or not 1 <= len(items) <= _HARD["max_observed_turns"]:
            raise SequenceDataError("serialized observations need a bounded turn list")
        turns += len(items)
        if turns > _HARD["max_source_turns"]:
            raise SequenceLimitError("serialized observation turn budget exceeded")
        for item in items:
            row = _object(item, {"id", "timestamp", "text"})
            size += _utf8_size(
                row["id"], _HARD["max_identifier_bytes"], name="utterance ID", empty=False
            )
            size += _utf8_size(row["text"], _HARD["max_turn_bytes"], name="utterance text")
            if size > _HARD["max_source_bytes"]:
                raise SequenceLimitError("serialized observation source-byte budget exceeded")


@dataclass(frozen=True, slots=True)
class ObservedPrefix:
    """Ordered completed turns supplied by a caller, with no future annotations."""

    conversation_id: str
    turns: tuple[ObservedTurn, ...]

    def __post_init__(self) -> None:
        size = _utf8_size(
            self.conversation_id, _HARD["max_identifier_bytes"], name="conversation ID", empty=False
        )
        if type(self.turns) is not tuple or not 1 <= len(self.turns) <= _HARD["max_observed_turns"]:
            raise SequenceDataError("observed turns must be a non-empty bounded tuple")
        seen: set[str] = set()
        previous: datetime | None = None
        for turn in self.turns:
            if not isinstance(turn, ObservedTurn):
                raise SequenceDataError("observed turns must contain ObservedTurn values")
            if turn.id in seen:
                raise SequenceDataError("observed utterance IDs must be unique")
            seen.add(turn.id)
            if previous is not None and turn.timestamp < previous:
                raise SequenceDataError("observation timestamps must be nondecreasing")
            previous = turn.timestamp
            size += _utf8_size(
                turn.id, _HARD["max_identifier_bytes"], name="utterance ID", empty=False
            )
            size += _utf8_size(turn.text, _HARD["max_turn_bytes"], name="utterance text")
            if size > _HARD["max_source_bytes"]:
                raise SequenceLimitError("observation source-byte budget exceeded")

    def validate_limits(self, limits: SequenceLimits | None = None) -> None:
        _prefix_counts(self, _limits(limits))

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turns": [turn.to_dict() for turn in self.turns],
        }

    @classmethod
    def from_dict(cls, value: Any) -> ObservedPrefix:
        _raw_observation_admission([value])
        data = _object(value, {"conversation_id", "turns"})
        turns = data["turns"]
        if type(turns) is not list or not 1 <= len(turns) <= _HARD["max_observed_turns"]:
            raise SequenceDataError("serialized observations need a bounded turn list")
        return cls(data["conversation_id"], tuple(ObservedTurn.from_dict(turn) for turn in turns))

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class SequenceForecastExample:
    """Compact supervision; endpoint is inclusive and zero-based."""

    conversation_index: int
    endpoint: int
    label: bool
    lead_turns: int | None

    def __post_init__(self) -> None:
        _integer(self.conversation_index, "conversation_index", _HARD["max_conversations"] - 1)
        _integer(self.endpoint, "endpoint", _HARD["max_observed_turns"] - 1)
        if type(self.label) is not bool:
            raise SequenceDataError("forecast label must be boolean")
        if self.label:
            _integer(self.lead_turns, "lead_turns", _HARD["max_source_turns"], minimum=1)
        elif self.lead_turns is not None:
            raise SequenceDataError("negative examples cannot have a lead time")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> SequenceForecastExample:
        return cls(**dict(_object(value, {field.name for field in fields(cls)})))


@dataclass(frozen=True, slots=True)
class SequenceAudit:
    source_conversations: int
    source_turns: int
    source_text_bytes: int
    source_identity_bytes: int
    header_turns: int
    eligible_conversations: int
    excluded_conversations: int
    observed_turns: int
    observed_text_bytes: int
    raw_tokens: int
    prefixes: int
    positive_conversations: int
    negative_conversations: int

    def __post_init__(self) -> None:
        for field in fields(self):
            _integer(getattr(self, field.name), field.name, _HARD["max_source_bytes"])
        if self.source_conversations != self.eligible_conversations + self.excluded_conversations:
            raise SequenceDataError("source conversation accounting mismatch")
        if self.eligible_conversations != self.positive_conversations + self.negative_conversations:
            raise SequenceDataError("eligible class accounting mismatch")
        if self.source_turns < self.header_turns + self.observed_turns:
            raise SequenceDataError("source turn accounting mismatch")
        if self.source_text_bytes < self.observed_text_bytes:
            raise SequenceDataError("source text accounting mismatch")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> SequenceAudit:
        return cls(**dict(_object(value, {field.name for field in fields(cls)})))


@dataclass(frozen=True, slots=True)
class SequenceForecastDataset:
    observations: tuple[ObservedPrefix, ...]
    examples: tuple[SequenceForecastExample, ...]
    group_digests: frozenset[str]
    audit: SequenceAudit

    def __post_init__(self) -> None:
        if (
            type(self.observations) is not tuple
            or len(self.observations) > _HARD["max_conversations"]
        ):
            raise SequenceDataError("observations must be a bounded tuple")
        if type(self.examples) is not tuple or len(self.examples) > _HARD["max_prefixes"]:
            raise SequenceDataError("examples must be a bounded tuple")
        if (
            type(self.group_digests) is not frozenset
            or len(self.group_digests) > _HARD["max_groups"]
        ):
            raise SequenceDataError("group digests must be a bounded frozenset")
        if any(
            type(value) is not str or not _HASH.fullmatch(value) for value in self.group_digests
        ):
            raise SequenceDataError("group identity requires lowercase SHA-256 digests")
        if not isinstance(self.audit, SequenceAudit):
            raise SequenceDataError("audit must be SequenceAudit")
        previous_id: str | None = None
        for observation in self.observations:
            if not isinstance(observation, ObservedPrefix):
                raise SequenceDataError("observation catalog requires ObservedPrefix values")
            if previous_id is not None and observation.conversation_id <= previous_id:
                raise SequenceDataError("observation catalog must have canonical unique IDs")
            previous_id = observation.conversation_id
            if _group_digest("conversation", observation.conversation_id) not in self.group_digests:
                raise SequenceDataError("observed conversation identity is missing from groups")
        last: dict[int, SequenceForecastExample] = {}
        prior_key = (-1, -1)
        for example in self.examples:
            if not isinstance(example, SequenceForecastExample):
                raise SequenceDataError("example catalog requires SequenceForecastExample values")
            key = (example.conversation_index, example.endpoint)
            if key <= prior_key or example.conversation_index >= len(self.observations):
                raise SequenceDataError("example handles must be canonical, unique and in range")
            prior_key = key
            observation = self.observations[example.conversation_index]
            if example.endpoint >= len(observation.turns):
                raise SequenceDataError("example endpoint is outside its observation")
            if (
                example.endpoint + 1 < len(observation.turns)
                and observation.turns[example.endpoint].timestamp
                == observation.turns[example.endpoint + 1].timestamp
            ):
                raise SequenceDataError("example endpoint splits an equal-time block")
            earlier = last.get(example.conversation_index)
            if earlier is not None and (
                earlier.label != example.label
                or (
                    example.label
                    and earlier.endpoint + int(earlier.lead_turns or 0)
                    != example.endpoint + int(example.lead_turns or 0)
                )
            ):
                raise SequenceDataError("conversation examples disagree on their future event")
            if example.label and example.endpoint + int(example.lead_turns or 0) < len(
                observation.turns
            ):
                raise SequenceDataError("future event is inside the observation catalog")
            last[example.conversation_index] = example
        if set(last) != set(range(len(self.observations))):
            raise SequenceDataError("every observation must have eligible examples")
        if any(
            last[index].endpoint != len(observation.turns) - 1
            for index, observation in enumerate(self.observations)
        ):
            raise SequenceDataError("observations must stop at their last eligible endpoint")
        if (
            self.audit.eligible_conversations,
            self.audit.prefixes,
            self.audit.positive_conversations,
        ) != (
            len(self.observations),
            len(self.examples),
            sum(example.label for example in last.values()),
        ):
            raise SequenceDataError("sequence audit does not match its examples")
        self.validate_limits(SequenceLimits(**_HARD))

    def validate_limits(self, limits: SequenceLimits | None = None) -> None:
        bounds = _limits(limits)
        audit = self.audit
        if (
            audit.source_conversations > bounds.max_conversations
            or audit.source_turns > bounds.max_source_turns
        ):
            raise SequenceLimitError("sequence source inventory budget exceeded")
        if len(self.examples) > bounds.max_prefixes or len(self.group_digests) > bounds.max_groups:
            raise SequenceLimitError("sequence example or group budget exceeded")
        if audit.source_text_bytes + audit.source_identity_bytes > bounds.max_source_bytes:
            raise SequenceLimitError("sequence source-byte budget exceeded")
        inventory_turns = sum(len(observation.turns) for observation in self.observations)
        if inventory_turns > bounds.max_source_turns:
            raise SequenceLimitError("sequence observation turn budget exceeded")
        if inventory_turns != audit.observed_turns:
            raise SequenceDataError("sequence audit does not match its observation catalog")
        turns = text_bytes = tokens = identities = 0
        for observation in self.observations:
            size, identity_size, count = _prefix_counts(observation, bounds)
            turns += len(observation.turns)
            text_bytes += size
            identities += identity_size
            tokens += count
            if text_bytes + identities > bounds.max_source_bytes:
                raise SequenceLimitError("sequence observation source-byte budget exceeded")
            if tokens > bounds.max_tokens:
                raise SequenceLimitError("sequence token budget exceeded")
        if (turns, text_bytes, tokens) != (
            audit.observed_turns,
            audit.observed_text_bytes,
            audit.raw_tokens,
        ) or identities > audit.source_identity_bytes:
            raise SequenceDataError("sequence audit does not match its observation catalog")

    def prefix(self, example: SequenceForecastExample) -> ObservedPrefix:
        if not isinstance(example, SequenceForecastExample):
            raise SequenceDataError("example is not a member of this dataset")
        key = (example.conversation_index, example.endpoint)
        lower, upper = 0, len(self.examples)
        while lower < upper:
            middle = (lower + upper) // 2
            candidate = self.examples[middle]
            if (candidate.conversation_index, candidate.endpoint) < key:
                lower = middle + 1
            else:
                upper = middle
        if lower == len(self.examples) or self.examples[lower] != example:
            raise SequenceDataError("example is not a member of this dataset")
        source = self.observations[example.conversation_index]
        return ObservedPrefix(source.conversation_id, source.turns[: example.endpoint + 1])

    @property
    def weights(self) -> tuple[float, ...]:
        counts = Counter(example.conversation_index for example in self.examples)
        return tuple(1 / counts[example.conversation_index] for example in self.examples)

    @property
    def observation_digest(self) -> str:
        return _digest([observation.to_dict() for observation in self.observations])

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": SEQUENCE_DATA_FORMAT,
            "observations": [item.to_dict() for item in self.observations],
            "examples": [item.to_dict() for item in self.examples],
            "group_digests": sorted(self.group_digests),
            "audit": self.audit.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> SequenceForecastDataset:
        data = _object(value, {"format", "observations", "examples", "group_digests", "audit"})
        if type(data["format"]) is not str or data["format"] != SEQUENCE_DATA_FORMAT:
            raise SequenceDataError("unsupported sequence dataset format")
        for name, bound in (
            ("observations", "max_conversations"),
            ("examples", "max_prefixes"),
            ("group_digests", "max_groups"),
        ):
            if type(data[name]) is not list or len(data[name]) > _HARD[bound]:
                raise SequenceDataError("serialized sequence arrays exceed their inventory bound")
        groups = data["group_digests"]
        if any(type(item) is not str for item in groups) or groups != sorted(set(groups)):
            raise SequenceDataError("serialized groups must be sorted and distinct")
        audit = SequenceAudit.from_dict(data["audit"])
        _raw_observation_admission(data["observations"])
        return cls(
            tuple(ObservedPrefix.from_dict(item) for item in data["observations"]),
            tuple(SequenceForecastExample.from_dict(item) for item in data["examples"]),
            frozenset(groups),
            audit,
        )

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())


@dataclass
class _Admission:
    limits: SequenceLimits
    conversations: int = 0
    turns: int = 0
    text_bytes: int = 0
    identity_bytes: int = 0
    headers: int = 0

    def identity(self, value: Any, name: str) -> None:
        self.identity_bytes += _utf8_size(
            value, self.limits.max_identifier_bytes, name=name, empty=False
        )
        self.check_bytes()

    def check_bytes(self) -> None:
        if self.text_bytes + self.identity_bytes > self.limits.max_source_bytes:
            raise SequenceLimitError("sequence source-byte budget exceeded")

    def conversation(self, value: Any) -> Conversation:
        if not isinstance(value, Conversation):
            raise SequenceDataError("sequence source must contain Conversation values")
        self.conversations += 1
        if self.conversations > self.limits.max_conversations:
            raise SequenceLimitError("sequence conversation budget exceeded")
        self.identity(value.id, "conversation ID")
        if not isinstance(value.metadata, Mapping):
            raise SequenceDataError("conversation metadata must be a mapping")
        if (
            type(value.utterances) is not tuple
            or len(value.utterances) > self.limits.max_source_turns - self.turns
        ):
            raise SequenceLimitError("sequence source-turn budget exceeded")
        return value

    def observations(
        self, conversation: Conversation, policy: SequencePolicy, *, supervised: bool
    ) -> list[tuple[Utterance, bool]]:
        result: list[tuple[Utterance, bool]] = []
        previous: datetime | None = None
        seen: set[str] = set()
        for turn in conversation.utterances:
            if not isinstance(turn, Utterance):
                raise SequenceDataError("sequence source must contain Utterance values")
            self.turns += 1
            self.identity(turn.id, "utterance ID")
            self.text_bytes += _utf8_size(
                turn.text, self.limits.max_turn_bytes, name="utterance text"
            )
            self.check_bytes()
            stamp = _timestamp(turn.timestamp)
            if turn.id in seen:
                raise SequenceDataError("source utterance IDs must be unique within a conversation")
            seen.add(turn.id)
            if previous is not None and stamp < previous:
                raise SequenceDataError("source timestamps must be nondecreasing")
            previous = stamp
            if not isinstance(turn.metadata, Mapping):
                raise SequenceDataError("utterance metadata must be a mapping")
            skip = turn.metadata.get(policy.skip_field, False)
            if type(skip) is not bool:
                raise SequenceDataError("non-observation mask must be boolean")
            if skip:
                self.headers += 1
                continue
            event = turn.metadata.get(policy.event_field) if supervised else False
            if type(event) is not bool:
                raise SequenceDataError("observed source events must be boolean")
            result.append((turn, event))
        return result


def observed_prefix(
    conversation: Conversation,
    *,
    policy: SequencePolicy | None = None,
    limits: SequenceLimits | None = None,
) -> ObservedPrefix:
    """Apply only an explicit header mask; never read outcomes or forecast groups."""
    bounds = _limits(limits)
    rule = _policy(policy, bounds)
    admission = _Admission(bounds)
    source = admission.conversation(conversation)
    observed = admission.observations(source, rule, supervised=False)
    if len(observed) < rule.min_turns:
        raise SequenceDataError("observed prefix is shorter than min_turns")
    if len(observed) > bounds.max_observed_turns:
        raise SequenceLimitError("observed-turn budget exceeded")
    result = ObservedPrefix(
        source.id, tuple(ObservedTurn(turn.id, turn.timestamp, turn.text) for turn, _ in observed)
    )
    result.validate_limits(bounds)
    return result


def prepare_sequence_forecasts(
    conversations: Iterable[Conversation],
    *,
    policy: SequencePolicy | None = None,
    limits: SequenceLimits | None = None,
) -> SequenceForecastDataset:
    """Prepare canonical compact handles strictly before the first future event."""
    bounds = _limits(limits)
    rule = _policy(policy, bounds)
    if not isinstance(conversations, Iterable):
        raise SequenceDataError("sequence source must be iterable")
    admission = _Admission(bounds)
    seen: set[str] = set()
    groups: set[str] = set()
    selected: list[tuple[ObservedPrefix, list[tuple[int, bool, int | None]]]] = []
    prefixes = total_tokens = observed_turns = observed_text = positives = 0
    for value in conversations:
        conversation = admission.conversation(value)
        if conversation.id in seen:
            raise SequenceDataError("source conversation IDs must be unique")
        seen.add(conversation.id)
        raw_groups = conversation.metadata.get(rule.groups_field)
        if (
            type(raw_groups) is not list
            or not 1 <= len(raw_groups) <= bounds.max_groups_per_conversation
        ):
            raise SequenceDataError("forecast groups must be a non-empty bounded list")
        local: set[str] = set()
        for group in raw_groups:
            if type(group) is not str:
                raise SequenceDataError("forecast groups must contain Unicode strings")
            admission.identity(group, "forecast group")
            if group in local:
                raise SequenceDataError("forecast groups must be distinct")
            local.add(group)
            groups.add(_group_digest("group", group))
            if len(groups) > bounds.max_groups:
                raise SequenceLimitError("sequence group budget exceeded")
        groups.add(_group_digest("conversation", conversation.id))
        if len(groups) > bounds.max_groups:
            raise SequenceLimitError("sequence group budget exceeded")
        observed = admission.observations(conversation, rule, supervised=True)
        first_event = next((index for index, (_turn, event) in enumerate(observed) if event), None)
        stop = first_event if first_event is not None else len(observed) - 1
        handles: list[tuple[int, bool, int | None]] = []
        for index in range(max(stop, 0)):
            turn = observed[index][0]
            if index + 1 < rule.min_turns or turn.timestamp == observed[index + 1][0].timestamp:
                continue
            if first_event is not None and turn.timestamp >= observed[first_event][0].timestamp:
                continue
            if prefixes + len(handles) >= bounds.max_prefixes:
                raise SequenceLimitError("sequence prefix budget exceeded")
            if index + 1 > bounds.max_observed_turns:
                raise SequenceLimitError("observed-turn budget exceeded")
            handles.append(
                (
                    index,
                    first_event is not None,
                    first_event - index if first_event is not None else None,
                )
            )
        if not handles:
            continue
        endpoint = handles[-1][0]
        turns = tuple(
            ObservedTurn(turn.id, turn.timestamp, turn.text) for turn, _ in observed[: endpoint + 1]
        )
        prefix = ObservedPrefix(conversation.id, turns)
        size, _identities, token_count = _prefix_counts(prefix, bounds)
        total_tokens += token_count
        if total_tokens > bounds.max_tokens:
            raise SequenceLimitError("sequence token budget exceeded")
        observed_text += size
        observed_turns += len(turns)
        prefixes += len(handles)
        positives += first_event is not None
        selected.append((prefix, handles))
    selected.sort(key=lambda item: item[0].conversation_id)
    examples = tuple(
        SequenceForecastExample(index, endpoint, label, lead)
        for index, (_observation, handles) in enumerate(selected)
        for endpoint, label, lead in handles
    )
    audit = SequenceAudit(
        admission.conversations,
        admission.turns,
        admission.text_bytes,
        admission.identity_bytes,
        admission.headers,
        len(selected),
        admission.conversations - len(selected),
        observed_turns,
        observed_text,
        total_tokens,
        prefixes,
        positives,
        len(selected) - positives,
    )
    result = SequenceForecastDataset(
        tuple(observation for observation, _handles in selected), examples, frozenset(groups), audit
    )
    result.validate_limits(bounds)
    return result
