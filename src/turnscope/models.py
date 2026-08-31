"""Validated, frozen domain records used throughout TurnScope."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, OSError) as error:
        raise ValueError("timestamp is outside the supported UTC range") from error


@dataclass(frozen=True, slots=True)
class Utterance:
    """One message in a conversation.

    ``reply_to`` identifies another utterance in the same conversation. Metadata
    is copied to prevent callers from mutating a validated object indirectly.
    """

    id: str
    role: str
    text: str
    timestamp: datetime
    reply_to: str | None = None
    token_count: int | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("utterance id must not be empty")
        if not self.role.strip():
            raise ValueError("utterance role must not be empty")
        if self.token_count is not None and (
            isinstance(self.token_count, bool)
            or not isinstance(self.token_count, int)
            or self.token_count < 0
        ):
            raise ValueError("token_count must be a non-negative integer")
        object.__setattr__(self, "timestamp", _utc(self.timestamp))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class Conversation:
    """An ordered input sequence with a stable conversation identifier."""

    id: str
    utterances: tuple[Utterance, ...]
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __init__(
        self,
        id: str,
        utterances: Sequence[Utterance],
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> None:
        if not id.strip():
            raise ValueError("conversation id must not be empty")
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "utterances", tuple(utterances))
        object.__setattr__(self, "metadata", dict(metadata or {}))

    def by_id(self) -> dict[str, Utterance]:
        """Return the last utterance for each ID without changing input order."""
        return {item.id: item for item in self.utterances}


@dataclass(frozen=True, slots=True)
class ContextWindow:
    """The context selected for one target utterance."""

    conversation_id: str
    target: Utterance
    context: tuple[Utterance, ...]
    policy: str
    token_total: int

    @property
    def utterance_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.context)


class Severity(IntEnum):
    """Ordered issue severity used by reports and CLI exit thresholds."""

    INFO = 10
    WARNING = 20
    ERROR = 30

    @classmethod
    def parse(cls, value: str) -> Severity:
        try:
            return cls[value.upper()]
        except KeyError as error:
            choices = ", ".join(item.name.lower() for item in cls)
            raise ValueError(f"unknown severity {value!r}; choose one of: {choices}") from error


@dataclass(frozen=True, slots=True)
class Issue:
    """A machine-readable reliability finding."""

    rule: str
    severity: Severity
    message: str
    conversation_id: str
    utterance_ids: tuple[str, ...] = ()
    details: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", dict(self.details))


@dataclass(frozen=True, slots=True)
class AuditReport:
    """All findings for one or more conversations."""

    issues: tuple[Issue, ...]
    conversations: int
    utterances: int

    def counts(self) -> dict[str, int]:
        result = {item.name.lower(): 0 for item in Severity}
        for issue in self.issues:
            result[issue.severity.name.lower()] += 1
        return result

    def failing(self, threshold: Severity = Severity.ERROR) -> bool:
        return any(issue.severity >= threshold for issue in self.issues)

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "conversations": self.conversations,
                "utterances": self.utterances,
                "issues": len(self.issues),
                "by_severity": self.counts(),
            },
            "issues": [
                {
                    "rule": item.rule,
                    "severity": item.severity.name.lower(),
                    "message": item.message,
                    "conversation_id": item.conversation_id,
                    "utterance_ids": list(item.utterance_ids),
                    "details": dict(item.details),
                }
                for item in self.issues
            ],
        }
