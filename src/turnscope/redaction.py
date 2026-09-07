"""Deterministic, reversible-by-policy text redaction for conversation exports."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from re import Pattern

from .models import Conversation, Utterance


@dataclass(frozen=True, slots=True)
class RedactionPolicy:
    """Named patterns enabled for a redaction pass."""

    email: bool = True
    phone: bool = True
    api_key: bool = True
    url: bool = False


@dataclass(frozen=True, slots=True)
class RedactionReport:
    """Counts and policy identity without retaining sensitive source values."""

    conversations: int
    utterances: int
    replacements: int
    by_kind: tuple[tuple[str, int], ...]
    policy: RedactionPolicy

    def to_dict(self) -> dict[str, object]:
        return {
            "conversations": self.conversations,
            "utterances": self.utterances,
            "replacements": self.replacements,
            "by_kind": dict(self.by_kind),
            "policy": {
                "email": self.policy.email,
                "phone": self.policy.phone,
                "api_key": self.policy.api_key,
                "url": self.policy.url,
            },
        }


_PATTERNS: dict[str, Pattern[str]] = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "phone": re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)"),
    "api_key": re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9_]{16,})\b"),
    "url": re.compile(r"\bhttps?://[^\s<>]+", re.IGNORECASE),
}


def redact_conversations(
    conversations: Iterable[Conversation], *, policy: RedactionPolicy | None = None
) -> tuple[tuple[Conversation, ...], RedactionReport]:
    """Return redacted immutable conversations and an aggregate report.

    Replacement labels contain only a short SHA-256 digest of the matched
    value. This keeps repeated identifiers linkable without writing source
    secrets to the report or logs.
    """

    active = policy or RedactionPolicy()
    enabled = tuple(name for name in ("email", "phone", "api_key", "url") if getattr(active, name))
    counts: Counter[str] = Counter()
    output: list[Conversation] = []
    utterance_count = 0
    for conversation in conversations:
        redacted: list[Utterance] = []
        for utterance in conversation.utterances:
            text = utterance.text
            for kind in enabled:
                pattern = _PATTERNS[kind]

                def replace(match: re.Match[str], *, _kind: str = kind) -> str:
                    value = match.group(0)
                    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
                    counts[_kind] += 1
                    return f"[REDACTED:{_kind}:{digest}]"

                text = pattern.sub(replace, text)
            redacted.append(
                Utterance(
                    id=utterance.id,
                    role=utterance.role,
                    text=text,
                    timestamp=utterance.timestamp,
                    reply_to=utterance.reply_to,
                    token_count=utterance.token_count,
                    metadata=utterance.metadata,
                )
            )
        utterance_count += len(redacted)
        output.append(Conversation(conversation.id, redacted, metadata=conversation.metadata))
    report = RedactionReport(
        conversations=len(output),
        utterances=utterance_count,
        replacements=sum(counts.values()),
        by_kind=tuple(sorted(counts.items())),
        policy=active,
    )
    return tuple(output), report


def redact_stream(
    conversations: Iterable[Conversation], *, policy: RedactionPolicy | None = None
) -> Iterator[Conversation]:
    """Yield redacted conversations when a report is not needed."""

    redacted, _ = redact_conversations(conversations, policy=policy)
    yield from redacted
