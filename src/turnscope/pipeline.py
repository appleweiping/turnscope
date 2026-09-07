"""Composable fit/transform feature pipelines for conversation analysis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Protocol, runtime_checkable

from .models import Conversation


@runtime_checkable
class ConversationTransformer(Protocol):
    """Minimal scikit-learn-like protocol for conversation feature steps."""

    def transform(self, conversation: Conversation) -> Any:
        """Produce one feature block for a conversation."""


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    """One immutable pipeline output with named feature blocks."""

    conversation_id: str
    blocks: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.conversation_id, str) or not self.conversation_id:
            raise ValueError("feature record conversation_id must be non-empty")
        if not isinstance(self.blocks, Mapping) or not all(
            isinstance(name, str) and name for name in self.blocks
        ):
            raise TypeError("feature blocks must be a mapping of non-empty names")
        object.__setattr__(self, "blocks", dict(self.blocks))

    def digest(self) -> str:
        """Return a stable digest suitable for a cache key."""
        payload = {
            "conversation_id": self.conversation_id,
            "blocks": _jsonable(self.blocks),
        }
        try:
            encoded = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError("feature blocks must be JSON-serializable") from error
        return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert frozen mappings and dataclass feature values to JSON values."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class CallableTransformer:
    """Adapter for a pure feature function."""

    function: Callable[[Conversation], Any]

    def transform(self, conversation: Conversation) -> Any:
        return self.function(conversation)


class FeaturePipeline:
    """Ordered feature steps with explicit fitting and deterministic outputs."""

    def __init__(self, steps: Sequence[tuple[str, ConversationTransformer]]) -> None:
        if not isinstance(steps, Sequence) or not steps:
            raise ValueError("feature pipeline requires at least one step")
        names = [name for name, _ in steps]
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("feature step names must be non-empty strings")
        if len(names) != len(set(names)):
            raise ValueError("feature step names must be unique")
        if not all(isinstance(transformer, ConversationTransformer) for _, transformer in steps):
            raise TypeError("feature steps must implement transform")
        self._steps = tuple(steps)
        self._fitted = False

    @property
    def step_names(self) -> tuple[str, ...]:
        """Return declared feature step names in execution order."""
        return tuple(name for name, _ in self._steps)

    @property
    def fitted(self) -> bool:
        """Whether every fit-capable step has been fitted."""
        return self._fitted

    def fit(self, conversations: Iterable[Conversation]) -> FeaturePipeline:
        """Fit all steps that expose a callable ``fit`` method."""
        materialized = tuple(conversations)
        if not materialized:
            raise ValueError("at least one conversation is required to fit a pipeline")
        if not all(isinstance(item, Conversation) for item in materialized):
            raise TypeError("pipeline input must contain Conversation values")
        if len({item.id for item in materialized}) != len(materialized):
            raise ValueError("pipeline conversations must have unique IDs")
        for _, transformer in self._steps:
            fit = getattr(transformer, "fit", None)
            if callable(fit):
                fit(materialized)
        self._fitted = True
        return self

    def transform(self, conversation: Conversation) -> FeatureRecord:
        """Transform one conversation after fitting all stateful steps."""
        if not self._fitted:
            raise ValueError("feature pipeline has not been fitted")
        if not isinstance(conversation, Conversation):
            raise TypeError("pipeline input must be a Conversation")
        blocks = {name: transformer.transform(conversation) for name, transformer in self._steps}
        return FeatureRecord(conversation.id, blocks)

    def fit_transform(self, conversations: Sequence[Conversation]) -> tuple[FeatureRecord, ...]:
        """Fit once and transform the original sequence order."""
        self.fit(conversations)
        return tuple(self.transform(conversation) for conversation in conversations)
