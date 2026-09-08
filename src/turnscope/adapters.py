"""Strict, deterministic adapters for common chat-export shapes."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, TextIO

from .io import DataFormatError, _validate_json_value, parse_json_value
from .models import Conversation, JsonValue, Utterance

ChatFormat = Literal["openai", "anthropic", "sharegpt", "convokit"]
_SYNTHETIC_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataFormatError(location, "expected an object")
    if not all(isinstance(key, str) for key in value):
        raise DataFormatError(location, "expected an object with string keys")
    return value


def _array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise DataFormatError(location, "expected an array")
    return value


def _nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise DataFormatError(location, "expected a string")
    _ensure_unicode(value, location)
    if not value.strip():
        raise DataFormatError(location, "must not be empty")
    return value


def _text_string(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise DataFormatError(location, "expected a string")
    _ensure_unicode(value, location)
    return value


def _ensure_unicode(value: str, location: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise DataFormatError(location, "string contains an unpaired Unicode surrogate") from error


def _conversation_id(
    data: dict[str, Any], override: str | None, fallback: str, location: str
) -> str:
    source_ids = {
        key: _nonempty_string(data[key], f"{location}.{key}")
        for key in ("id", "conversation_id")
        if key in data
    }
    if len(set(source_ids.values())) > 1:
        raise DataFormatError(location, "conflicting id and conversation_id fields")
    source_id = next(iter(source_ids.values()), None)
    if override is not None:
        selected = _nonempty_string(override, f"{location}.conversation_id")
        if source_id is not None and source_id != selected:
            raise DataFormatError(location, "explicit conversation ID conflicts with source ID")
        return selected
    if source_id is not None:
        return source_id
    return fallback


def _content(
    value: Any,
    location: str,
    *,
    block_types: frozenset[str],
    warnings: list[str],
) -> str:
    if isinstance(value, str):
        _ensure_unicode(value, location)
        return value
    blocks = _array(value, location)
    pieces: list[str] = []
    for index, raw_block in enumerate(blocks):
        block_location = f"{location}[{index}]"
        block = _object(raw_block, block_location)
        block_type = _nonempty_string(block.get("type"), f"{block_location}.type")
        if block_type not in block_types:
            supported = ", ".join(sorted(block_types))
            raise DataFormatError(
                f"{block_location}.type",
                f"unsupported content block {block_type!r}; supported: {supported}",
            )
        text = _text_string(block.get("text"), f"{block_location}.text")
        pieces.append(text)
        _warn_unrepresented(block, {"type", "text"}, block_location, warnings)
    if len(pieces) > 1:
        warnings.append(f"{location}: text-block boundaries flattened with newline separators")
    return "\n".join(pieces)


def _timestamp(data: dict[str, Any], index: int, location: str) -> datetime:
    if "timestamp" in data and "created_at" in data:
        raise DataFormatError(location, "conflicting timestamp and created_at fields")
    key = "timestamp" if "timestamp" in data else "created_at" if "created_at" in data else None
    if key is None:
        return _SYNTHETIC_EPOCH + timedelta(microseconds=index)
    value = data[key]
    timestamp_location = f"{location}.{key}"
    if isinstance(value, str):
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise DataFormatError(timestamp_location, "expected an ISO 8601 timestamp") from error
        if timestamp.tzinfo is None:
            raise DataFormatError(timestamp_location, "timestamp must include a timezone")
        try:
            return timestamp.astimezone(timezone.utc)
        except (OverflowError, OSError) as error:
            raise DataFormatError(
                timestamp_location, "timestamp is outside the UTC range"
            ) from error
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataFormatError(timestamp_location, "expected an ISO timestamp or Unix seconds")
    if not math.isfinite(value):
        raise DataFormatError(timestamp_location, "Unix timestamp must be finite")
    try:
        return datetime.fromtimestamp(value, timezone.utc)
    except (OverflowError, OSError, ValueError) as error:
        raise DataFormatError(
            timestamp_location, "Unix timestamp is outside the UTC range"
        ) from error


def _optional_string(data: dict[str, Any], key: str, location: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    return _nonempty_string(value, f"{location}.{key}")


def _optional_token_count(data: dict[str, Any], location: str) -> int | None:
    value = data.get("token_count")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataFormatError(f"{location}.token_count", "expected a non-negative integer or null")
    return value


def _warn_unrepresented(
    data: dict[str, Any], represented: set[str], location: str, warnings: list[str]
) -> None:
    extras = sorted(set(data) - represented)
    if extras:
        warnings.append(f"{location}: fields not represented: {', '.join(extras)}")


def _adapter_metadata(source_format: ChatFormat, warnings: list[str]) -> dict[str, JsonValue]:
    metadata: dict[str, JsonValue] = {"source_format": source_format}
    if warnings:
        metadata["adapter_warnings"] = list(warnings)
    return metadata


def _utterance(
    data: dict[str, Any],
    *,
    location: str,
    conversation_id: str,
    index: int,
    role: str,
    text: str,
    source_format: ChatFormat,
    extra_metadata: Mapping[str, JsonValue] | None = None,
) -> Utterance:
    item_id = (
        _nonempty_string(data["id"], f"{location}.id")
        if "id" in data
        else f"{conversation_id}:{index}"
    )
    metadata: dict[str, JsonValue] = {"source_format": source_format, "source_index": index}
    if "name" in data:
        metadata["name"] = _nonempty_string(data["name"], f"{location}.name")
    if extra_metadata:
        metadata.update(extra_metadata)
    try:
        return Utterance(
            id=item_id,
            role=role,
            text=text,
            timestamp=_timestamp(data, index, location),
            reply_to=_optional_string(data, "reply_to", location),
            token_count=_optional_token_count(data, location),
            metadata=metadata,
        )
    except ValueError as error:
        raise DataFormatError(location, str(error)) from error


def adapt_openai(
    value: Any, *, conversation_id: str | None = None, location: str = "openai"
) -> Conversation:
    """Adapt an OpenAI-style ``messages`` object or a bare message array."""
    if isinstance(value, list):
        if conversation_id is None:
            raise DataFormatError(location, "bare message arrays require conversation_id")
        data: dict[str, Any] = {}
        messages = value
    else:
        data = _object(value, location)
        messages = _array(data.get("messages"), f"{location}.messages")
    item_id = _conversation_id(data, conversation_id, "conversation", location)
    warnings: list[str] = []
    _warn_unrepresented(data, {"id", "conversation_id", "messages"}, location, warnings)
    utterances: list[Utterance] = []
    for index, raw_message in enumerate(messages):
        message_location = f"{location}.messages[{index}]"
        message = _object(raw_message, message_location)
        role = _nonempty_string(message.get("role"), f"{message_location}.role")
        if role not in {"assistant", "developer", "function", "system", "tool", "user"}:
            raise DataFormatError(f"{message_location}.role", f"unsupported role {role!r}")
        for field in ("audio", "function_call", "refusal", "tool_call_id", "tool_calls"):
            if field in message and message[field] not in (None, [], ""):
                raise DataFormatError(
                    f"{message_location}.{field}",
                    "field cannot be represented by the text-only conversation model",
                )
        text = _content(
            message.get("content"),
            f"{message_location}.content",
            block_types=frozenset({"input_text", "output_text", "text"}),
            warnings=warnings,
        )
        _warn_unrepresented(
            message,
            {
                "content",
                "created_at",
                "id",
                "name",
                "reply_to",
                "role",
                "timestamp",
                "token_count",
            },
            message_location,
            warnings,
        )
        utterances.append(
            _utterance(
                message,
                location=message_location,
                conversation_id=item_id,
                index=index,
                role=role,
                text=text,
                source_format="openai",
            )
        )
    _validate_json_value(value, location)
    return Conversation(item_id, utterances, _adapter_metadata("openai", warnings))


def adapt_anthropic(
    value: Any, *, conversation_id: str | None = None, location: str = "anthropic"
) -> Conversation:
    """Adapt an Anthropic Messages request with optional system content."""
    data = _object(value, location)
    messages = _array(data.get("messages"), f"{location}.messages")
    item_id = _conversation_id(data, conversation_id, "conversation", location)
    warnings: list[str] = []
    _warn_unrepresented(
        data,
        {"id", "conversation_id", "messages", "system"},
        location,
        warnings,
    )
    utterances: list[Utterance] = []
    offset = 0
    if "system" in data:
        text = _content(
            data["system"],
            f"{location}.system",
            block_types=frozenset({"text"}),
            warnings=warnings,
        )
        utterances.append(
            _utterance(
                {},
                location=f"{location}.system",
                conversation_id=item_id,
                index=0,
                role="system",
                text=text,
                source_format="anthropic",
            )
        )
        offset = 1
    for source_index, raw_message in enumerate(messages):
        message_location = f"{location}.messages[{source_index}]"
        message = _object(raw_message, message_location)
        role = _nonempty_string(message.get("role"), f"{message_location}.role")
        if role not in {"user", "assistant"}:
            raise DataFormatError(f"{message_location}.role", "expected 'user' or 'assistant'")
        text = _content(
            message.get("content"),
            f"{message_location}.content",
            block_types=frozenset({"text"}),
            warnings=warnings,
        )
        _warn_unrepresented(
            message,
            {
                "content",
                "created_at",
                "id",
                "name",
                "reply_to",
                "role",
                "timestamp",
                "token_count",
            },
            message_location,
            warnings,
        )
        utterances.append(
            _utterance(
                message,
                location=message_location,
                conversation_id=item_id,
                index=source_index + offset,
                role=role,
                text=text,
                source_format="anthropic",
            )
        )
    _validate_json_value(value, location)
    return Conversation(item_id, utterances, _adapter_metadata("anthropic", warnings))


_SHAREGPT_ROLES = {
    "assistant": "assistant",
    "function": "tool",
    "gpt": "assistant",
    "human": "user",
    "system": "system",
    "tool": "tool",
    "user": "user",
}


def adapt_sharegpt(
    value: Any, *, conversation_id: str | None = None, location: str = "sharegpt"
) -> Conversation:
    """Adapt a ShareGPT ``conversations`` object."""
    data = _object(value, location)
    messages = _array(data.get("conversations"), f"{location}.conversations")
    item_id = _conversation_id(data, conversation_id, "conversation", location)
    warnings: list[str] = []
    _warn_unrepresented(
        data,
        {"id", "conversation_id", "conversations"},
        location,
        warnings,
    )
    utterances: list[Utterance] = []
    for index, raw_message in enumerate(messages):
        message_location = f"{location}.conversations[{index}]"
        message = _object(raw_message, message_location)
        source_role = _nonempty_string(message.get("from"), f"{message_location}.from")
        try:
            role = _SHAREGPT_ROLES[source_role]
        except KeyError as error:
            choices = ", ".join(sorted(_SHAREGPT_ROLES))
            raise DataFormatError(
                f"{message_location}.from",
                f"unsupported role {source_role!r}; supported: {choices}",
            ) from error
        text = _text_string(message.get("value"), f"{message_location}.value")
        _warn_unrepresented(
            message,
            {
                "created_at",
                "from",
                "id",
                "name",
                "reply_to",
                "timestamp",
                "token_count",
                "value",
            },
            message_location,
            warnings,
        )
        utterances.append(
            _utterance(
                message,
                location=message_location,
                conversation_id=item_id,
                index=index,
                role=role,
                text=text,
                source_format="sharegpt",
            )
        )
    _validate_json_value(value, location)
    return Conversation(item_id, utterances, _adapter_metadata("sharegpt", warnings))


def adapt_convokit(
    value: Any, *, conversation_id: str | None = None, location: str = "convokit"
) -> Conversation:
    """Adapt a ConvoKit utterance collection into one validated conversation.

    ConvoKit stores utterances as JSONL records with ``speaker``,
    ``conversation_id``, ``reply_to``, ``timestamp``, ``text``, and optional
    ``meta`` fields.  TurnScope keeps speaker identity in metadata and uses a
    deterministic ``speaker:<id>`` role because ConvoKit does not impose a
    user/assistant role vocabulary.
    """

    if isinstance(value, list):
        data: dict[str, Any] = {}
        raw_utterances = value
    else:
        data = _object(value, location)
        raw_utterances = _array(data.get("utterances"), f"{location}.utterances")
    utterances_data = [
        _object(item, f"{location}.utterances[{index}]")
        for index, item in enumerate(raw_utterances)
    ]
    inferred_ids = {
        _nonempty_string(item["conversation_id"], f"{location}.utterances[{index}].conversation_id")
        for index, item in enumerate(utterances_data)
        if "conversation_id" in item and item["conversation_id"] is not None
    }
    if len(inferred_ids) > 1:
        raise DataFormatError(location, "utterances contain conflicting conversation_id fields")
    selected_data = dict(data)
    if not any(key in selected_data for key in ("id", "conversation_id")) and inferred_ids:
        selected_data["conversation_id"] = next(iter(inferred_ids))
    item_id = _conversation_id(selected_data, conversation_id, "conversation", location)
    warnings: list[str] = []
    _warn_unrepresented(
        data,
        {"id", "conversation_id", "utterances", "meta", "speakers"},
        location,
        warnings,
    )
    converted: list[Utterance] = []
    for index, utterance in enumerate(utterances_data):
        utterance_location = f"{location}.utterances[{index}]"
        source_conversation = utterance.get("conversation_id")
        if source_conversation is not None:
            source_conversation = _nonempty_string(
                source_conversation, f"{utterance_location}.conversation_id"
            )
            if source_conversation != item_id:
                raise DataFormatError(
                    f"{utterance_location}.conversation_id",
                    f"does not match conversation {item_id!r}",
                )
        speaker = _nonempty_string(utterance.get("speaker"), f"{utterance_location}.speaker")
        role = _optional_string(utterance, "role", utterance_location) or f"speaker:{speaker}"
        text = _text_string(utterance.get("text"), f"{utterance_location}.text")
        extra_metadata: dict[str, JsonValue] = {"speaker": speaker}
        if "meta" in utterance:
            utterance_meta = utterance["meta"]
            if not isinstance(utterance_meta, dict):
                raise DataFormatError(f"{utterance_location}.meta", "expected an object")
            extra_metadata["convokit_meta"] = utterance_meta
        _warn_unrepresented(
            utterance,
            {
                "conversation_id",
                "created_at",
                "id",
                "meta",
                "name",
                "reply_to",
                "role",
                "speaker",
                "text",
                "timestamp",
                "token_count",
            },
            utterance_location,
            warnings,
        )
        normalized = dict(utterance)
        if normalized.get("timestamp") is None:
            normalized.pop("timestamp", None)
        if normalized.get("created_at") is None:
            normalized.pop("created_at", None)
        converted.append(
            _utterance(
                normalized,
                location=utterance_location,
                conversation_id=item_id,
                index=index,
                role=role,
                text=text,
                source_format="convokit",
                extra_metadata=extra_metadata,
            )
        )
    metadata = _adapter_metadata("convokit", warnings)
    for key in ("meta", "speakers"):
        if key in data:
            if not isinstance(data[key], dict):
                raise DataFormatError(f"{location}.{key}", "expected an object")
            metadata[f"convokit_{key}"] = data[key]
    _validate_json_value(value, location)
    return Conversation(item_id, converted, metadata)


def adapt_conversation(
    value: Any,
    *,
    format: ChatFormat,
    conversation_id: str | None = None,
    location: str | None = None,
) -> Conversation:
    """Dispatch to a strict adapter without guessing the source format."""
    selected_location = location or format
    if format == "openai":
        return adapt_openai(value, conversation_id=conversation_id, location=selected_location)
    if format == "anthropic":
        return adapt_anthropic(value, conversation_id=conversation_id, location=selected_location)
    if format == "sharegpt":
        return adapt_sharegpt(value, conversation_id=conversation_id, location=selected_location)
    if format == "convokit":
        return adapt_convokit(value, conversation_id=conversation_id, location=selected_location)
    raise ValueError("format must be 'openai', 'anthropic', 'sharegpt', or 'convokit'")


def iter_adapted_conversations(
    values: Iterable[Any], *, format: ChatFormat, id_prefix: str = "conversation"
) -> Iterator[Conversation]:
    """Lazily adapt an iterable, assigning stable fallback conversation IDs."""
    _nonempty_string(id_prefix, "id_prefix")
    for index, value in enumerate(values):
        override = None if _source_has_id(value) else f"{id_prefix}-{index}"
        yield adapt_conversation(
            value,
            format=format,
            conversation_id=override,
            location=f"records[{index}]",
        )


def _source_has_id(value: Any) -> bool:
    return isinstance(value, dict) and ("id" in value or "conversation_id" in value)


def iter_adapted_jsonl(
    stream: TextIO, *, format: ChatFormat, id_prefix: str = "conversation"
) -> Iterator[Conversation]:
    """Parse and adapt JSONL incrementally with physical line locations."""
    _nonempty_string(id_prefix, "id_prefix")
    if format == "convokit":
        yield from iter_adapted_convokit_jsonl(stream, id_prefix=id_prefix)
        return
    record_index = 0
    for line_number, line in enumerate(stream, 1):
        if not line.strip():
            continue
        location = f"line {line_number}"
        value = parse_json_value(line, location=location)
        override = None if _source_has_id(value) else f"{id_prefix}-{record_index}"
        yield adapt_conversation(
            value,
            format=format,
            conversation_id=override,
            location=location,
        )
        record_index += 1


def iter_adapted_path(
    path: str | Path, *, format: ChatFormat, id_prefix: str = "conversation"
) -> Iterator[Conversation]:
    """Stream an adapter-format JSONL file from disk."""
    with Path(path).open(encoding="utf-8") as stream:
        yield from iter_adapted_jsonl(stream, format=format, id_prefix=id_prefix)


def iter_adapted_convokit_jsonl(
    stream: TextIO, *, id_prefix: str = "conversation"
) -> Iterator[Conversation]:
    """Group ConvoKit ``utterances.jsonl`` rows by conversation ID.

    Input rows may be interleaved; grouping preserves first-seen conversation
    order and source order within each conversation.
    """

    _nonempty_string(id_prefix, "id_prefix")
    groups: dict[str, list[dict[str, Any]]] = {}
    for line_number, line in enumerate(stream, 1):
        if not line.strip():
            continue
        location = f"line {line_number}"
        value = _object(parse_json_value(line, location=location), location)
        raw_id = value.get("conversation_id")
        if raw_id is None:
            conversation_key = f"{id_prefix}-{len(groups)}"
        else:
            conversation_key = _nonempty_string(raw_id, f"{location}.conversation_id")
        groups.setdefault(conversation_key, []).append(value)
    for conversation_key, utterances in groups.items():
        yield adapt_convokit(
            {"conversation_id": conversation_key, "utterances": utterances},
            location=f"conversation {conversation_key}",
        )
