"""Strict JSON and JSONL readers and writers for conversation data."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, TextIO

from .models import Conversation, JsonValue, Utterance


class DataFormatError(ValueError):
    """Raised when input is valid JSON but violates the TurnScope schema."""

    def __init__(self, location: str, message: str) -> None:
        super().__init__(f"{location}: {message}")
        self.location = location
        self.message = message


def _reject_non_finite(value: str) -> NoReturn:
    raise ValueError(f"non-finite number {value!r} is not valid JSON")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite number {value!r} is not valid JSON")
    return parsed


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r} is not valid JSON input")
        result[key] = value
    return result


def parse_json_value(text: str, *, location: str = "JSON") -> Any:
    """Parse one strict JSON value with duplicate and non-finite checks."""
    try:
        value = json.loads(
            text,
            parse_constant=_reject_non_finite,
            parse_float=_finite_float,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise DataFormatError(
            location,
            f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}",
        ) from error
    except ValueError as error:
        raise DataFormatError(location, str(error)) from error
    return _normalize_parsed_json(value, location)


def _normalize_string(value: str, location: str) -> str:
    """Combine valid escaped surrogate pairs and reject unpaired surrogates."""
    try:
        return value.encode("utf-16", "surrogatepass").decode("utf-16")
    except UnicodeError as error:
        raise DataFormatError(location, "string contains an unpaired Unicode surrogate") from error


def _normalize_parsed_json(value: Any, location: str) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataFormatError(location, "non-finite number is not valid JSON")
        return value
    if isinstance(value, str):
        return _normalize_string(value, location)
    if isinstance(value, list):
        return [
            _normalize_parsed_json(item, f"{location}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = _normalize_string(key, f"{location} object key")
            if normalized_key in normalized:
                raise DataFormatError(
                    location,
                    f"duplicate object key {normalized_key!r} after Unicode normalization",
                )
            normalized[normalized_key] = _normalize_parsed_json(
                item, f"{location}.{normalized_key}"
            )
        return normalized
    raise DataFormatError(location, "expected a JSON value")


def _validate_json_value(value: Any, location: str) -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeError as error:
            raise DataFormatError(
                location, "string contains an unpaired Unicode surrogate"
            ) from error
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataFormatError(location, "non-finite number is not valid JSON")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{location}[{index}]")
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, item in value.items():
            _validate_json_value(key, f"{location} object key")
            _validate_json_value(item, f"{location}.{key}")
        return
    raise DataFormatError(location, "expected a JSON value")


def _expect_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataFormatError(location, "expected a JSON object")
    return value


def _required_string(data: dict[str, Any], key: str, location: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise DataFormatError(f"{location}.{key}", "expected a string")
    return value


def _metadata(data: dict[str, Any], location: str) -> dict[str, JsonValue]:
    value = data.get("metadata", {})
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DataFormatError(f"{location}.metadata", "expected an object with string keys")
    _validate_json_value(value, f"{location}.metadata")
    return value


def utterance_from_dict(value: Any, location: str = "utterance") -> Utterance:
    """Parse and validate an utterance mapping."""
    data = _expect_object(value, location)
    _validate_json_value(data, location)
    timestamp_text = _required_string(data, "timestamp", location)
    try:
        timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
    except ValueError as error:
        raise DataFormatError(f"{location}.timestamp", "expected an ISO 8601 timestamp") from error
    reply_to = data.get("reply_to")
    if reply_to is not None and not isinstance(reply_to, str):
        raise DataFormatError(f"{location}.reply_to", "expected a string or null")
    token_count = data.get("token_count")
    if token_count is not None and (
        not isinstance(token_count, int) or isinstance(token_count, bool)
    ):
        raise DataFormatError(f"{location}.token_count", "expected an integer or null")
    try:
        return Utterance(
            id=_required_string(data, "id", location),
            role=_required_string(data, "role", location),
            text=_required_string(data, "text", location),
            timestamp=timestamp,
            reply_to=reply_to,
            token_count=token_count,
            metadata=_metadata(data, location),
        )
    except ValueError as error:
        raise DataFormatError(location, str(error)) from error


def conversation_from_dict(value: Any, location: str = "conversation") -> Conversation:
    """Parse a conversation mapping without reordering its utterances."""
    data = _expect_object(value, location)
    _validate_json_value(data, location)
    utterances = data.get("utterances")
    if not isinstance(utterances, list):
        raise DataFormatError(f"{location}.utterances", "expected an array")
    try:
        return Conversation(
            id=_required_string(data, "id", location),
            utterances=[
                utterance_from_dict(item, f"{location}.utterances[{index}]")
                for index, item in enumerate(utterances)
            ],
            metadata=_metadata(data, location),
        )
    except ValueError as error:
        if isinstance(error, DataFormatError):
            raise
        raise DataFormatError(location, str(error)) from error


def utterance_to_dict(item: Utterance) -> dict[str, JsonValue]:
    """Serialize an utterance using canonical UTC timestamps."""
    result: dict[str, JsonValue] = {
        "id": item.id,
        "role": item.role,
        "text": item.text,
        "timestamp": item.timestamp.isoformat().replace("+00:00", "Z"),
    }
    if item.reply_to is not None:
        result["reply_to"] = item.reply_to
    if item.token_count is not None:
        result["token_count"] = item.token_count
    if item.metadata:
        result["metadata"] = dict(item.metadata)
    return result


def conversation_to_dict(item: Conversation) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {
        "id": item.id,
        "utterances": [utterance_to_dict(value) for value in item.utterances],
    }
    if item.metadata:
        result["metadata"] = dict(item.metadata)
    return result


def iter_conversations(stream: TextIO, *, format: str = "json") -> Iterator[Conversation]:
    """Yield native conversations, reading JSONL one physical line at a time.

    Standard-library JSON parsing still materializes a complete ``json``
    document. Use ``jsonl`` when bounded input memory is required.
    """
    if format == "jsonl":
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = parse_json_value(line, location=f"line {line_number}")
            yield conversation_from_dict(value, f"line {line_number}")
        return
    if format != "json":
        raise ValueError("format must be 'json' or 'jsonl'")
    try:
        value = json.load(
            stream,
            parse_constant=_reject_non_finite,
            parse_float=_finite_float,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise DataFormatError("JSON", f"invalid JSON: {error.msg}") from error
    except ValueError as error:
        raise DataFormatError("JSON", str(error)) from error
    normalized = _normalize_parsed_json(value, "JSON")
    values = normalized if isinstance(normalized, list) else [normalized]
    for index, item in enumerate(values):
        yield conversation_from_dict(item, f"conversations[{index}]")


def load_conversations(stream: TextIO, *, format: str = "json") -> list[Conversation]:
    """Materialize native conversations from JSON or JSONL."""
    return list(iter_conversations(stream, format=format))


def iter_path(path: str | Path, *, format: str | None = None) -> Iterator[Conversation]:
    """Yield conversations from a path while keeping the file lifetime scoped."""
    resolved = Path(path)
    selected = format or ("jsonl" if resolved.suffix.lower() == ".jsonl" else "json")
    try:
        with resolved.open(encoding="utf-8") as stream:
            yield from iter_conversations(stream, format=selected)
    except UnicodeError as error:
        raise DataFormatError(str(resolved), "input is not valid UTF-8") from error


def load_path(path: str | Path, *, format: str | None = None) -> list[Conversation]:
    return list(iter_path(path, format=format))


def dump_conversations(items: Iterable[Conversation], stream: TextIO, *, format: str) -> None:
    if format == "json":
        stream.write("[")
        first = True
        for index, item in enumerate(items):
            value = conversation_to_dict(item)
            _validate_json_value(value, f"conversations[{index}]")
            rendered = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
            if first:
                stream.write("\n")
                first = False
            else:
                stream.write(",\n")
            stream.write("\n".join(f"  {line}" for line in rendered.splitlines()))
        stream.write("\n]\n" if not first else "]\n")
    elif format == "jsonl":
        for index, item in enumerate(items):
            value = conversation_to_dict(item)
            _validate_json_value(value, f"line {index + 1}")
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    else:
        raise ValueError("format must be 'json' or 'jsonl'")
