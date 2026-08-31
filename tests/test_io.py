import io
import json

import pytest

from turnscope.io import (
    DataFormatError,
    conversation_from_dict,
    conversation_to_dict,
    dump_conversations,
    load_conversations,
    load_path,
)
from turnscope.models import Conversation


def _valid() -> dict[str, object]:
    return {
        "id": "c1",
        "metadata": {"split": "test"},
        "utterances": [
            {
                "id": "u1",
                "role": "user",
                "text": "héllo",
                "timestamp": "2026-01-01T00:00:00Z",
                "token_count": 1,
                "metadata": {"ok": True},
            }
        ],
    }


def test_round_trip_json_and_jsonl(tmp_path) -> None:  # type: ignore[no-untyped-def]
    conversation = conversation_from_dict(_valid())
    assert conversation_to_dict(conversation) == _valid()
    for format in ("json", "jsonl"):
        stream = io.StringIO()
        dump_conversations([conversation], stream, format=format)
        loaded = load_conversations(io.StringIO(stream.getvalue()), format=format)
        assert loaded == [conversation]
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(_valid()) + "\n\n", encoding="utf-8")
    assert load_path(path) == [conversation]


@pytest.mark.parametrize(
    ("mutate", "location"),
    [
        (lambda data: data.update(id=3), ".id"),
        (lambda data: data.update(utterances={}), ".utterances"),
        (lambda data: data.update(metadata=[]), ".metadata"),
        (lambda data: data["utterances"][0].update(timestamp="yesterday"), ".timestamp"),
        (lambda data: data["utterances"][0].update(reply_to=3), ".reply_to"),
        (lambda data: data["utterances"][0].update(token_count=True), ".token_count"),
    ],
)
def test_schema_errors_are_located(mutate, location: str) -> None:  # type: ignore[no-untyped-def]
    data = _valid()
    mutate(data)
    with pytest.raises(DataFormatError, match=location):
        conversation_from_dict(data)


def test_invalid_json_and_format_errors() -> None:
    with pytest.raises(DataFormatError, match="invalid JSON"):
        load_conversations(io.StringIO("{"))
    with pytest.raises(DataFormatError, match="line 2"):
        load_conversations(io.StringIO("\n{"), format="jsonl")
    with pytest.raises(ValueError, match="format"):
        load_conversations(io.StringIO("[]"), format="yaml")
    with pytest.raises(ValueError, match="format"):
        dump_conversations([], io.StringIO(), format="yaml")


@pytest.mark.parametrize(
    ("format", "text"),
    [
        ("json", '[{"id":"c","metadata":{"x":NaN},"utterances":[]}]'),
        ("jsonl", '{"id":"c","metadata":{"x":Infinity},"utterances":[]}\n'),
        ("json", '[{"id":"c","metadata":{"x":1e400},"utterances":[]}]'),
        ("jsonl", '{"id":"c","metadata":{"x":-1e400},"utterances":[]}\n'),
        ("json", '[{"id":"c","unknown":{"nested":1e400},"utterances":[]}]'),
    ],
)
def test_non_finite_numbers_are_rejected(format: str, text: str) -> None:
    with pytest.raises(DataFormatError, match="non-finite"):
        load_conversations(io.StringIO(text), format=format)


@pytest.mark.parametrize("format", ["json", "jsonl"])
def test_duplicate_object_keys_are_rejected(format: str) -> None:
    text = '{"id":"first","id":"second","utterances":[]}'
    with pytest.raises(DataFormatError, match="duplicate object key 'id'"):
        load_conversations(io.StringIO(text), format=format)


@pytest.mark.parametrize("value", [float("inf"), {"nested": [float("nan")]}, ("tuple",)])
def test_direct_metadata_requires_strict_json_values(value: object) -> None:
    data = _valid()
    data["metadata"] = {"value": value}
    with pytest.raises(DataFormatError, match=r"(non-finite number|expected a JSON value)"):
        conversation_from_dict(data)


def test_direct_input_rejects_non_json_values_even_in_unknown_fields() -> None:
    data = _valid()
    data["unknown"] = {"nested": ("tuple",)}
    with pytest.raises(DataFormatError, match="expected a JSON value"):
        conversation_from_dict(data)


@pytest.mark.parametrize("format", ["json", "jsonl"])
def test_writers_refuse_non_finite_metadata(format: str) -> None:
    conversation = Conversation("c", [], metadata={"score": float("nan")})
    with pytest.raises(ValueError, match="Out of range float"):
        dump_conversations([conversation], io.StringIO(), format=format)


def test_timestamp_utc_overflow_is_located() -> None:
    data = _valid()
    data["utterances"][0]["timestamp"] = "9999-12-31T23:59:59-23:59"  # type: ignore[index]
    with pytest.raises(DataFormatError, match="supported UTC range"):
        conversation_from_dict(data)
