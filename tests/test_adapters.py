import io
import json
from datetime import datetime, timezone

import pytest

from turnscope.adapters import (
    adapt_anthropic,
    adapt_conversation,
    adapt_openai,
    adapt_sharegpt,
    iter_adapted_conversations,
    iter_adapted_jsonl,
    iter_adapted_path,
)
from turnscope.io import DataFormatError


def test_openai_adapter_handles_strings_blocks_and_source_fields() -> None:
    result = adapt_openai(
        {
            "id": "openai-1",
            "messages": [
                {
                    "id": "m1",
                    "role": "developer",
                    "content": "Be concise.",
                    "created_at": 1,
                    "name": "policy",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "hello"},
                        {"type": "text", "text": "world"},
                    ],
                    "timestamp": "2026-01-01T08:00:00+08:00",
                    "reply_to": "m1",
                },
            ],
        }
    )
    assert result.id == "openai-1"
    assert [item.role for item in result.utterances] == ["developer", "user"]
    assert result.utterances[0].id == "m1"
    assert result.utterances[0].metadata["name"] == "policy"
    assert result.utterances[0].timestamp == datetime.fromtimestamp(1, timezone.utc)
    assert result.utterances[1].id == "openai-1:1"
    assert result.utterances[1].text == "hello\nworld"
    assert result.utterances[1].reply_to == "m1"
    assert result.utterances[1].timestamp == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_openai_bare_messages_use_explicit_id_and_synthetic_time() -> None:
    result = adapt_openai(
        [{"role": "system", "content": "rules"}, {"role": "assistant", "content": "ok"}],
        conversation_id="bare",
    )
    assert result.id == "bare"
    assert [item.id for item in result.utterances] == ["bare:0", "bare:1"]
    assert result.utterances[1].timestamp > result.utterances[0].timestamp
    with pytest.raises(DataFormatError, match="require conversation_id"):
        adapt_openai([{"role": "user", "content": "x"}])


def test_adapter_warnings_are_durable_without_copying_values() -> None:
    result = adapt_openai(
        {
            "model": "vendor-model",
            "api_key": "must-not-appear",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "one", "cache_control": {}},
                        {"type": "text", "text": "two"},
                    ],
                    "vendor_field": {"secret": "must-not-appear"},
                    "token_count": 2,
                }
            ],
        }
    )
    warnings = result.metadata["adapter_warnings"]
    assert isinstance(warnings, list)
    rendered = "\n".join(str(item) for item in warnings)
    assert all(field in rendered for field in ("api_key", "model", "cache_control", "vendor_field"))
    assert "boundaries flattened" in rendered
    assert "must-not-appear" not in rendered
    assert result.utterances[0].token_count == 2

    with pytest.raises(DataFormatError, match="non-finite"):
        adapt_openai({"messages": [], "ignored": float("inf")})
    with pytest.raises(DataFormatError, match="expected a JSON value"):
        adapt_openai({"messages": [], "ignored": ("not", "json")})


@pytest.mark.parametrize(
    "value",
    [
        {"id": "one", "conversation_id": "two", "messages": []},
        {"id": "one", "messages": []},
    ],
)
def test_adapter_rejects_conflicting_conversation_ids(value: object) -> None:
    with pytest.raises(DataFormatError, match="conflict"):
        adapt_openai(value, conversation_id="override")


def test_adapter_rejects_conflicting_time_and_tool_semantics() -> None:
    with pytest.raises(DataFormatError, match="conflicting timestamp"):
        adapt_openai(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "x",
                        "timestamp": "2026-01-01T00:00:00Z",
                        "created_at": 0,
                    }
                ]
            }
        )
    with pytest.raises(DataFormatError, match="text-only"):
        adapt_openai(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "call-1"}],
                    }
                ]
            }
        )


def test_anthropic_adapter_includes_system_and_text_blocks() -> None:
    result = adapt_anthropic(
        {
            "conversation_id": "claude-1",
            "system": [{"type": "text", "text": "safe"}],
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            ],
        }
    )
    assert [item.role for item in result.utterances] == ["system", "user", "assistant"]
    assert [item.text for item in result.utterances] == ["safe", "question", "answer"]
    assert result.utterances[-1].metadata == {"source_format": "anthropic", "source_index": 2}


def test_sharegpt_adapter_maps_documented_roles() -> None:
    result = adapt_sharegpt(
        {
            "id": "share-1",
            "conversations": [
                {"from": "human", "value": "question"},
                {"from": "gpt", "value": "answer"},
                {"from": "function", "value": "result"},
            ],
        }
    )
    assert [item.role for item in result.utterances] == ["user", "assistant", "tool"]
    assert result.metadata == {"source_format": "sharegpt"}


@pytest.mark.parametrize(
    ("factory", "value", "match"),
    [
        (adapt_openai, 3, r"openai: expected an object"),
        (adapt_openai, {}, r"openai.messages: expected an array"),
        (adapt_openai, {"messages": [3]}, r"messages\[0\]: expected an object"),
        (adapt_openai, {"messages": [{"content": "x"}]}, r"messages\[0\].role"),
        (
            adapt_openai,
            {"messages": [{"role": "alien", "content": "x"}]},
            r"messages\[0\].role: unsupported role",
        ),
        (
            adapt_openai,
            {"messages": [{"role": "user", "content": None}]},
            r"messages\[0\].content: expected an array",
        ),
        (
            adapt_openai,
            {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
            r"content\[0\].type: unsupported",
        ),
        (
            adapt_openai,
            {"messages": [{"role": "user", "content": [{"type": "text"}]}]},
            r"content\[0\].text",
        ),
        (
            adapt_anthropic,
            {"messages": [{"role": "tool", "content": "x"}]},
            r"messages\[0\].role",
        ),
        (
            adapt_anthropic,
            {"system": [{"type": "tool_use", "id": "1"}], "messages": []},
            r"system\[0\].type: unsupported",
        ),
        (
            adapt_sharegpt,
            {"conversations": [{"from": "unknown", "value": "x"}]},
            r"conversations\[0\].from: unsupported",
        ),
        (
            adapt_sharegpt,
            {"conversations": [{"from": "human", "value": 3}]},
            r"conversations\[0\].value",
        ),
    ],
)
def test_adapter_schema_errors_have_precise_paths(factory, value: object, match: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(DataFormatError, match=match):
        factory(value)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("id", "", r"\.id: must not be empty"),
        ("name", 2, r"\.name: expected a string"),
        ("reply_to", 2, r"\.reply_to: expected a string"),
        ("timestamp", "yesterday", r"\.timestamp: expected an ISO"),
        ("timestamp", "2026-01-01", r"timestamp must include a timezone"),
        ("timestamp", True, r"expected an ISO timestamp or Unix seconds"),
        ("timestamp", float("inf"), r"Unix timestamp must be finite"),
        ("timestamp", 10**30, r"outside the UTC range"),
    ],
)
def test_openai_message_field_boundaries(field: str, value: object, match: str) -> None:
    message: dict[str, object] = {"role": "user", "content": "x", field: value}
    with pytest.raises(DataFormatError, match=match):
        adapt_openai({"messages": [message]})


def test_dispatch_and_lazy_iterable_fallback_ids() -> None:
    consumed: list[int] = []

    def values():  # type: ignore[no-untyped-def]
        for index in range(2):
            consumed.append(index)
            yield {"messages": [{"role": "user", "content": str(index)}]}

    adapted = iter_adapted_conversations(values(), format="openai", id_prefix="batch")
    assert consumed == []
    assert next(adapted).id == "batch-0"
    assert consumed == [0]
    assert next(adapted).id == "batch-1"
    with pytest.raises(StopIteration):
        next(adapted)
    with pytest.raises(ValueError, match="format"):
        adapt_conversation({}, format="invalid")  # type: ignore[arg-type]
    with pytest.raises(DataFormatError, match="id_prefix"):
        next(iter_adapted_conversations([], format="openai", id_prefix=""))


def test_jsonl_adapter_is_incremental_and_reports_physical_lines(tmp_path) -> None:  # type: ignore[no-untyped-def]
    first = {"messages": [{"role": "user", "content": "one"}]}
    second = {"messages": [{"role": "assistant", "content": "two"}]}
    stream = io.StringIO(json.dumps(first) + "\n\n" + json.dumps(second) + "\n")
    items = iter_adapted_jsonl(stream, format="openai", id_prefix="line")
    assert next(items).id == "line-0"
    assert next(items).id == "line-1"
    with pytest.raises(StopIteration):
        next(items)

    with pytest.raises(DataFormatError, match="line 2"):
        list(iter_adapted_jsonl(io.StringIO("\n{\n"), format="sharegpt"))
    with pytest.raises(DataFormatError, match="duplicate object key"):
        list(iter_adapted_jsonl(io.StringIO('{"messages":[],"messages":[]}\n'), format="openai"))
    with pytest.raises(DataFormatError, match="non-finite"):
        list(iter_adapted_jsonl(io.StringIO('{"messages":[],"ignored":1e400}\n'), format="openai"))

    path = tmp_path / "sharegpt.jsonl"
    path.write_text(
        json.dumps({"id": "p", "conversations": [{"from": "human", "value": "hi"}]}) + "\n",
        encoding="utf-8",
    )
    assert [item.id for item in iter_adapted_path(path, format="sharegpt")] == ["p"]
