# Data format

JSON input may contain one conversation object or an array. JSONL contains one conversation object per nonblank line.
The `.jsonl` extension selects JSONL automatically; all other extensions select JSON.

```json
{
  "id": "conversation-id",
  "metadata": {"split": "validation"},
  "utterances": [
    {
      "id": "message-id",
      "role": "user",
      "text": "Hello",
      "timestamp": "2026-01-01T10:00:00Z",
      "reply_to": null,
      "token_count": 1,
      "metadata": {"channel": "web"}
    }
  ]
}
```

`id`, `utterances`, and the four utterance fields `id`, `role`, `text`, and `timestamp` are required. IDs and roles must
be nonempty strings. Timestamps are ISO 8601 values with an explicit timezone. `reply_to` is a string or null;
`token_count` is a non-negative integer or null. Metadata is an optional JSON object with string keys.
JSON object keys must be unique and numeric values must be finite. Duplicate keys, `NaN`, and positive or negative
infinity—including values that overflow while parsing, such as `1e400`—are rejected on input. Non-finite values are
also rejected on output.

The parser intentionally does not require IDs to be unique or replies to exist: those are audit findings, so callers can
load and diagnose an imperfect dataset. Type errors, missing required fields, invalid timestamps, negative token counts,
and naive timestamps are parsing failures because they prevent unambiguous interpretation.
