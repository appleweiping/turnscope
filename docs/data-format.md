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
Valid escaped UTF-16 surrogate pairs are normalized to their Unicode scalar value. Unpaired surrogates and invalid
UTF-8 input files are rejected with a located error, so serialization cannot fail later on malformed text.

The parser intentionally does not require IDs to be unique or replies to exist: those are audit findings, so callers can
load and diagnose an imperfect dataset. Type errors, missing required fields, invalid timestamps, negative token counts,
and naive timestamps are parsing failures because they prevent unambiguous interpretation.

`iter_conversations` and `iter_path` parse JSONL one physical line at a time. They do not pre-read later records, so a
late error can follow earlier yielded conversations. JSON is parsed as one complete document. JSONL bounds dataset-level
input memory, but each line still contains and materializes one conversation; feed an utterance iterator directly to
`ContextBuilder.iter_build` when an individual conversation requires streaming.
`dump_conversations` consumes both JSON and JSONL iterables one conversation at a time; it never first converts the
dataset iterable to a list. Each individual conversation is materialized for JSON serialization.
