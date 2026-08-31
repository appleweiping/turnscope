# Chat format adapters

TurnScope never guesses an input dialect. Call `adapt_openai`, `adapt_anthropic`, or `adapt_sharegpt`, or pass an
explicit format to `adapt_conversation`. Every consumed field is type-checked and errors identify the source path, for
example `openai.messages[2].content[0].text` or `line 17.conversations[1].from`.

## OpenAI

The adapter accepts an object with a `messages` array or a bare array when `conversation_id` is supplied. Roles
`developer`, `system`, `user`, `assistant`, `tool`, and legacy `function` are accepted. Content may be a string or an
array of `text`, `input_text`, and `output_text` blocks. Other block types fail explicitly rather than being dropped.
Non-empty `tool_calls`, `function_call`, `tool_call_id`, audio, and refusal fields also fail because the text-only model
cannot preserve their execution semantics. Bare arrays require an explicit `conversation_id`.

## Anthropic

The adapter accepts a Messages-style object with a `messages` array and optional `system` content. User and assistant
messages accept strings or `text` blocks. System content becomes the first `system` utterance. Tool-use and multimodal
blocks are currently outside the adapter's text-only contract and produce located errors.

## ShareGPT

The adapter reads the conventional `conversations` array. It maps `human`/`user` to `user`, `gpt`/`assistant` to
`assistant`, `function`/`tool` to `tool`, and preserves `system`. Unknown role labels fail explicitly.

## IDs, time, and provenance

- A source `id` or `conversation_id` is preserved. Otherwise callers may provide a conversation ID; batch helpers assign
  stable `<prefix>-<index>` IDs.
- Message IDs are preserved when present and otherwise become `<conversation-id>:<index>`.
- ISO 8601 `timestamp` and ISO or numeric Unix `created_at` values are normalized to UTC.
- Formats without message time receive deterministic timestamps beginning at `1970-01-01T00:00:00Z`, one microsecond
  apart in source order. No wall clock is read.
- Adapter name and source index are recorded in metadata. Unconsumed vendor-specific fields are not copied.

If both source ID aliases or both timestamp aliases are present, they must not conflict. Ignored field names and
text-block boundary flattening are recorded as strings in conversation metadata under `adapter_warnings`; source values
are never copied into those warnings. Multiple text blocks are joined with newline separators. A native `token_count`
field is preserved after non-negative-integer validation.

Synthetic IDs and timestamps are deterministic conversion artifacts, not claims about source-system identity or event
time. Supply real values before adapting when those semantics matter.

## Streaming JSONL

`iter_adapted_jsonl()` and `iter_adapted_path()` parse one nonblank physical line at a time. Each line must contain one
source conversation object. Duplicate JSON keys, non-finite numbers, malformed JSON, and schema errors include the
physical line number. A single conversation object on a line is still parsed as one JSON value; use
`ContextBuilder.iter_build()` with an utterance iterator when a single conversation itself cannot fit in memory.
