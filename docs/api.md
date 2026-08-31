# Python API

The public names below are exported from `turnscope`. Runtime modules depend only on the Python standard library.

## Loading and adapters

- `iter_conversations(stream, format="json")` lazily yields native records. JSONL is line-streamed; a JSON document is
  parsed as one value by the standard library.
- `iter_path(path, format=None)` keeps its input file open only while the returned iterator is consumed.
- `iter_adapted_jsonl(stream, format=..., id_prefix="conversation")` and `iter_adapted_path(...)` perform strict,
  located format conversion one line at a time.
- `adapt_openai`, `adapt_anthropic`, `adapt_sharegpt`, and `adapt_conversation` convert one in-memory source value.
- Existing `load_conversations` and `load_path` continue to return lists for v0.1 compatibility.

Adapter results remain `Conversation` values. When conversion omits vendor fields, their names—not their values—appear
in `conversation.metadata["adapter_warnings"]`. Conflicting aliases and nonrepresentable multimodal/tool semantics are
located errors.

Iterators are single-pass. Parser or adapter errors can occur after earlier records have been yielded. Materialize with
`list(...)` before side effects when transactional behavior is required.

## Window construction

`ContextBuilder(policy, token_counter=...)` provides two entry points:

```python
windows = builder.build(conversation, target_ids={"answer-7"})
iterator = builder.iter_build("conversation-id", utterance_iterator)
```

`build` is the all-or-nothing compatibility API. `iter_build` yields each window as soon as its target is read. When
target IDs are requested, an unknown ID is reported only after input exhaustion, so prior yielded windows remain valid.
Input positions, not timestamps, define what is prior.

Built-in policy classes are `TurnWindowPolicy`, `TokenBudgetPolicy`, `TimeWindowPolicy`, and `ReplyChainPolicy`.
Third-party policies implementing the original `WindowPolicy.select` protocol remain supported through a compatibility
path; their complexity is policy-defined.

## Token counters

`TokenCounter` is a runtime-checkable callable protocol. Results must be non-negative integers and deterministic for a
given string. Supplied utterance `token_count` values still take precedence.

- `WhitespaceTokenCounter` counts Unicode-whitespace-delimited runs.
- `Utf8ByteTokenCounter(bytes_per_token=4)` returns the ceiling of UTF-8 byte length divided by a fixed positive value.
- `whitespace_tokens` remains the v0.1-compatible function form.

Neither implementation claims model-tokenizer equivalence. Pass a custom callable for exact model accounting.

## Audit

`Auditor.audit()` now consumes the conversation iterable in one pass instead of tupleizing it. The returned
`AuditReport` still retains every issue, so memory is proportional to findings. Rules continue to operate on one
materialized `Conversation` at a time.
