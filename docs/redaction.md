# Conversation redaction

`turnscope redact` creates a deterministic sanitized export for review or
sharing. It detects email addresses, phone-like numbers, and common `sk-*` or
`gh*_` API-key forms. URLs are opt-in because they are often useful context.
Replacement labels contain only a short digest, so repeated identifiers remain
linkable without writing the original value to the report.

```bash
turnscope redact conversations.jsonl --output sanitized.jsonl --report redaction.json
```

The output format follows the input suffix unless `--format json` or
`--format jsonl` is supplied. Use `--no-email`, `--no-phone`, or `--no-api-key`
to disable a detector. The Python API is
`redact_conversations(conversations, policy=RedactionPolicy(...))`.

This is a conservative pattern-based sanitizer, not a guarantee of anonymity;
review the sanitized output before external publication.
