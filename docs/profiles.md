# Named profiles

Build and audit settings can be kept in a strict JSON profile file:

```json
{
  "profiles": {
    "support": {
      "policy": {
        "kind": "token",
        "value": 512,
        "include_target": true,
        "token_counter": "whitespace"
      },
      "audit": {"token_budget": 4096}
    }
  }
}
```

Use it from both commands:

```console
turnscope build conversations.jsonl --config profiles.json --profile support
turnscope audit conversations.jsonl --config profiles.json --profile support
```

Supported policy kinds are `turn`, `token`, `time`, and `reply-chain`; token
profiles may select `whitespace`, deterministic `utf8-byte`, optional
`tiktoken`, or optional `huggingface` counters. A tiktoken profile may set
`encoding`; a Hugging Face profile must set `model` and may set
`local_files_only` (default `true`) and `revision` (default `main`). Pin
`revision` to a model commit for reproducible remote loads. Optional packages
are imported only when the counter is called.
Unknown fields, duplicate JSON keys, invalid values, and unknown profile names
are rejected before any output is written. CLI flags remain available for
one-off invocations and cannot be mixed with profile-owned settings.
