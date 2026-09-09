# Python API

The public names below are exported from `turnscope`. Core runtime operations use the Python standard library;
optional tokenizers and expected-context fitting load their explicitly selected dependencies lazily.

## Fitted expected contexts

`ExpectedContextModel` fits a context SVD and centered ridge map on explicit reply,
predecessor, or positional-successor examples. Its immutable `ExpectedContextState`
stores separate frozen source/context TF-IDF statistics and learned numeric parameters.
`predict(text)`, `project_context(text)`, and `transform(conversation)` return
`ContextPrediction` coordinates with token coverage; `evaluate(conversations)` reports
heldout MSE and a fixed training-context-mean baseline. `iter_context_pairs` exposes
the exact relation semantics. `save/load` and `to_dict/from_dict` use versioned,
validated JSON. Only fitting needs the `context` NumPy extra. Read the full
[estimator, leakage, artifact, and resource contract](expected-context.md).

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
- `TiktokenTokenCounter(encoding="cl100k_base")` uses the optional `tiktoken`
  package and an exact named encoding.
- `HuggingFaceTokenCounter(model, local_files_only=True, revision="main")` uses
  an optional fast Transformers tokenizer; local-only loading is the safe
  default. Pin `revision` to a model commit for reproducible remote loads.
- `whitespace_tokens` remains the v0.1-compatible function form.

The model-tokenizer counters defer optional imports until first use and cache
loaded encoders. Their package/model versions are part of your experiment
environment, so record them in an audit manifest when exact reproducibility is
required. Pass a custom callable for other tokenizer libraries.

## Corpus-wide speaker profiles

`corpus_speaker_profiles()` aggregates stable speaker identities across a
conversation collection, including conversation/utterance counts, lexical
vocabulary size, role composition, and reply edges. It uses utterance roles by
default or a required metadata field when provider-specific speaker IDs are
available. Duplicate conversation IDs and missing identities fail explicitly.

The same analysis is available as `turnscope speaker-profile DATASET`, which
emits deterministic JSON and accepts `--field` for metadata-backed identities.

## Reply-aware coordination and legacy conditional rates

`reply_coordination(conversations, categories=None, speaker_field=None, ...)`
aggregates explicit reply links between stable speakers. Its
`ReplyCoordinationScore` retains conditional and partner-specific baseline
rates, their difference, and independently gated per-pair/category counts.
The CLI is `turnscope reply-coordination DATASET`. See
[reply coordination](reply-coordination.md) for direction, equations,
thresholds, graph validation, and lexicon limitations.

The legacy `linguistic_coordination(conversation, categories)` measures a
conditional category response rate for adjacent turns by different roles,
without subtracting a baseline. The returned
`CoordinationScore` values retain conditioned and coordinated turn counts, so a
small sample is not mistaken for a strong estimate; `None` means the source
provided no evidence for that category. `default_coordination_categories()`
provides a dependency-free starter vocabulary. The equivalent CLI command is
`turnscope coordination DATASET`, and the local service exposes the same
per-conversation reports through the `coordination` operation.

## Fitted conversation vectors

`TfidfVectorizer.fit()` learns vocabulary and conversation-level document
frequencies. `transform_conversation()` and `transform_corpus()` apply frozen
parameters with none/L1/L2 normalization. `save()`/`load()` provide versioned
JSON persistence, while `SparseSimilarityIndex`, `sparse_cosine`, and
`normalize_sparse` provide sparse retrieval and geometry. See
[fitted vectors](vectors.md) for arithmetic, CLI, validation, and complexity.

## Audit

`Auditor.audit()` now consumes the conversation iterable in one pass instead of tupleizing it. The returned
`AuditReport` still retains every issue, so memory is proportional to findings. Rules continue to operate on one
materialized `Conversation` at a time.
