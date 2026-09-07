# Changelog

All notable changes follow the principles of Keep a Changelog.

## [Unreleased]

- Add opt-in packaging entry points for external tokenizers and audit rules,
  with a discovery CLI and strict plugin contracts.

- Add strict named JSON profiles for reusable context policies, token counters,
  and audit budgets through `build --config` and `audit --config`.

- Add atomic, resumable `corpus export` JSONL output and paged
  `CorpusStore.iter_conversations`.
- Add per-speaker network centrality plus density, reciprocity, and weak-component
  diagnostics to interaction-network reports.
- Add deterministic conversation redaction for common emails, phones, API keys,
  and opt-in URLs, with a redacted-count report and `redact` CLI.

### Added

- Add directional adjacent-turn linguistic coordination scores with evidence counts
  and explicit no-evidence values.
- Add per-speaker type-token ratio and lexical-entropy diversity profiles.
- Add corpus-wide speaker identity profiles with reply-edge aggregates and a
  deterministic `speaker-profile` CLI.
- Atomic SQLite corpus imports, replacement, keyset pagination, and corpus CLI queries.
- Iterative reply-forest analysis and explicit role/speaker interaction aggregation.
- Fit/transform TF-IDF vectors and deterministic conversation/speaker feature aggregates.
- Deterministic BM25-style conversation search with filtering and vocabulary diagnostics.
- Versioned, deterministic search-index snapshots with digest-authenticated save/load support.
- Ordered fit/transform feature pipelines for composing TF-IDF and conversation transformers.
- Dependency-free fitted conversation classifier with authenticated JSON persistence and deterministic probabilities.
- Add a `turnscope classify` CLI for fitting, authenticating, and applying the
  dependency-free conversation classifier.
- Add corpus-level interaction-network aggregation with deterministic speaker
  nodes, directed reply edges, signed latency summaries, and a `network` CLI.
- Add `InteractionNetworkAccumulator` and incremental JSONL network aggregation;
  the `network` CLI now avoids materializing the complete corpus.

## [0.2.0] - 2026-08-31

### Added

- Streaming JSONL/path readers, one-pass audit consumption, and lazy context construction for large inputs.
- Strict OpenAI, Anthropic, and ShareGPT adapters with deterministic fallback IDs and timestamps.
- Runtime-checkable token-counter protocol implementations for whitespace and fixed UTF-8 byte estimates.
- Reproducible corpus generation, builder benchmarks, complexity documentation, and randomized equivalence tests.

### Changed

- Built-in policies use incremental state and output-sensitive algorithms instead of rebuilding or rescanning the full
  conversation prefix for every target.
- JSON array and JSONL writers consume iterables incrementally.

### Fixed

- Reject float overflow and malformed Unicode at the strict JSON boundary, including adapter JSONL input.
- Report adapter field loss and reject conflicting IDs, timestamps, and unrepresentable tool semantics.
- Keep token-cost caches bounded by retained policy state and remove duplicate-ID audit rescans.
- Validate custom token counters consistently in builder and audit rules.

### Compatibility

- `ContextBuilder.build`, native load helpers, `whitespace_tokens`, policy selection semantics, CLI commands, and custom
  v0.1 `WindowPolicy` implementations remain supported.

## [0.1.0] - 2026-08-31

### Added

- Frozen core records with explicitly shallow-copied, mutable metadata.
- JSON/JSONL parsing, four context policies, seven audit rules, CLI, and report renderers.
- Typed APIs, cross-version CI, tests, examples, and architecture documentation.
