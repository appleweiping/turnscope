# Architecture

TurnScope has a one-way dependency structure: models sit at the center; parsing creates models; policies select models;
the builder assembles windows; rules create issues; and reporting serializes the resulting windows or audit report.
The CLI is deliberately thin and coordinates these public APIs.

## Construction path

`ContextBuilder.iter_build` consumes the original sequence exactly once. Before visiting each target, it exposes only
preceding positions to an incremental built-in policy state. This makes future information structurally unavailable,
including when the input has bad timestamps. `build` materializes this iterator for v0.1 compatibility. Each policy
returns existing frozen utterance records; the builder calculates context tokens and records the policy name for
provenance.

Token costs are lazy per-utterance state. Overlapping windows reuse a retained cost, while eviction releases both the
utterance and its cached cost; equal text in different utterances is not retained by a global cache.

Token budgets select a contiguous suffix. If an older message does not fit, selection stops rather than skipping it and
creating a misleading fragmented history. Reply-chain selection is the exception to contiguity: it follows explicit
ancestry and protects against malformed cycles.

Turn windows use a bounded deque, token windows retain the maximal eligible suffix, and reply chains update an ID index
once per message. Chronological time windows advance a boundary rather than rescanning the prefix. Out-of-order time
falls back to the original scan semantics. See [performance and scaling](performance.md) for complexity and benchmark
methodology. Third-party v0.1 policies remain source-compatible through a complete-prior-sequence adapter.

## Audit path

An `AuditRule` is a structural protocol. `Auditor` consumes conversations once and runs rules in configured order and
conversations in input order.
Findings therefore have stable ordering without a hidden sort key. The default set covers structural integrity while
`ConversationBudgetRule` is opt-in because a useful limit depends on the downstream model.
Built-in rules scan each conversation at most once per rule; duplicate-ID details are produced from the same count pass
rather than rescanning once per duplicated value.

Severities are ordered integers, enabling a report to answer whether it fails a requested threshold. JSON carries
machine-readable details, while Markdown favors human triage. Neither renderer changes the report.

## Extension boundaries

- Supply a token counter callable to `ContextBuilder` or token rules.
- Implement `WindowPolicy` to add a selection strategy.
- Implement `AuditRule` to add a reliability check.
- Consume records with frozen core fields and copied, intentionally mutable metadata rather than parser internals.

Runtime modules use only the Python standard library. Development tooling is isolated in the `dev` extra.
