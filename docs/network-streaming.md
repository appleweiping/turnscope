# Streaming interaction networks

`InteractionNetworkAccumulator` aggregates speaker and reply-edge metadata one
conversation at a time. It retains only speaker/conversation membership and
edge counters, not message text or complete conversation objects. The batch
`interaction_network` function now delegates to the same accumulator, so both
paths share identical validation and deterministic output semantics.

The `turnscope network` command uses `iter_path` and therefore streams JSONL
inputs. It still reports signed reply latency, negative-latency counts,
conversation coverage, speaker activity, and directed edge summaries. Invalid
reply forests or missing speaker metadata fail before a report is emitted.

```python
from turnscope import InteractionNetworkAccumulator, iter_path

accumulator = InteractionNetworkAccumulator(speaker_field="speaker_id")
for conversation in iter_path("conversations.jsonl"):
    accumulator.add(conversation)
report = accumulator.finish()
```

Use `interaction_network(...)` when the input is already an in-memory iterable;
the result and JSON schema are the same.

