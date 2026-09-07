# Corpus interaction networks

`interaction_edges()` describes one conversation. `interaction_network()` lifts
that analysis to a corpus: it creates deterministic speaker nodes, aggregates
directed reply edges, counts distinct conversations containing each edge, and
preserves signed reply latency (including negative timestamps) instead of
clamping or dropping it.

```python
from turnscope import interaction_network, load_path

report = interaction_network(load_path("conversations.jsonl"), speaker_field="speaker_id")
print(report.to_dict())
```

The equivalent CLI is:

```bash
turnscope network conversations.jsonl --field speaker_id --output network.json
```

Roles are used as endpoints when `speaker_field` is omitted. A metadata field
must contain a non-empty string for every utterance when it is selected. The
report is sorted by speaker and `(sender, recipient)` edge, so it can be
committed as a reproducible corpus-analysis artifact.
