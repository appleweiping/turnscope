# Tabular window exports

`turnscope tabular` writes one deterministic CSV row per target utterance. The
default columns contain IDs, roles, UTC timestamp, policy, token total, and
JSON-encoded context ID/role/token arrays. Message text is intentionally not
included unless `--include-text` is supplied.

```bash
turnscope tabular examples/conversations.json --output windows.csv \
  --policy token --value 512
```

The Python API exposes the same contract:

```python
from turnscope import windows_csv

csv_text = windows_csv(windows, include_text=False)
```

Rows preserve the order produced by the context builder. JSON arrays inside
CSV cells use compact UTF-8 JSON, so IDs containing commas remain lossless.
