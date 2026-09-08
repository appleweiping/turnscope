# TurnScope local service

`TurnScopeService` exposes the same deterministic workflows as the library and
CLI without requiring a framework or a remote dependency. `create_server()`
binds to loopback by default and accepts JSON `POST /v1/dispatch` requests.

```python
from turnscope import TurnScopeService

result = TurnScopeService().dispatch(
    {
        "operation": "build",
        "input": "examples/conversations.json",
        "policy": "turn",
        "value": 2,
    }
)
```

Supported operations are `audit`, `build`, `tabular`, `search`, `network`, and
`coordination`. Paths are
read-only inputs; callers own the process boundary and should not expose the
default server beyond a trusted local machine.

The `network` operation computes a deterministic interaction graph and returns
speaker nodes, directed reply edges, centrality, weak-connectivity information,
and aggregate metrics. Set the optional `speaker_field` metadata key when a
custom utterance field stores speaker identity.

`coordination` returns one report per conversation. It uses the default
dependency-free function-word categories unless `categories` is supplied as a
JSON object mapping category names to arrays of words. Each directional score
includes its support counts and is JSON-safe for downstream analysis.
