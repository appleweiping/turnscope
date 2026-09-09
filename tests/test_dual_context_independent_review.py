"""Independent resource and arithmetic checks of the shared-context core."""

from __future__ import annotations

import pytest

from turnscope import _dual_context_artifact as artifact
from turnscope.dual_context import DualContextModel


def test_pending_json_graph_nodes_are_admitted_before_repeated_container_expansion(monkeypatch):
    """A tiny cyclic Python graph must not amplify into an over-budget stack."""
    visits = 0

    class ObservedList(list):
        def __iter__(self):
            nonlocal visits
            visits += 1
            # Each expansion enqueues fifty children. After two expansions,
            # queued + already visited nodes have exhausted this 100-node cap.
            assert visits <= 2, "JSON walker expanded beyond its pending-node budget"
            return super().__iter__()

    graph = ObservedList()
    graph.extend([graph] * 50)
    monkeypatch.setattr(artifact, "MAX_JSON_NODES", 100)
    with pytest.raises(ValueError, match=r"structural|nodes|circular|cyclic|limit"):
        DualContextModel.from_dict(graph)
