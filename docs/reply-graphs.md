# Reply-forest and interaction analysis

`reply_forest(conversation)` constructs an immutable, validated forest with
parents, children, root IDs, breadth-first traversal, depth, and descendant
counts. Root depth is zero and descendant counts exclude the node itself.
Roots and siblings retain input order; parents need not precede replies in the
input. Duplicate IDs, missing parents and cycles fail explicitly.

```python
from turnscope import interaction_edges, reply_forest

# conversation is a native Conversation loaded using iter_path or CorpusStore.
forest = reply_forest(conversation)
for root in forest.roots:
    print(root, forest.subtree(root))
for edge in interaction_edges(conversation, speaker_field="speaker_id"):
    print(edge.sender, edge.recipient, edge.replies, edge.mean_latency_seconds)
```

`ancestors(id)` returns nearest parent first. `subtree(id)` includes the requested
node. Construction is linear in messages plus edges and uses no recursion;
deep threads do not depend on Python's recursion limit.

Interaction edges point **from the replying speaker to the parent's speaker**.
Without `speaker_field`, endpoints are roles, not human identities. With an
explicit field, every message must have a nonempty string speaker ID in metadata.
No identity is inferred across conversations. Self-replies are counted. Signed
latency is child timestamp minus parent timestamp; negative values are retained
and counted separately. Timestamp correctness is a separate auditor concern.

These graph primitives do not provide fitted models, social-science validity,
or evidence that a role-level network identifies real participants.
