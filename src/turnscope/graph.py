"""Reply-forest analysis independent of input order and Python recursion depth."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .models import Conversation


@dataclass(frozen=True)
class ReplyForest:
    """A validated reply forest with explicit root-to-leaf depth and descendants.

    Edges point from a parent message to its reply. Root depth is zero; descendant
    counts exclude the message itself. Missing parents, duplicate IDs and cycles
    are errors rather than silently repaired links. Chronology is not imposed:
    use TurnScope's auditor to inspect timestamp inconsistencies separately.
    """

    conversation_id: str
    roots: tuple[str, ...]
    parents: Mapping[str, str | None]
    children: Mapping[str, tuple[str, ...]]
    depths: Mapping[str, int]
    descendants: Mapping[str, int]
    traversal: tuple[str, ...]

    def ancestors(self, utterance_id: str) -> tuple[str, ...]:
        """Return nearest parent first; unknown IDs raise KeyError."""
        result: list[str] = []
        parent = self.parents[utterance_id]
        while parent is not None:
            result.append(parent)
            parent = self.parents[parent]
        return tuple(result)

    def subtree(self, utterance_id: str) -> tuple[str, ...]:
        """Return a breadth-first subtree, including the requested message."""
        if utterance_id not in self.parents:
            raise KeyError(utterance_id)
        queue = deque([utterance_id])
        result: list[str] = []
        while queue:
            current = queue.popleft()
            result.append(current)
            queue.extend(self.children[current])
        return tuple(result)


def reply_forest(conversation: Conversation) -> ReplyForest:
    """Build a forest in O(messages + reply edges) time and memory.

    Roots and siblings retain their original input order, even when the input
    places a reply before its parent. Algorithms are iterative for deep threads.
    """
    parents: dict[str, str | None] = {}
    children: dict[str, list[str]] = {}
    for item in conversation.utterances:
        if item.id in parents:
            raise ValueError(f"duplicate utterance ID: {item.id!r}")
        parents[item.id] = item.reply_to
        children[item.id] = []
    roots: list[str] = []
    for item in conversation.utterances:
        if item.reply_to is None:
            roots.append(item.id)
        elif item.reply_to not in parents:
            raise ValueError(f"unknown reply parent {item.reply_to!r} for {item.id!r}")
        else:
            children[item.reply_to].append(item.id)
    depths = dict.fromkeys(roots, 0)
    queue = deque(roots)
    traversal: list[str] = []
    while queue:
        current = queue.popleft()
        traversal.append(current)
        for child in children[current]:
            depths[child] = depths[current] + 1
            queue.append(child)
    if len(traversal) != len(parents):
        unresolved = sorted(set(parents) - set(traversal))
        raise ValueError(f"reply cycle or descendants of a cycle: {unresolved!r}")
    descendants = dict.fromkeys(parents, 0)
    for current in reversed(traversal):
        parent = parents[current]
        if parent is not None:
            descendants[parent] += descendants[current] + 1
    return ReplyForest(
        conversation.id,
        tuple(roots),
        MappingProxyType(parents),
        MappingProxyType({key: tuple(value) for key, value in children.items()}),
        MappingProxyType(depths),
        MappingProxyType(descendants),
        tuple(traversal),
    )


@dataclass(frozen=True)
class InteractionEdge:
    """Directed replies from one role or explicit speaker to another."""

    sender: str
    recipient: str
    replies: int
    mean_latency_seconds: float
    negative_latencies: int


def interaction_edges(
    conversation: Conversation, *, speaker_field: str | None = None
) -> tuple[InteractionEdge, ...]:
    """Aggregate reply edges, including self-replies, with signed latency.

    By default endpoints are *roles*, not presumed human identities. Set
    ``speaker_field`` to a metadata key for speaker analysis; every utterance
    must then contain a nonempty string at that key. Negative latency is retained
    and counted instead of clamped or silently excluded. No cross-conversation
    identity inference is performed.
    """
    forest = reply_forest(conversation)
    records = conversation.by_id()
    speakers: dict[str, str] = {}
    for item in conversation.utterances:
        value = item.role if speaker_field is None else item.metadata.get(speaker_field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing or invalid speaker for utterance {item.id!r}")
        speakers[item.id] = value
    counts: Counter[tuple[str, str]] = Counter()
    totals: dict[tuple[str, str], float] = defaultdict(float)
    negatives: Counter[tuple[str, str]] = Counter()
    for child, parent in forest.parents.items():
        if parent is None:
            continue
        key = speakers[child], speakers[parent]
        latency = (records[child].timestamp - records[parent].timestamp).total_seconds()
        counts[key] += 1
        totals[key] += latency
        negatives[key] += int(latency < 0)
    return tuple(
        InteractionEdge(
            sender,
            recipient,
            count,
            totals[sender, recipient] / count,
            negatives[sender, recipient],
        )
        for (sender, recipient), count in sorted(counts.items())
    )
