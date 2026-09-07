"""Reply-forest analysis independent of input order and Python recursion depth."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping
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


@dataclass(frozen=True, slots=True)
class NetworkEdge:
    """A reply edge aggregated across a conversation collection."""

    sender: str
    recipient: str
    replies: int
    conversations: int
    mean_latency_seconds: float
    negative_latencies: int


@dataclass(frozen=True, slots=True)
class SpeakerSummary:
    """Corpus-level activity counts for one role or speaker identity."""

    speaker: str
    conversations: int
    utterances: int
    replies_sent: int
    replies_received: int


@dataclass(frozen=True, slots=True)
class InteractionNetwork:
    """Deterministic corpus-level network with node and edge summaries."""

    conversations: int
    speakers: Mapping[str, SpeakerSummary]
    edges: tuple[NetworkEdge, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible network report."""

        return {
            "conversations": self.conversations,
            "speakers": [
                {
                    "speaker": summary.speaker,
                    "conversations": summary.conversations,
                    "utterances": summary.utterances,
                    "replies_sent": summary.replies_sent,
                    "replies_received": summary.replies_received,
                }
                for summary in self.speakers.values()
            ],
            "edges": [
                {
                    "sender": edge.sender,
                    "recipient": edge.recipient,
                    "replies": edge.replies,
                    "conversations": edge.conversations,
                    "mean_latency_seconds": edge.mean_latency_seconds,
                    "negative_latencies": edge.negative_latencies,
                }
                for edge in self.edges
            ],
        }


class InteractionNetworkAccumulator:
    """Incrementally aggregate reply interactions without retaining conversations."""

    def __init__(self, *, speaker_field: str | None = None) -> None:
        if speaker_field is not None and (
            not isinstance(speaker_field, str) or not speaker_field.strip()
        ):
            raise ValueError("speaker_field must be a non-empty string or None")
        self.speaker_field = speaker_field
        self._conversations = 0
        self._speaker_conversations: dict[str, set[str]] = defaultdict(set)
        self._utterance_counts: Counter[str] = Counter()
        self._edge_replies: Counter[tuple[str, str]] = Counter()
        self._edge_conversations: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._edge_latency: defaultdict[tuple[str, str], float] = defaultdict(float)
        self._edge_negative: Counter[tuple[str, str]] = Counter()
        self._sent: Counter[str] = Counter()
        self._received: Counter[str] = Counter()

    def add(self, conversation: Conversation) -> None:
        """Consume one conversation; duplicate IDs are rejected by graph validation."""

        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        speakers: dict[str, str] = {}
        for item in conversation.utterances:
            value = (
                item.role if self.speaker_field is None else item.metadata.get(self.speaker_field)
            )
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"missing or invalid speaker for utterance {item.id!r}")
            speakers[item.id] = value
            self._speaker_conversations[value].add(conversation.id)
            self._utterance_counts[value] += 1
        for edge in interaction_edges(conversation, speaker_field=self.speaker_field):
            key = edge.sender, edge.recipient
            self._edge_replies[key] += edge.replies
            self._edge_conversations[key].add(conversation.id)
            self._edge_latency[key] += edge.mean_latency_seconds * edge.replies
            self._edge_negative[key] += edge.negative_latencies
            self._sent[edge.sender] += edge.replies
            self._received[edge.recipient] += edge.replies
        self._conversations += 1

    def finish(self) -> InteractionNetwork:
        """Return a deterministic snapshot of the consumed aggregate."""

        summaries = {
            speaker: SpeakerSummary(
                speaker,
                len(self._speaker_conversations[speaker]),
                self._utterance_counts[speaker],
                self._sent[speaker],
                self._received[speaker],
            )
            for speaker in sorted(self._speaker_conversations)
        }
        edges = tuple(
            NetworkEdge(
                sender,
                recipient,
                count,
                len(self._edge_conversations[sender, recipient]),
                self._edge_latency[sender, recipient] / count,
                self._edge_negative[sender, recipient],
            )
            for (sender, recipient), count in sorted(self._edge_replies.items())
        )
        return InteractionNetwork(self._conversations, MappingProxyType(summaries), edges)


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


def interaction_network(
    conversations: Iterable[Conversation],
    *,
    speaker_field: str | None = None,
) -> InteractionNetwork:
    """Aggregate reply interactions across conversations in deterministic order.

    The function preserves signed latency semantics from :func:`interaction_edges`.
    A speaker's conversation count is based on appearances, while edge
    conversation counts count distinct conversations containing that direction.
    """

    accumulator = InteractionNetworkAccumulator(speaker_field=speaker_field)
    for conversation in conversations:
        accumulator.add(conversation)
    return accumulator.finish()
