"""Reply-conditioned lexical coordination with partner-specific response baselines."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .graph import reply_forest
from .models import Conversation
from .transformers import _tokens, default_coordination_categories


@dataclass(frozen=True, slots=True)
class ReplyCoordinationScore:
    """Observed parent-speaker to responding-speaker category statistics.

    ``source`` wrote the parent and ``target`` wrote its reply. The target's
    baseline uses all its replies to this source, including replies to parents
    that do not use the category. Counts remain available below support gates.
    """

    source: str
    target: str
    category: str
    replies: int
    conditioned_replies: int
    response_category_replies: int
    coordinated_replies: int
    conditional_rate: float | None
    baseline_rate: float
    score: float | None
    support_failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize counts, rates, and explicit reasons for withholding a score."""
        return {
            "source": self.source,
            "target": self.target,
            "category": self.category,
            "replies": self.replies,
            "conditioned_replies": self.conditioned_replies,
            "response_category_replies": self.response_category_replies,
            "coordinated_replies": self.coordinated_replies,
            "conditional_rate": self.conditional_rate,
            "baseline_rate": self.baseline_rate,
            "score": self.score,
            "support_failures": list(self.support_failures),
        }


def _categories(categories: Mapping[str, Iterable[str]]) -> dict[str, frozenset[str]]:
    if not isinstance(categories, Mapping) or not categories:
        raise ValueError("categories must be a non-empty mapping")
    normalized: dict[str, frozenset[str]] = {}
    for name, words in categories.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("category names must be non-empty strings")
        if isinstance(words, (str, bytes, Mapping)) or not isinstance(words, Iterable):
            raise ValueError(f"category {name!r} must contain an iterable of terms")
        terms: set[str] = set()
        for word in words:
            if not isinstance(word, str) or not word or _tokens(word) != (word.casefold(),):
                raise ValueError(f"category {name!r} terms must each be one lexical token")
            terms.add(word.casefold())
        if not terms:
            raise ValueError(f"category {name!r} must contain at least one term")
        normalized[name] = frozenset(terms)
    return dict(sorted(normalized.items()))


def reply_coordination(
    conversations: Iterable[Conversation],
    categories: Mapping[str, Iterable[str]] | None = None,
    *,
    speaker_field: str | None = None,
    min_replies: int = 1,
    min_conditioned_replies: int = 1,
    min_response_category_replies: int = 0,
) -> tuple[ReplyCoordinationScore, ...]:
    """Aggregate actual cross-speaker replies and subtract the response baseline.

    Each observed ordered speaker pair produces a row for every category.
    Support thresholds apply separately to each pair/category; insufficient
    support yields a ``None`` score while retaining counts and measured rates.
    Roots and same-speaker replies contribute no samples. Missing parents,
    cycles, duplicate conversation/utterance IDs, and missing speaker metadata
    fail rather than inventing adjacency or silently discarding malformed links.
    """
    if speaker_field is not None and (
        not isinstance(speaker_field, str) or not speaker_field.strip()
    ):
        raise ValueError("speaker_field must be a non-empty string or None")
    for name, value, minimum in (
        ("min_replies", min_replies, 1),
        ("min_conditioned_replies", min_conditioned_replies, 1),
        ("min_response_category_replies", min_response_category_replies, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    normalized = _categories(
        default_coordination_categories() if categories is None else categories
    )
    lexicon: dict[str, set[str]] = defaultdict(set)
    for category, words in normalized.items():
        for word in words:
            lexicon[word].add(category)
    seen: set[str] = set()
    pairs: Counter[tuple[str, str]] = Counter()
    conditioned: Counter[tuple[str, str, str]] = Counter()
    response: Counter[tuple[str, str, str]] = Counter()
    coordinated: Counter[tuple[str, str, str]] = Counter()
    for conversation in conversations:
        if not isinstance(conversation, Conversation):
            raise TypeError("conversations must contain Conversation values")
        if conversation.id in seen:
            raise ValueError(f"duplicate conversation ID: {conversation.id!r}")
        seen.add(conversation.id)
        forest = reply_forest(conversation)
        speakers: dict[str, str] = {}
        annotations: dict[str, set[str]] = {}
        for item in conversation.utterances:
            identity = item.role if speaker_field is None else item.metadata.get(speaker_field)
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError(
                    f"missing speaker identity for conversation {conversation.id!r}, "
                    f"utterance {item.id!r}"
                )
            speakers[item.id] = identity
            present: set[str] = set()
            for token in set(_tokens(item.text)):
                present.update(lexicon.get(token, ()))
            annotations[item.id] = present
        for reply_id, parent_id in forest.parents.items():
            if parent_id is None or speakers[parent_id] == speakers[reply_id]:
                continue
            source, target = speakers[parent_id], speakers[reply_id]
            pairs[source, target] += 1
            parent_categories = annotations[parent_id]
            reply_categories = annotations[reply_id]
            conditioned.update((source, target, category) for category in parent_categories)
            response.update((source, target, category) for category in reply_categories)
            coordinated.update(
                (source, target, category) for category in parent_categories & reply_categories
            )
    scores: list[ReplyCoordinationScore] = []
    for (source, target), total in sorted(pairs.items()):
        for category in normalized:
            key = source, target, category
            parent_count, response_count, both_count = (
                conditioned[key],
                response[key],
                coordinated[key],
            )
            conditional_rate = both_count / parent_count if parent_count else None
            baseline_rate = response_count / total
            failures = tuple(
                name
                for name, count, required in (
                    ("min_replies", total, min_replies),
                    ("min_conditioned_replies", parent_count, min_conditioned_replies),
                    (
                        "min_response_category_replies",
                        response_count,
                        min_response_category_replies,
                    ),
                )
                if count < required
            )
            scores.append(
                ReplyCoordinationScore(
                    source,
                    target,
                    category,
                    total,
                    parent_count,
                    response_count,
                    both_count,
                    conditional_rate,
                    baseline_rate,
                    conditional_rate - baseline_rate
                    if conditional_rate is not None and not failures
                    else None,
                    failures,
                )
            )
    return tuple(scores)
