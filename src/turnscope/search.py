"""Deterministic BM25-style retrieval over conversation utterances."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .models import Conversation
from .transformers import _tokens


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One utterance match ranked by lexical relevance."""

    conversation_id: str
    utterance_id: str
    score: float
    matched_terms: tuple[str, ...]


class ConversationSearchIndex:
    """An in-memory BM25 index over normalized utterance tokens."""

    def __init__(self, conversations: Iterable[Conversation] = ()) -> None:
        self._documents: dict[tuple[str, str], tuple[str, ...]] = {}
        self._lengths: dict[tuple[str, str], int] = {}
        self._postings: dict[str, dict[tuple[str, str], int]] = defaultdict(dict)
        self._conversation_docs: dict[str, set[tuple[str, str]]] = defaultdict(set)
        self._average_length = 0.0
        for conversation in conversations:
            self.add(conversation)

    @property
    def documents(self) -> int:
        """Number of indexed utterances."""
        return len(self._documents)

    @property
    def average_length(self) -> float:
        """Mean token count used by BM25 normalization."""
        return self._average_length

    def add(self, conversation: Conversation) -> None:
        """Index or replace one conversation without mutating it."""
        self.remove(conversation.id)
        for utterance in conversation.utterances:
            key = (conversation.id, utterance.id)
            tokens = _tokens(utterance.text)
            self._documents[key] = tokens
            self._lengths[key] = len(tokens)
            for token, count in Counter(tokens).items():
                self._postings[token][key] = count
            self._conversation_docs[conversation.id].add(key)
        self._recompute_average()

    def remove(self, conversation_id: str) -> bool:
        """Remove a conversation and return whether it was present."""
        keys = self._conversation_docs.pop(conversation_id, None)
        if not keys:
            return False
        for key in keys:
            tokens = set(self._documents.pop(key, ()))
            self._lengths.pop(key, None)
            for token in tokens:
                postings = self._postings[token]
                postings.pop(key, None)
                if not postings:
                    self._postings.pop(token, None)
        self._recompute_average()
        return True

    def query(
        self,
        text: str,
        *,
        limit: int = 10,
        conversation_id: str | None = None,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> tuple[SearchHit, ...]:
        """Return stable BM25-ranked utterance hits."""
        if not isinstance(text, str):
            raise TypeError("query text must be a string")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if (
            not isinstance(k1, (int, float))
            or isinstance(k1, bool)
            or not math.isfinite(k1)
            or k1 <= 0
        ):
            raise ValueError("k1 must be finite and positive")
        if (
            not isinstance(b, (int, float))
            or isinstance(b, bool)
            or not math.isfinite(b)
            or not 0 <= b <= 1
        ):
            raise ValueError("b must be finite and between zero and one")
        query_terms = tuple(dict.fromkeys(_tokens(text)))
        if not query_terms or not self._documents:
            return ()
        allowed = self._conversation_docs.get(conversation_id, set()) if conversation_id else None
        scores: dict[tuple[str, str], float] = defaultdict(float)
        matched: dict[tuple[str, str], set[str]] = defaultdict(set)
        total = len(self._documents)
        for term in query_terms:
            postings = self._postings.get(term, {})
            if not postings:
                continue
            idf = math.log(1.0 + (total - len(postings) + 0.5) / (len(postings) + 0.5))
            for key, frequency in postings.items():
                if allowed is not None and key not in allowed:
                    continue
                length = self._lengths[key]
                normalization = 1.0 - float(b) + float(b) * length / max(self._average_length, 1.0)
                scores[key] += (
                    idf * frequency * (float(k1) + 1.0) / (frequency + float(k1) * normalization)
                )
                matched[key].add(term)
        ordered = sorted(scores, key=lambda key: (-scores[key], key[0], key[1]))[:limit]
        return tuple(
            SearchHit(key[0], key[1], scores[key], tuple(sorted(matched[key]))) for key in ordered
        )

    def terms(self) -> Mapping[str, int]:
        """Return vocabulary document frequencies for diagnostics."""
        return MappingProxyType(
            {term: len(postings) for term, postings in sorted(self._postings.items())}
        )

    def _recompute_average(self) -> None:
        self._average_length = (
            sum(self._lengths.values()) / len(self._lengths) if self._lengths else 0.0
        )
