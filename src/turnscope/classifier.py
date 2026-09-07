"""Deterministic, dependency-free conversation text classification."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .models import Conversation

_TOKEN = re.compile(r"[\w]+(?:['-][\w]+)*", re.UNICODE)


def _tokens(conversation: Conversation) -> tuple[str, ...]:
    return tuple(
        match.group(0).casefold()
        for utterance in conversation.utterances
        for match in _TOKEN.finditer(utterance.text)
    )


@dataclass(frozen=True, slots=True)
class ClassifierState:
    """Fitted multinomial model parameters."""

    vocabulary: tuple[str, ...]
    labels: tuple[str, ...]
    priors: Mapping[str, float]
    token_log_probabilities: Mapping[str, Mapping[str, float]]
    documents: int


class ConversationClassifier:
    """A transparent multinomial Naive Bayes model over conversation text."""

    def __init__(self, *, alpha: float = 1.0, max_features: int | None = None) -> None:
        if (
            not isinstance(alpha, (int, float))
            or isinstance(alpha, bool)
            or not math.isfinite(alpha)
        ):
            raise ValueError("alpha must be a finite positive number")
        if alpha <= 0:
            raise ValueError("alpha must be a finite positive number")
        if max_features is not None and (
            isinstance(max_features, bool) or not isinstance(max_features, int) or max_features < 1
        ):
            raise ValueError("max_features must be a positive integer or None")
        self._alpha = float(alpha)
        self._max_features = max_features
        self._state: ClassifierState | None = None

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has been called successfully."""
        return self._state is not None

    @property
    def state(self) -> ClassifierState:
        """Return fitted parameters or raise a clear error before fitting."""
        if self._state is None:
            raise ValueError("classifier has not been fitted")
        return self._state

    def fit(
        self, conversations: Iterable[Conversation], labels: Mapping[str, str]
    ) -> ConversationClassifier:
        """Fit on conversations keyed by their stable IDs."""
        materialized = tuple(conversations)
        if not materialized:
            raise ValueError("at least one conversation is required")
        if not all(isinstance(item, Conversation) for item in materialized):
            raise TypeError("conversations must contain Conversation values")
        ids = [item.id for item in materialized]
        if len(set(ids)) != len(ids):
            raise ValueError("conversation IDs must be unique")
        if not isinstance(labels, Mapping):
            raise TypeError("labels must be a mapping from conversation ID to label")
        missing = set(ids) - labels.keys()
        extra = set(labels) - set(ids)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing labels: {', '.join(sorted(missing))}")
            if extra:
                details.append(f"unknown labels: {', '.join(sorted(extra))}")
            raise ValueError("; ".join(details))
        if not all(
            isinstance(labels[item.id], str) and labels[item.id].strip() for item in materialized
        ):
            raise TypeError("labels must be non-empty strings")

        counts_by_label: dict[str, Counter[str]] = {}
        document_counts: Counter[str] = Counter()
        vocabulary_counts: Counter[str] = Counter()
        for conversation in materialized:
            label = labels[conversation.id]
            tokens = Counter(_tokens(conversation))
            counts_by_label.setdefault(label, Counter()).update(tokens)
            document_counts[label] += 1
            vocabulary_counts.update(tokens.keys())
        candidates = sorted(vocabulary_counts, key=lambda token: (-vocabulary_counts[token], token))
        if self._max_features is not None:
            candidates = candidates[: self._max_features]
        vocabulary = tuple(candidates)
        labels_ordered = tuple(sorted(counts_by_label))
        total_documents = len(materialized)
        priors = MappingProxyType(
            {label: math.log(document_counts[label] / total_documents) for label in labels_ordered}
        )
        token_log_probabilities: dict[str, Mapping[str, float]] = {}
        for label in labels_ordered:
            counts = counts_by_label[label]
            denominator = sum(counts[token] for token in vocabulary) + self._alpha * len(vocabulary)
            if not denominator:
                denominator = self._alpha * max(1, len(vocabulary))
            token_log_probabilities[label] = MappingProxyType(
                {
                    token: math.log((counts[token] + self._alpha) / denominator)
                    for token in vocabulary
                }
            )
        self._state = ClassifierState(
            vocabulary,
            labels_ordered,
            priors,
            MappingProxyType(token_log_probabilities),
            total_documents,
        )
        return self

    def predict_proba(self, conversation: Conversation) -> Mapping[str, float]:
        """Return normalized label probabilities in lexical label order."""
        scores = self._scores(conversation)
        maximum = max(scores.values())
        weights = {label: math.exp(score - maximum) for label, score in scores.items()}
        total = sum(weights.values())
        return MappingProxyType({label: weights[label] / total for label in self.state.labels})

    def predict(self, conversation: Conversation) -> str:
        """Return the highest-scoring label, breaking ties lexically."""
        probabilities = self.predict_proba(conversation)
        return min(self.state.labels, key=lambda label: (-probabilities[label], label))

    def digest(self) -> str:
        """Return a stable digest of fitted parameters and hyperparameters."""
        state = self.state
        payload = {
            "alpha": self._alpha,
            "max_features": self._max_features,
            "vocabulary": state.vocabulary,
            "labels": state.labels,
            "priors": dict(state.priors),
            "token_log_probabilities": {
                label: dict(values) for label, values in state.token_log_probabilities.items()
            },
            "documents": state.documents,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def save(self, path: str | Path) -> None:
        """Write a self-contained JSON model artifact."""
        state = self.state
        payload = {
            "format": "turnscope.classifier/v1",
            "alpha": self._alpha,
            "max_features": self._max_features,
            "vocabulary": list(state.vocabulary),
            "labels": list(state.labels),
            "priors": dict(state.priors),
            "token_log_probabilities": {
                label: dict(values) for label, values in state.token_log_probabilities.items()
            },
            "documents": state.documents,
            "digest": self.digest(),
        }
        Path(path).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> ConversationClassifier:
        """Load and authenticate a JSON model artifact."""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load classifier artifact: {error}") from error
        if not isinstance(payload, dict) or payload.get("format") != "turnscope.classifier/v1":
            raise ValueError("unsupported classifier artifact format")
        model = cls(alpha=payload.get("alpha", 1.0), max_features=payload.get("max_features"))
        try:
            vocabulary = tuple(payload["vocabulary"])
            labels = tuple(payload["labels"])
            priors = MappingProxyType(
                {str(key): float(value) for key, value in payload["priors"].items()}
            )
            probabilities = MappingProxyType(
                {
                    str(label): MappingProxyType(
                        {str(token): float(value) for token, value in values.items()}
                    )
                    for label, values in payload["token_log_probabilities"].items()
                }
            )
            documents = int(payload["documents"])
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ValueError("invalid classifier artifact fields") from error
        if (
            not vocabulary
            or not all(isinstance(token, str) and token for token in vocabulary)
            or len(set(vocabulary)) != len(vocabulary)
            or not labels
            or tuple(sorted(labels)) != labels
            or documents < 1
            or set(probabilities) != set(labels)
            or any(set(values) != set(vocabulary) for values in probabilities.values())
        ):
            raise ValueError("invalid classifier artifact values")
        model._state = ClassifierState(vocabulary, labels, priors, probabilities, documents)
        if payload.get("digest") != model.digest():
            raise ValueError("classifier artifact digest mismatch")
        return model

    def _scores(self, conversation: Conversation) -> dict[str, float]:
        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        state = self.state
        counts = Counter(token for token in _tokens(conversation) if token in state.vocabulary)
        return {
            label: state.priors[label]
            + sum(
                count * state.token_log_probabilities[label][token]
                for token, count in counts.items()
            )
            for label in state.labels
        }
