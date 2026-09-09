"""Frozen, conversation-weighted cumulative lexical forecasting of first future events."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .forecast_data import ForecastDataset, ForecastPrefix, prepare_forecast_examples
from .forecast_metrics import choose_threshold, forecast_metrics
from .models import Conversation
from .transformers import _tokens


@dataclass(frozen=True, slots=True)
class ForecastState:
    vocabulary: tuple[str, ...]
    log_probabilities: tuple[Mapping[str, float], Mapping[str, float]]
    positive_prior: float
    training_conversations: int
    training_prefixes: int
    validation_conversations: int
    validation_prefixes: int
    training_groups: frozenset[str]
    validation_groups: frozenset[str]
    threshold: float
    validation_balanced_accuracy: float


@dataclass(frozen=True, slots=True)
class ForecastPrediction:
    probability: float
    alert: bool
    threshold: float
    tokens: int
    known_tokens: int

    def to_dict(self) -> dict[str, object]:
        return {
            "probability": self.probability,
            "alert": self.alert,
            "threshold": self.threshold,
            "tokens": self.tokens,
            "known_tokens": self.known_tokens,
            "coverage": self.known_tokens / self.tokens if self.tokens else 0.0,
        }


class PrefixEventForecaster:
    """A weighted multinomial-NB baseline with validation-only threshold selection.

    Each training conversation has total sample weight one, shared equally by
    its eligible prefixes. Inference uses only cumulative lexical counts. No
    annotations, group identity, full-record length or future text are features.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        max_features: int = 10_000,
        min_turns: int = 2,
        event_field: str = "event",
        skip_field: str = "is_section_header",
        groups_field: str = "forecast_groups",
        max_prefixes: int = 30_000,
        max_prefix_cells: int = 4_000_000,
        max_tokens: int = 2_000_000,
    ) -> None:
        if type(alpha) not in (int, float):
            raise ValueError("alpha must be a finite positive number")
        try:
            alpha = float(alpha)
        except OverflowError as error:
            raise ValueError("alpha must be a finite positive number") from error
        if not math.isfinite(alpha) or not 1e-12 <= alpha <= 1e12:
            raise ValueError("alpha must be in [1e-12, 1e12]")
        for name, value, limit in (
            ("max_features", max_features, 100_000),
            ("min_turns", min_turns, 10_000),
            ("max_prefixes", max_prefixes, 100_000),
            ("max_prefix_cells", max_prefix_cells, 16_000_000),
            ("max_tokens", max_tokens, 10_000_000),
        ):
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"{name} must be an integer in [1, {limit}]")
        for name, field in (
            ("event_field", event_field),
            ("skip_field", skip_field),
            ("groups_field", groups_field),
        ):
            if not isinstance(field, str) or not field.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if event_field == skip_field:
            raise ValueError("event_field and skip_field must differ")
        self._config: dict[str, Any] = {
            "alpha": alpha,
            "max_features": max_features,
            "min_turns": min_turns,
            "event_field": event_field,
            "skip_field": skip_field,
            "groups_field": groups_field,
            "max_prefixes": max_prefixes,
            "max_prefix_cells": max_prefix_cells,
            "max_tokens": max_tokens,
        }
        self._state: ForecastState | None = None

    @property
    def state(self) -> ForecastState:
        if self._state is None:
            raise ValueError("prefix forecaster has not been fitted")
        return self._state

    def prepare(self, conversations: Iterable[Conversation]) -> ForecastDataset:
        """Expose exact supervised eligibility without fitting anything."""
        return prepare_forecast_examples(
            conversations,
            **{
                key: value
                for key, value in self._config.items()
                if key not in {"alpha", "max_features"}
            },
        )

    def fit(
        self, training: Iterable[Conversation], validation: Iterable[Conversation]
    ) -> PrefixEventForecaster:
        """Fit vocabulary/likelihoods on train; choose only the threshold on validation."""
        train, val = self.prepare(training), self.prepare(validation)
        if train.group_digests & val.group_digests:
            raise ValueError("training and validation groups overlap")
        if not train.examples or not val.examples:
            raise ValueError("training and validation need eligible forecast prefixes")
        labels: dict[str, bool] = {}
        latest: dict[str, ForecastPrefix] = {}
        frequencies: Counter[str] = Counter()
        for example in train.examples:
            identifier = example.prefix.conversation_id
            labels[identifier] = example.label
            latest[identifier] = example.prefix
            frequencies[identifier] += 1
        if set(labels.values()) != {False, True}:
            raise ValueError("training requires eligible conversations from both classes")
        document_frequency: Counter[str] = Counter()
        for prefix in latest.values():
            document_frequency.update(prefix.counts.keys())
        vocabulary = tuple(
            sorted(document_frequency, key=lambda term: (-document_frequency[term], term))[
                : self._config["max_features"]
            ]
        )
        known = set(vocabulary)
        counts: tuple[Counter[str], Counter[str]] = (Counter(), Counter())
        # Canonical order fixes accumulation order when input conversations are permuted.
        for example in sorted(
            train.examples, key=lambda item: (item.prefix.conversation_id, item.prefix.turns)
        ):
            weight = 1 / frequencies[example.prefix.conversation_id]
            counts[int(example.label)].update(
                {
                    term: weight * count
                    for term, count in example.prefix.counts.items()
                    if term in known
                }
            )
        probabilities = []
        for class_counts in counts:
            denominator = math.fsum(class_counts.values()) + self._config["alpha"] * len(vocabulary)
            probabilities.append(
                MappingProxyType(
                    {
                        term: math.log((class_counts[term] + self._config["alpha"]) / denominator)
                        for term in vocabulary
                    }
                )
            )
        prior = sum(labels.values()) / len(labels)
        new_state = ForecastState(
            vocabulary,
            (probabilities[0], probabilities[1]),
            prior,
            train.conversations,
            len(train.examples),
            val.conversations,
            len(val.examples),
            train.group_digests,
            val.group_digests,
            0.5,
            0.0,
        )
        # A temporary model makes failed validation tuning an all-or-nothing refit.
        candidate = PrefixEventForecaster(**self._config)
        candidate._state = new_state
        scores = [candidate.score_prefix(example.prefix).probability for example in val.examples]
        threshold, accuracy = choose_threshold(val.examples, scores)
        self._state = ForecastState(
            vocabulary,
            new_state.log_probabilities,
            prior,
            train.conversations,
            len(train.examples),
            val.conversations,
            len(val.examples),
            train.group_digests,
            val.group_digests,
            threshold,
            accuracy,
        )
        return self

    def _prediction(self, counts: Mapping[str, int]) -> ForecastPrediction:
        state = self.state
        if any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError("forecast counts must be non-negative integers")
        total = sum(counts.values())
        if total > self._config["max_tokens"]:
            raise ValueError("forecast token budget exceeded")
        negative, positive = state.log_probabilities
        log_odds = math.log(state.positive_prior) - math.log1p(-state.positive_prior)
        log_odds += math.fsum(
            count * (positive[term] - negative[term])
            for term, count in counts.items()
            if term in positive
        )
        if log_odds >= 0:
            probability = 1 / (1 + math.exp(-log_odds))
        else:
            exponent = math.exp(log_odds)
            probability = exponent / (1 + exponent)
        # Binary64 rounding otherwise produces exact 0/1 and infinite log loss.
        probability = min(1 - 1e-15, max(1e-15, probability))
        return ForecastPrediction(
            probability,
            probability >= state.threshold,
            state.threshold,
            total,
            sum(count for term, count in counts.items() if term in positive),
        )

    def score_prefix(self, prefix: ForecastPrefix) -> ForecastPrediction:
        if not isinstance(prefix, ForecastPrefix):
            raise TypeError("prefix must be a ForecastPrefix")
        return self._prediction(prefix.counts)

    def predict(self, observed: Conversation) -> ForecastPrediction:
        """Score an observed prefix using only text and the configured non-observation mask.

        The boolean skip_field excludes headers consistently with supervised
        preparation. Outcome/event labels and other metadata are never read.
        """
        if not isinstance(observed, Conversation):
            raise TypeError("observed prefix must be a Conversation")
        if len(observed.by_id()) != len(observed.utterances):
            raise ValueError("forecast utterance IDs must be unique")
        selected = []
        previous = None
        for item in observed.utterances:
            if previous is not None and item.timestamp < previous:
                raise ValueError("forecast timestamps must be nondecreasing")
            previous = item.timestamp
            skipped = item.metadata.get(self._config["skip_field"], False)
            if type(skipped) is not bool:
                raise ValueError("forecast non-observation mask must be boolean")
            if not skipped:
                selected.append(item)
        if len(selected) < self._config["min_turns"]:
            raise ValueError("observed prefix is shorter than min_turns")
        return self._prediction(Counter(token for item in selected for token in _tokens(item.text)))

    def evaluate(self, heldout: Iterable[Conversation]) -> dict[str, Any]:
        """Evaluate unseen groups with frozen parameters and the validation-chosen threshold."""
        state = self.state
        dataset = self.prepare(heldout)
        if dataset.group_digests & (state.training_groups | state.validation_groups):
            raise ValueError("heldout groups overlap training or validation groups")
        scores = [self.score_prefix(example.prefix).probability for example in dataset.examples]
        return {
            "model_digest": self.digest,
            "excluded_conversations": dataset.excluded_conversations,
            "group_count": len(dataset.group_digests),
            "model": forecast_metrics(dataset.examples, scores, state.threshold),
            "training_prior_baseline": forecast_metrics(
                dataset.examples, [state.positive_prior] * len(scores), state.threshold
            ),
        }

    @property
    def digest(self) -> str:
        return str(self.to_dict()["sha256"])

    def to_dict(self) -> dict[str, Any]:
        from .forecast_artifact import encode_forecaster

        return encode_forecaster(self)

    @classmethod
    def from_dict(cls, value: object) -> PrefixEventForecaster:
        from .forecast_artifact import decode_forecaster

        return decode_forecaster(value)

    def save(self, path: str | Path) -> None:
        from .forecast_artifact import save_forecaster

        save_forecaster(self.to_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> PrefixEventForecaster:
        from .forecast_artifact import load_forecaster

        return cls.from_dict(load_forecaster(Path(path)))
