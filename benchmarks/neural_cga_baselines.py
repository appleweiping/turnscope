"""Input-matched, training-only lexical/prior comparators for prepared neural data.

No dataset loading, model training, network access or test-set selection happens
on import. These are experimental comparators, not public pretrained models.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from turnscope.forecast_data import ForecastExample, ForecastPrefix
from turnscope.forecast_metrics import choose_threshold, forecast_metrics
from turnscope.neural_forecast_data import (
    ObservedPrefix,
    SequenceForecastDataset,
    SequenceLimitError,
    SequenceLimits,
)
from turnscope.neural_token_data import (
    EOS_ID,
    PAD_ID,
    UNK_ID,
    EncodedPrefix,
    SequenceVocabulary,
    encode_observed_prefix,
    fit_sequence_vocabulary,
)

FORMAT = "turnscope.neural-cga-baselines.v1"
PROBABILITY_FLOOR = 1e-15


def _json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def protocol() -> dict[str, Any]:
    """Frozen before outcomes; returned copies cannot change the fitting policy."""
    return {
        "format": FORMAT,
        "features": "prepared observed text only; cumulative multinomial counts",
        "min_document_frequency": 1,
        "max_features": 10000,
        "max_turn_tokens": 128,
        "long_turn_policy": "head",
        "turn_encoding": "up to 128 lexical IDs then one EOS; UNK retained; PAD forbidden",
        "smoothing": {"alpha": 1.0, "support": "UNK, EOS and fitted lexical rows; no PAD"},
        "training_weight": "each eligible conversation totals 1 across its eligible prefixes",
        "prior": "positive eligible training conversations / eligible training conversations",
        "policy_selection": "separate conversation-max balanced-accuracy threshold per baseline",
        "threshold_tie": "highest threshold among 0, 1 and policy conversation maxima",
        "decision": "probability >= threshold",
        "probability_floor": PROBABILITY_FLOOR,
        "epoch_selection_used": False,
        "model_validation_used": False,
        "test_tuning": False,
    }


@dataclass(frozen=True, slots=True)
class NeuralBaselineLimits:
    """Per-partition encoding/update bounds and fitted scalar-array admission."""

    max_encoded_tokens: int = 2_000_000
    max_count_updates: int = 2_000_000
    max_parameter_cells: int = 100_000

    def __post_init__(self) -> None:
        for name, maximum in (
            ("max_encoded_tokens", 10_000_000),
            ("max_count_updates", 10_000_000),
            ("max_parameter_cells", 1_000_000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be a bounded positive integer")


def _bounds(
    data_limits: SequenceLimits | None, limits: NeuralBaselineLimits | None
) -> tuple[SequenceLimits, NeuralBaselineLimits]:
    data = SequenceLimits() if data_limits is None else data_limits
    work = NeuralBaselineLimits() if limits is None else limits
    if type(data) is not SequenceLimits or type(work) is not NeuralBaselineLimits:
        raise ValueError("baseline bounds require their exact typed contracts")
    return data, work


def _admit(data: SequenceForecastDataset, bounds: SequenceLimits) -> str:
    if type(data) is not SequenceForecastDataset:
        raise ValueError("baselines accept only prepared SequenceForecastDataset values")
    data.validate_limits(bounds)
    if not data.examples:
        raise ValueError("baselines require eligible forecast prefixes")
    return data.digest


def _unchanged(data: SequenceForecastDataset, digest: str) -> None:
    if data.digest != digest:
        raise ValueError("prepared baseline input changed during execution")


def _metric_examples(data: SequenceForecastDataset) -> tuple[ForecastExample, ...]:
    # Metric handles carry labels separately and no lexical counts/raw text.
    result = []
    for item in data.examples:
        observed = data.observations[item.conversation_index]
        turn = observed.turns[item.endpoint]
        result.append(
            ForecastExample(
                ForecastPrefix(
                    observed.conversation_id, turn.id, item.endpoint + 1, turn.timestamp, {}
                ),
                item.label,
                item.lead_turns,
            )
        )
    return tuple(result)


def _handles(data: SequenceForecastDataset) -> tuple[tuple[int, ...], ...]:
    indices: list[list[int]] = [[] for _ in data.observations]
    for index, example in enumerate(data.examples):
        indices[example.conversation_index].append(index)
    return tuple(tuple(row) for row in indices)


@dataclass
class _EncodingAudit:
    observations: int = 0
    observed_turns: int = 0
    raw_tokens: int = 0
    retained_tokens: int = 0
    known_tokens: int = 0
    truncated_turns: int = 0
    encoded_tokens_including_eos: int = 0
    count_updates: int = 0

    def add(self, encoded: EncodedPrefix, limits: NeuralBaselineLimits) -> None:
        self.observations += 1
        self.observed_turns += len(encoded.turns)
        for field in ("raw_tokens", "retained_tokens", "known_tokens", "truncated_turns"):
            setattr(self, field, getattr(self, field) + getattr(encoded, field))
        self.encoded_tokens_including_eos += encoded.retained_tokens + len(encoded.turns)
        # Check before the count dictionaries or fitting/scoring updates.
        if self.encoded_tokens_including_eos > limits.max_encoded_tokens:
            raise SequenceLimitError("baseline encoded-token budget exceeded")

    def turn(self, ids: tuple[int, ...], limits: NeuralBaselineLimits) -> Counter[int]:
        counts = Counter(ids)
        self.count_updates += len(counts)
        if self.count_updates > limits.max_count_updates:
            raise SequenceLimitError("baseline count-update budget exceeded")
        return counts

    def report(self) -> dict[str, int | float]:
        return {
            **asdict(self),
            "unknown_tokens": self.retained_tokens - self.known_tokens,
            "eos_tokens": self.observed_turns,
            "retained_token_fraction": self.retained_tokens / self.raw_tokens
            if self.raw_tokens
            else 1.0,
        }


def _encode(
    observed: ObservedPrefix, vocabulary: SequenceVocabulary, bounds: SequenceLimits
) -> EncodedPrefix:
    """Feature boundary: neither labels, groups nor future source fields are accepted."""
    encoded = encode_observed_prefix(observed, vocabulary, limits=bounds)
    if (
        encoded.source_digest != observed.digest
        or encoded.vocabulary_digest != vocabulary.digest
        or encoded.vocabulary_size != vocabulary.size
    ):
        raise ValueError("encoding is not bound to the supplied observation/vocabulary")
    if any(PAD_ID in turn or turn[-1] != EOS_ID for turn in encoded.turns):
        raise ValueError("baseline features require no PAD and one terminal EOS")
    return encoded


def _logs(counts: tuple[tuple[float, ...], tuple[float, ...]]) -> tuple[tuple[float, ...], ...]:
    result = []
    for row in counts:
        denominator = math.log(math.fsum(row) + len(row))
        result.append(tuple(math.log(value + 1.0) - denominator for value in row))
    return tuple(result)


def _probability(log_odds: float) -> float:
    if not math.isfinite(log_odds):
        raise ValueError("baseline log odds must be finite")
    if log_odds >= 0:
        score = 1 / (1 + math.exp(-log_odds))
    else:
        odds = math.exp(log_odds)
        score = odds / (1 + odds)
    return min(1 - PROBABILITY_FLOOR, max(PROBABILITY_FLOOR, score))


def _scores(
    data: SequenceForecastDataset,
    vocabulary: SequenceVocabulary,
    likelihoods: tuple[tuple[float, ...], ...],
    prior: float,
    bounds: SequenceLimits,
    limits: NeuralBaselineLimits,
) -> tuple[tuple[float, ...], dict[str, int | float]]:
    handles = _handles(data)
    ratios = tuple(positive - negative for negative, positive in zip(*likelihoods, strict=True))
    scores = []
    audit = _EncodingAudit()
    for observed, indices in zip(data.observations, handles, strict=True):
        encoded = _encode(observed, vocabulary, bounds)
        audit.add(encoded, limits)
        current = math.log(prior) - math.log1p(-prior)
        endpoint_to_index = {data.examples[index].endpoint: index for index in indices}
        for endpoint, ids in enumerate(encoded.turns):
            counts = audit.turn(ids, limits)
            current = math.fsum(
                [current, *(count * ratios[token - 1] for token, count in counts.items())]
            )
            if endpoint in endpoint_to_index:
                scores.append(_probability(current))
    if len(scores) != len(data.examples):
        raise ValueError("baseline prediction inventory mismatch")
    return tuple(scores), audit.report()


@dataclass(frozen=True, slots=True)
class NeuralBaselineState:
    vocabulary: SequenceVocabulary
    class_token_counts: tuple[tuple[float, ...], tuple[float, ...]]
    log_likelihoods: tuple[tuple[float, ...], ...]
    prior: float
    lexical_threshold: float
    prior_threshold: float
    training_digest: str
    policy_validation_digest: str
    training_groups: frozenset[str]
    policy_validation_groups: frozenset[str]
    data_limits: SequenceLimits
    limits: NeuralBaselineLimits
    _provenance_json: bytes

    def __post_init__(self) -> None:
        _bounds(self.data_limits, self.limits)
        if type(self.vocabulary) is not SequenceVocabulary:
            raise ValueError("baseline vocabulary must be immutable SequenceVocabulary")
        if type(self.class_token_counts) is not tuple or len(self.class_token_counts) != 2:
            raise ValueError("baseline requires negative/positive count rows")
        for row in self.class_token_counts:
            if type(row) is not tuple or len(row) != self.vocabulary.size - 1:
                raise ValueError("count rows exclude PAD and include UNK/EOS/lexical IDs")
            if any(type(value) is not float or not 0 <= value <= 1e12 for value in row):
                raise ValueError("token counts must be bounded finite non-negative floats")
        if self.log_likelihoods != _logs(self.class_token_counts):
            raise ValueError("likelihoods disagree with alpha-one counts")
        for name in ("prior", "lexical_threshold", "prior_threshold"):
            value = getattr(self, name)
            if type(value) is not float or not 0 <= value <= 1:
                raise ValueError("baseline probabilities must be finite floats in [0, 1]")
        if not 0 < self.prior < 1:
            raise ValueError("training requires eligible conversations from both classes")
        for groups in (self.training_groups, self.policy_validation_groups):
            if type(groups) is not frozenset:
                raise ValueError("baseline groups must be immutable")
        if self.training_groups & self.policy_validation_groups:
            raise ValueError("training and policy-validation groups overlap")
        if type(self._provenance_json) is not bytes:
            raise ValueError("baseline provenance must be immutable encoded JSON")

    def _body(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "protocol": protocol(),
            "vocabulary_digest": self.vocabulary.digest,
            "class_token_counts": self.class_token_counts,
            "log_likelihoods": self.log_likelihoods,
            "prior": self.prior,
            "lexical_threshold": self.lexical_threshold,
            "prior_threshold": self.prior_threshold,
            "training_digest": self.training_digest,
            "policy_validation_digest": self.policy_validation_digest,
            "training_groups": sorted(self.training_groups),
            "policy_validation_groups": sorted(self.policy_validation_groups),
            "data_limits": self.data_limits.to_dict(),
            "limits": asdict(self.limits),
            "provenance": json.loads(self._provenance_json),
        }

    @property
    def digest(self) -> str:
        return _digest(self._body())


@dataclass(frozen=True, slots=True)
class FittedNeuralBaselines:
    state: NeuralBaselineState

    def __post_init__(self) -> None:
        if type(self.state) is not NeuralBaselineState:
            raise ValueError("fitted baselines require immutable baseline state")

    def summary(self) -> dict[str, Any]:
        state = self.state
        return {
            "format": FORMAT,
            "state_digest": state.digest,
            "protocol": protocol(),
            "protocol_digest": _digest(protocol()),
            "vocabulary_digest": state.vocabulary.digest,
            "vocabulary_size_including_reserved": state.vocabulary.size,
            "smoothing_categories_excluding_pad": state.vocabulary.size - 1,
            "prior": state.prior,
            "lexical_threshold": state.lexical_threshold,
            "prior_threshold": state.prior_threshold,
            "training_digest": state.training_digest,
            "policy_validation_digest": state.policy_validation_digest,
            "data_limits": state.data_limits.to_dict(),
            "limits": asdict(state.limits),
            "provenance": json.loads(state._provenance_json),
            "model_capacity_matched": False,
            "probability_calibration_claimed": False,
            "whole_repository_parity_claimed": False,
        }

    def evaluate(self, heldout: SequenceForecastDataset) -> dict[str, Any]:
        state = self.state
        before = state.digest
        source = _admit(heldout, state.data_limits)
        if heldout.group_digests & (state.training_groups | state.policy_validation_groups):
            raise ValueError("heldout groups overlap baseline fitting partitions")
        scores, encoding = _scores(
            heldout,
            state.vocabulary,
            state.log_likelihoods,
            state.prior,
            state.data_limits,
            state.limits,
        )
        examples = _metric_examples(heldout)
        priors = (state.prior,) * len(examples)
        result = {
            "format": "turnscope.neural-cga-baseline-report.v1",
            "state_digest": before,
            "partition_digest": source,
            "observation_digest": heldout.observation_digest,
            "support": heldout.audit.to_dict(),
            "encoding": encoding,
            "lexical_nb": forecast_metrics(examples, scores, state.lexical_threshold),
            "prior": forecast_metrics(examples, priors, state.prior_threshold),
            "prediction_inventory_digest": _digest({"lexical_nb": scores, "prior": priors}),
            "epoch_selection_used": False,
            "model_validation_used": False,
            "test_tuning": False,
            "probability_calibration_claimed": False,
            "whole_repository_parity_claimed": False,
        }
        _unchanged(heldout, source)
        if state.digest != before:
            raise ValueError("frozen baseline state changed during evaluation")
        return result


def fit_neural_baselines(
    training: SequenceForecastDataset,
    policy_validation: SequenceForecastDataset,
    *,
    data_limits: SequenceLimits | None = None,
    limits: NeuralBaselineLimits | None = None,
) -> FittedNeuralBaselines:
    """Fit on training; select two independent policies, never read model validation/test."""
    bounds, work = _bounds(data_limits, limits)
    training_digest = _admit(training, bounds)
    policy_digest = _admit(policy_validation, bounds)
    if training.group_digests & policy_validation.group_digests:
        raise ValueError("training and policy-validation groups overlap, including exclusions")
    for data in (training, policy_validation):
        if not data.audit.positive_conversations or not data.audit.negative_conversations:
            raise ValueError("fitting partitions require eligible conversations from both classes")
    vocabulary = fit_sequence_vocabulary(
        training,
        min_document_frequency=1,
        max_features=10000,
        max_turn_tokens=128,
        long_turn_policy="head",
        limits=bounds,
    )
    vocabulary_digest = vocabulary.digest
    # Two count rows and two likelihood rows, no dense per-prefix matrices.
    if 4 * (vocabulary.size - 1) > work.max_parameter_cells:
        raise SequenceLimitError("baseline parameter-cell budget exceeded")
    counts = [[0.0] * (vocabulary.size - 1) for _ in range(2)]
    audit = _EncodingAudit()
    handles = _handles(training)
    for observed, indices in zip(training.observations, handles, strict=True):
        encoded = _encode(observed, vocabulary, bounds)
        audit.add(encoded, work)
        endpoints = tuple(training.examples[index].endpoint for index in indices)
        row = counts[int(training.examples[indices[0]].label)]
        # A token in turn t occurs in every eligible prefix ending at/after t.
        # Summing these weights is algebraically the average of cumulative counts.
        for turn_index, ids in enumerate(encoded.turns):
            weight = (len(endpoints) - bisect_left(endpoints, turn_index)) / len(endpoints)
            for token, count in audit.turn(ids, work).items():
                row[token - 1] += count * weight
    frozen_counts = (tuple(counts[0]), tuple(counts[1]))
    likelihoods = _logs(frozen_counts)
    prior = training.audit.positive_conversations / training.audit.eligible_conversations
    scores, policy_encoding = _scores(
        policy_validation, vocabulary, likelihoods, prior, bounds, work
    )
    examples = _metric_examples(policy_validation)
    lexical_threshold, lexical_ba = choose_threshold(examples, scores)
    prior_threshold, prior_ba = choose_threshold(examples, (prior,) * len(examples))
    _unchanged(training, training_digest)
    _unchanged(policy_validation, policy_digest)
    if vocabulary.digest != vocabulary_digest:
        raise ValueError("frozen training vocabulary changed during fitting")
    state = NeuralBaselineState(
        vocabulary,
        frozen_counts,
        likelihoods,
        prior,
        lexical_threshold,
        prior_threshold,
        training_digest,
        policy_digest,
        training.group_digests,
        policy_validation.group_digests,
        bounds,
        work,
        _json(
            {
                "training_support": training.audit.to_dict(),
                "policy_validation_support": policy_validation.audit.to_dict(),
                "training_observation_digest": training.observation_digest,
                "policy_validation_observation_digest": policy_validation.observation_digest,
                "training_encoding": audit.report(),
                "policy_validation_encoding": policy_encoding,
                "weighted_token_mass_negative_positive": [math.fsum(row) for row in counts],
                "policy_validation_lexical_balanced_accuracy": lexical_ba,
                "policy_validation_prior_balanced_accuracy": prior_ba,
                "policy_validation_lexical_metrics": forecast_metrics(
                    examples, scores, lexical_threshold
                ),
                "policy_validation_prior_metrics": forecast_metrics(
                    examples, (prior,) * len(examples), prior_threshold
                ),
                "class_order": ["negative", "positive"],
                "reserved_ids": {"pad": PAD_ID, "unk": UNK_ID, "eos": EOS_ID},
            }
        ),
    )
    return FittedNeuralBaselines(state)
