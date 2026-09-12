"""Causal hierarchical event forecasting with separate epoch and alert selection.

The optional neural dependency is loaded only for fitted inference; Torch is
loaded only by fitting. Numerical model construction is original and does not
load pretrained checkpoints or delegate predictions to another project.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from .forecast_data import ForecastExample, ForecastPrefix
from .forecast_metrics import choose_threshold, forecast_metrics
from .models import Conversation
from .neural_forecast_data import (
    ObservedPrefix,
    SequenceForecastDataset,
    SequenceLimits,
    SequencePolicy,
    _digest,
    observed_prefix,
    prepare_sequence_forecasts,
)
from .neural_forecast_math import (
    FrozenNeuralParameters,
    NeuralArchitecture,
    NeuralNumericLimits,
    _integer,
    admit_encoded_turns,
    infer_encoded_turns,
)
from .neural_forecast_train import NeuralTrainingConfig, NeuralTrainingResult, train_sequence_model
from .neural_token_data import SequenceVocabulary, encode_observed_prefix, fit_sequence_vocabulary

if TYPE_CHECKING:
    from .neural_forecast_artifact import NeuralArtifactLimits, NeuralSaveResult

MODEL_FORMAT = "turnscope.hierarchical-gru-forecast.v1"
PROBABILITY_FLOOR = 1e-15


@dataclass(frozen=True, slots=True)
class NeuralForecastConfig:
    embedding_dim: int = 64
    word_hidden: int = 64
    turn_hidden: int = 64
    word_layers: int = 1
    turn_layers: int = 1
    max_turn_tokens: int = 128
    long_turn_policy: str = "head"
    max_inference_affine_multiplications: int = 100_000_000_000

    def __post_init__(self) -> None:
        self.architecture(3)
        _integer(self.max_turn_tokens, "max_turn_tokens", 1, 1024)
        _integer(
            self.max_inference_affine_multiplications,
            "max_inference_affine_multiplications",
            1,
            1_000_000_000_000,
        )
        if type(self.long_turn_policy) is not str or self.long_turn_policy not in (
            "head",
            "reject",
        ):
            raise ValueError("long_turn_policy must be 'head' or 'reject'")

    def architecture(self, vocabulary_size: int) -> NeuralArchitecture:
        return NeuralArchitecture(
            vocabulary_size,
            self.embedding_dim,
            self.word_hidden,
            self.turn_hidden,
            self.word_layers,
            self.turn_layers,
        )

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NeuralForecastState:
    config: NeuralForecastConfig
    policy: SequencePolicy
    data_limits: SequenceLimits
    vocabulary: SequenceVocabulary
    training: NeuralTrainingResult
    threshold: float
    policy_validation_balanced_accuracy: float
    training_groups: frozenset[str]
    validation_groups: frozenset[str]
    policy_validation_groups: frozenset[str]
    policy_validation_digest: str
    policy_validation_conversations: int
    policy_validation_prefixes: int
    policy_reuses_model_validation: bool
    _identity: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # All members are immutable. Hash arrays once, not for every prediction.
        object.__setattr__(
            self,
            "_identity",
            _digest(
                {
                    "summary": self.summary(),
                    "training_groups": sorted(self.training_groups),
                    "model_validation_groups": sorted(self.validation_groups),
                    "policy_validation_groups": sorted(self.policy_validation_groups),
                }
            ),
        )

    @property
    def parameters(self) -> FrozenNeuralParameters:
        return self.training.parameters

    def summary(self) -> dict[str, Any]:
        result = self.training
        return {
            "format": MODEL_FORMAT,
            "config": self.config.to_dict(),
            "eligibility_policy": self.policy.to_dict(),
            "data_limits": self.data_limits.to_dict(),
            "numeric_limits": self.parameters.limits.to_dict(),
            "training_config": result.config.to_dict(),
            "parameter_digest": self.parameters.digest,
            "vocabulary_digest": self.vocabulary.digest,
            "vocabulary_size": self.vocabulary.size,
            "parameter_count": self.parameters.architecture.parameter_count,
            "training_partition_digest": result.training_partition_digest,
            "model_validation_partition_digest": result.validation_partition_digest,
            "policy_validation_partition_digest": self.policy_validation_digest,
            "training_conversations": result.training_conversations,
            "model_validation_conversations": result.validation_conversations,
            "policy_validation_conversations": self.policy_validation_conversations,
            "training_prefixes": result.training_prefixes,
            "model_validation_prefixes": result.validation_prefixes,
            "policy_validation_prefixes": self.policy_validation_prefixes,
            "policy_reuses_model_validation": self.policy_reuses_model_validation,
            "selected_epoch": result.selected_epoch,
            "history": [asdict(epoch) for epoch in result.history],
            "initial_parameter_sha256": dict(result.initial_parameter_sha256),
            "changed_parameter_names": list(result.changed_parameter_names),
            "maximum_estimated_workspace_bytes": result.maximum_estimated_workspace_bytes,
            "estimated_affine_multiplications": result.estimated_affine_multiplications,
            "torch_version": result.torch_version,
            "numpy_version": result.numpy_version,
            "torch_threads": result.torch_threads,
            "threshold": self.threshold,
            "threshold_rule": "probability >= threshold",
            "threshold_objective": (
                "conversation-max balanced accuracy; higher-threshold exact ties"
            ),
            "policy_validation_balanced_accuracy": self.policy_validation_balanced_accuracy,
            "probability_floor": PROBABILITY_FLOOR,
            "probability_calibration_claimed": False,
        }

    @property
    def digest(self) -> str:
        return self._identity


@dataclass(frozen=True, slots=True)
class NeuralForecastPrediction:
    logit: float
    probability: float
    alert: bool
    threshold: float
    observed_turns: int
    raw_tokens: int
    retained_tokens: int
    known_tokens: int
    truncated_turns: int
    input_digest: str
    model_digest: str
    affine_multiplications: int
    estimated_workspace_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "retained_token_fraction": self.retained_tokens / self.raw_tokens
            if self.raw_tokens
            else 1.0,
            "known_retained_token_fraction": self.known_tokens / self.retained_tokens
            if self.retained_tokens
            else 0.0,
        }


def _clip(probability: float) -> float:
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("model probability must be finite in [0, 1]")
    return min(1 - PROBABILITY_FLOOR, max(PROBABILITY_FLOOR, probability))


def _metric_examples(data: SequenceForecastDataset) -> tuple[ForecastExample, ...]:
    """Use existing audited decision/loss metrics, without constructing lexical features."""
    examples = []
    for item in data.examples:
        observed = data.observations[item.conversation_index]
        endpoint = observed.turns[item.endpoint]
        examples.append(
            ForecastExample(
                ForecastPrefix(
                    observed.conversation_id, endpoint.id, item.endpoint + 1, endpoint.timestamp, {}
                ),
                item.label,
                item.lead_turns,
            )
        )
    return tuple(examples)


def _dataset_scores(
    data: SequenceForecastDataset,
    vocabulary: SequenceVocabulary,
    parameters: FrozenNeuralParameters,
    limits: SequenceLimits,
    max_affine_multiplications: int,
) -> tuple[float, ...]:
    # Each conversation is encoded once; all earlier endpoints come from its
    # unidirectional states, never from independently copied prefix sequences.
    encoded_inputs = []
    work = 0
    for observation in data.observations:
        encoded = encode_observed_prefix(observation, vocabulary, limits=limits)
        admission = admit_encoded_turns(
            parameters.architecture, encoded.turns, limits=parameters.limits
        )
        work += admission.affine_multiplications
        if work > max_affine_multiplications:
            raise ValueError("inference dataset exceeds max_inference_affine_multiplications")
        encoded_inputs.append(encoded)
    by_conversation = []
    for encoded in encoded_inputs:
        result = infer_encoded_turns(parameters, encoded.turns)
        by_conversation.append(result.probabilities)
    return tuple(
        _clip(by_conversation[item.conversation_index][item.endpoint]) for item in data.examples
    )


class HierarchicalEventForecaster:
    """Train all recurrent levels; choose a policy only on declared validation data.

    Fitting is an atomic replacement of a private candidate. It requires distinct
    training and model-validation groups. A third partition can keep threshold
    selection separate; two-partition reuse is explicit in every state summary.
    This object is not a concurrent training coordinator or resumable optimizer.
    """

    def __init__(
        self,
        *,
        config: NeuralForecastConfig | None = None,
        training_config: NeuralTrainingConfig | None = None,
        policy: SequencePolicy | None = None,
        data_limits: SequenceLimits | None = None,
        numeric_limits: NeuralNumericLimits | None = None,
    ) -> None:
        self._config = config if config is not None else NeuralForecastConfig()
        self._training_config = (
            training_config if training_config is not None else NeuralTrainingConfig()
        )
        self._policy = policy if policy is not None else SequencePolicy()
        self._data_limits = data_limits if data_limits is not None else SequenceLimits()
        self._numeric_limits = (
            numeric_limits if numeric_limits is not None else NeuralNumericLimits()
        )
        for value, expected in (
            (self._config, NeuralForecastConfig),
            (self._training_config, NeuralTrainingConfig),
            (self._policy, SequencePolicy),
            (self._data_limits, SequenceLimits),
            (self._numeric_limits, NeuralNumericLimits),
        ):
            if type(value) is not expected:
                raise ValueError("forecaster configuration and limits must be typed contracts")
        if self._config.max_turn_tokens + 1 > self._numeric_limits.max_turn_tokens:
            raise ValueError("token policy including EOS exceeds the numeric turn limit")
        if self._policy.min_turns > self._data_limits.max_observed_turns:
            raise ValueError("min_turns exceeds the observed-turn budget")
        self._state: NeuralForecastState | None = None

    @property
    def state(self) -> NeuralForecastState:
        if self._state is None:
            raise ValueError("hierarchical event forecaster has not been fitted")
        return self._state

    @property
    def digest(self) -> str:
        return self.state.digest

    @property
    def training_summary(self) -> dict[str, Any]:
        return self.state.summary()

    def save(self, path: str | os.PathLike[str], *, overwrite: bool = False) -> NeuralSaveResult:
        """Publish a validated inference artifact, exclusively by default.

        The result distinguishes successful publication with a cleanup warning
        from failure before publication. Optimizer state is never serialized.
        """
        from .neural_forecast_artifact import save_neural_forecaster

        return save_neural_forecaster(self, path, overwrite=overwrite)

    @staticmethod
    def load(
        path: str | os.PathLike[str], *, limits: NeuralArtifactLimits | None = None
    ) -> HierarchicalEventForecaster:
        """Restore closed inference data without importing Torch or executing code."""
        from .neural_forecast_artifact import load_neural_forecaster

        return load_neural_forecaster(path, limits=limits)

    def prepare(self, conversations: Iterable[Conversation]) -> SequenceForecastDataset:
        return prepare_sequence_forecasts(
            conversations, policy=self._policy, limits=self._data_limits
        )

    def _data(
        self, value: SequenceForecastDataset | Iterable[Conversation]
    ) -> SequenceForecastDataset:
        if isinstance(value, SequenceForecastDataset):
            if type(value) is not SequenceForecastDataset:
                raise ValueError("prepared partitions must use the closed sequence dataset type")
            data = value
        else:
            data = self.prepare(value)
        data.validate_limits(self._data_limits)
        if any(item.endpoint + 1 < self._policy.min_turns for item in data.examples):
            raise ValueError("prepared data contains an endpoint before configured min_turns")
        return data

    def fit(
        self,
        training: SequenceForecastDataset | Iterable[Conversation],
        validation: SequenceForecastDataset | Iterable[Conversation],
        *,
        policy_validation: SequenceForecastDataset | Iterable[Conversation] | None = None,
    ) -> HierarchicalEventForecaster:
        train, validation_data = self._data(training), self._data(validation)
        policy_data = (
            validation_data if policy_validation is None else self._data(policy_validation)
        )
        if train.group_digests & validation_data.group_digests:
            raise ValueError("training and model-validation groups overlap")
        if policy_validation is not None and policy_data.group_digests & (
            train.group_digests | validation_data.group_digests
        ):
            raise ValueError("policy-validation groups overlap another fitting partition")
        if {example.label for example in policy_data.examples} != {False, True}:
            raise ValueError("policy validation requires eligible conversations from both classes")
        vocabulary = fit_sequence_vocabulary(
            train,
            min_document_frequency=self._training_config.min_document_frequency,
            max_features=self._training_config.max_features,
            max_turn_tokens=self._config.max_turn_tokens,
            long_turn_policy=self._config.long_turn_policy,
            limits=self._data_limits,
        )
        architecture = self._config.architecture(vocabulary.size)
        # The third partition is admitted before training rather than discovering
        # an impossible deployment input only after expensive optimizer updates.
        policy_work = 0
        for observation in policy_data.observations:
            encoded = encode_observed_prefix(observation, vocabulary, limits=self._data_limits)
            admission = admit_encoded_turns(
                architecture, encoded.turns, limits=self._numeric_limits
            )
            policy_work += admission.affine_multiplications
            if policy_work > self._config.max_inference_affine_multiplications:
                raise ValueError("policy validation exceeds max_inference_affine_multiplications")
        result = train_sequence_model(
            train,
            validation_data,
            vocabulary,
            architecture,
            config=self._training_config,
            numeric_limits=self._numeric_limits,
            data_limits=self._data_limits,
        )
        scores = _dataset_scores(
            policy_data,
            vocabulary,
            result.parameters,
            self._data_limits,
            self._config.max_inference_affine_multiplications,
        )
        threshold, accuracy = choose_threshold(_metric_examples(policy_data), scores)
        candidate = NeuralForecastState(
            self._config,
            self._policy,
            self._data_limits,
            vocabulary,
            result,
            threshold,
            accuracy,
            train.group_digests,
            validation_data.group_digests,
            policy_data.group_digests,
            policy_data.digest,
            len(policy_data.observations),
            len(policy_data.examples),
            policy_validation is None,
        )
        # Constructing the immutable candidate validates/hashes its identity before
        # the only state-changing operation.
        self._state = candidate
        return self

    def predict(self, observed: ObservedPrefix) -> NeuralForecastPrediction:
        state = self.state
        if type(observed) is not ObservedPrefix:
            raise ValueError("predict requires an ObservedPrefix, not a supervised example")
        if len(observed.turns) < state.policy.min_turns:
            raise ValueError("prediction needs configured min_turns complete observations")
        encoded = encode_observed_prefix(observed, state.vocabulary, limits=state.data_limits)
        admission = admit_encoded_turns(
            state.parameters.architecture, encoded.turns, limits=state.parameters.limits
        )
        if admission.affine_multiplications > state.config.max_inference_affine_multiplications:
            raise ValueError("prediction exceeds max_inference_affine_multiplications")
        result = infer_encoded_turns(state.parameters, encoded.turns)
        probability = _clip(result.probabilities[-1])
        return NeuralForecastPrediction(
            result.logits[-1],
            probability,
            probability >= state.threshold,
            state.threshold,
            len(observed.turns),
            encoded.raw_tokens,
            encoded.retained_tokens,
            encoded.known_tokens,
            encoded.truncated_turns,
            encoded.source_digest,
            state.digest,
            result.work.affine_multiplications,
            result.work.estimated_workspace_bytes,
        )

    def predict_conversation(self, conversation: Conversation) -> NeuralForecastPrediction:
        state = self.state
        return self.predict(
            observed_prefix(conversation, policy=state.policy, limits=state.data_limits)
        )

    def transform(
        self, observations: Iterable[ObservedPrefix]
    ) -> tuple[NeuralForecastPrediction, ...]:
        state = self.state
        admitted: list[ObservedPrefix] = []
        turns = tokens = source_bytes = affine_work = 0
        for observed in observations:
            if len(admitted) >= state.data_limits.max_conversations:
                raise ValueError("inference collection exceeds max_conversations")
            if type(observed) is not ObservedPrefix:
                raise ValueError("transform requires ObservedPrefix values")
            observed.validate_limits(state.data_limits)
            if len(observed.turns) < state.policy.min_turns:
                raise ValueError("prediction needs configured min_turns complete observations")
            turns += len(observed.turns)
            source_bytes += len(observed.conversation_id.encode("utf-8")) + sum(
                len(turn.id.encode("utf-8")) + len(turn.text.encode("utf-8"))
                for turn in observed.turns
            )
            if (
                turns > state.data_limits.max_source_turns
                or source_bytes > state.data_limits.max_source_bytes
            ):
                raise ValueError("inference collection exceeds aggregate source budgets")
            encoded = encode_observed_prefix(observed, state.vocabulary, limits=state.data_limits)
            tokens += encoded.raw_tokens
            admission = admit_encoded_turns(
                state.parameters.architecture, encoded.turns, limits=state.parameters.limits
            )
            affine_work += admission.affine_multiplications
            if tokens > state.data_limits.max_tokens:
                raise ValueError("inference collection exceeds aggregate token budget")
            if affine_work > state.config.max_inference_affine_multiplications:
                raise ValueError(
                    "inference collection exceeds max_inference_affine_multiplications"
                )
            admitted.append(observed)
        return tuple(self.predict(observed) for observed in admitted)

    def evaluate(self, heldout: SequenceForecastDataset | Iterable[Conversation]) -> dict[str, Any]:
        state = self.state
        data = self._data(heldout)
        if data.group_digests & (
            state.training_groups | state.validation_groups | state.policy_validation_groups
        ):
            raise ValueError("evaluation groups overlap a fitting partition")
        if not data.examples:
            raise ValueError("evaluation needs eligible forecast prefixes")
        scores = _dataset_scores(
            data,
            state.vocabulary,
            state.parameters,
            state.data_limits,
            state.config.max_inference_affine_multiplications,
        )
        return {
            "format": "turnscope.neural-forecast-report.v1",
            "model_digest": state.digest,
            "partition_digest": data.digest,
            "support": data.audit.to_dict(),
            "metrics": forecast_metrics(_metric_examples(data), scores, state.threshold),
            "policy_reuses_model_validation": state.policy_reuses_model_validation,
            "probability_calibration_claimed": False,
            "whole_repository_parity_claimed": False,
        }
