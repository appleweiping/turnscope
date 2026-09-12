"""Separate three-partition forecasting workflow for explicitly named controls.

These models never reinterpret a main-model artifact. All fitting candidates
select their own epoch and policy threshold; heldout evaluation cannot overlap
any fitting group. This module performs no file or network I/O.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from .forecast_metrics import choose_threshold, forecast_metrics
from .models import Conversation
from .neural_ablation_math import (
    AblationPoolingLimits,
    FrozenAblationParameters,
    ablation_numerical_version,
    admit_ablation_turns,
    infer_ablation_turns,
)
from .neural_ablation_train import (
    AblationTrainingLimits,
    AblationTrainingResult,
    train_ablation_model,
)
from .neural_forecast import PROBABILITY_FLOOR, _clip, _metric_examples
from .neural_forecast_data import (
    ObservedPrefix,
    SequenceForecastDataset,
    SequenceLimits,
    SequencePolicy,
    _digest,
    observed_prefix,
    prepare_sequence_forecasts,
)
from .neural_forecast_math import NeuralArchitecture, NeuralNumericLimits, _integer
from .neural_forecast_train import NeuralTrainingConfig, _real
from .neural_token_data import (
    EncodedPrefix,
    SequenceVocabulary,
    encode_observed_prefix,
    fit_sequence_vocabulary,
)

if TYPE_CHECKING:
    from .neural_forecast_artifact import NeuralArtifactLimits, NeuralSaveResult

ABLATION_MODEL_FORMAT = "turnscope.neural-ablation-forecast.v1"
_HASH = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class AblationForecastConfig:
    variant: str
    max_turn_tokens: int = 128
    long_turn_policy: str = "head"
    max_inference_affine_multiplications: int = 100_000_000_000
    max_inference_pooling_operations: int = 100_000_000_000

    def __post_init__(self) -> None:
        ablation_numerical_version(self.variant)
        _integer(self.max_turn_tokens, "max_turn_tokens", 1, 1024)
        for name in ("max_inference_affine_multiplications", "max_inference_pooling_operations"):
            _integer(getattr(self, name), name, 1, 1_000_000_000_000)
        if type(self.long_turn_policy) is not str or self.long_turn_policy not in (
            "head",
            "reject",
        ):
            raise ValueError("long_turn_policy must be 'head' or 'reject'")

    def architecture(self, vocabulary_size: int) -> NeuralArchitecture:
        return NeuralArchitecture(vocabulary_size)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _settings(
    config: AblationForecastConfig,
    policy: SequencePolicy,
    data: SequenceLimits,
    numeric: NeuralNumericLimits,
) -> None:
    for value, kind in (
        (config, AblationForecastConfig),
        (policy, SequencePolicy),
        (data, SequenceLimits),
        (numeric, NeuralNumericLimits),
    ):
        if type(value) is not kind:
            raise ValueError("forecaster configuration and limits must be typed contracts")
    if config.max_turn_tokens + 1 > numeric.max_turn_tokens:
        raise ValueError("token policy including EOS exceeds the numeric turn limit")
    if policy.min_turns > min(data.max_observed_turns, numeric.max_observed_turns):
        raise ValueError("min_turns exceeds an observed-turn budget")


@dataclass(frozen=True, slots=True)
class AblationForecastState:
    config: AblationForecastConfig
    policy: SequencePolicy
    data_limits: SequenceLimits
    vocabulary: SequenceVocabulary
    training: AblationTrainingResult
    threshold: float
    policy_validation_balanced_accuracy: float
    training_groups: frozenset[str]
    validation_groups: frozenset[str]
    policy_validation_groups: frozenset[str]
    policy_validation_digest: str
    policy_validation_conversations: int
    policy_validation_prefixes: int
    _identity: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self.training) is not AblationTrainingResult
            or type(self.vocabulary) is not SequenceVocabulary
        ):
            raise ValueError("state requires typed training result and vocabulary")
        _settings(self.config, self.policy, self.data_limits, self.parameters.limits)
        result = self.training
        vocabulary = self.vocabulary
        if self.config.variant != result.variant:
            raise ValueError("state and trained variant differ")
        if self.parameters.architecture != self.config.architecture(vocabulary.size):
            raise ValueError("state architecture differs from the pinned vocabulary")
        vocabulary.validate_limits(self.data_limits)
        if (vocabulary.max_turn_tokens, vocabulary.long_turn_policy) != (
            self.config.max_turn_tokens,
            self.config.long_turn_policy,
        ) or vocabulary.documents != result.training_conversations:
            raise ValueError("state vocabulary disagrees with training support or token policy")
        if (
            vocabulary.digest != result.vocabulary_digest
            or len(vocabulary.tokens) > result.config.max_features
        ):
            raise ValueError("state vocabulary differs from the training result")
        if any(
            value < result.config.min_document_frequency
            for value in vocabulary.document_frequencies
        ):
            raise ValueError("vocabulary violates training minimum document frequency")
        _real(self.threshold, "threshold", 0, 1)
        _real(self.policy_validation_balanced_accuracy, "policy balanced accuracy", 0, 1)
        if type(self.policy_validation_digest) is not str or not _HASH.fullmatch(
            self.policy_validation_digest
        ):
            raise ValueError("policy partition must have a SHA-256 digest")
        supports = (
            (self.training_groups, result.training_conversations, result.training_prefixes),
            (self.validation_groups, result.validation_conversations, result.validation_prefixes),
            (
                self.policy_validation_groups,
                self.policy_validation_conversations,
                self.policy_validation_prefixes,
            ),
        )
        for groups, conversations, prefixes in supports:
            _integer(
                conversations, "partition conversations", 2, self.data_limits.max_conversations
            )
            _integer(prefixes, "partition prefixes", conversations, self.data_limits.max_prefixes)
            if (
                type(groups) is not frozenset
                or not conversations <= len(groups) <= self.data_limits.max_groups
            ):
                raise ValueError("partition groups must be a bounded immutable inventory")
            if any(type(value) is not str or not _HASH.fullmatch(value) for value in groups):
                raise ValueError("partition groups require lowercase SHA-256 digests")
            if prefixes + conversations * (
                self.policy.min_turns - 1
            ) > self.data_limits.max_source_turns or prefixes > conversations * (
                self.data_limits.max_observed_turns - self.policy.min_turns + 1
            ):
                raise ValueError("partition support cannot fit the sequence limits")
        if (
            self.training_groups & self.validation_groups
            or self.training_groups & self.policy_validation_groups
            or self.validation_groups & self.policy_validation_groups
        ):
            raise ValueError("all three fitting partitions require distinct groups")
        if (
            len(
                {
                    result.training_partition_digest,
                    result.validation_partition_digest,
                    self.policy_validation_digest,
                }
            )
            != 3
        ):
            raise ValueError("all three fitting partition digests must differ")
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
    def parameters(self) -> FrozenAblationParameters:
        return self.training.parameters

    @property
    def digest(self) -> str:
        return self._identity

    def summary(self) -> dict[str, Any]:
        result = self.training
        return {
            "format": ABLATION_MODEL_FORMAT,
            "variant": self.config.variant,
            "numerical_version": ablation_numerical_version(self.config.variant),
            "reference_architecture": self.parameters.architecture.to_dict(),
            "config": self.config.to_dict(),
            "eligibility_policy": self.policy.to_dict(),
            "data_limits": self.data_limits.to_dict(),
            "numeric_limits": self.parameters.limits.to_dict(),
            "pooling_limits": self.parameters.pooling_limits.to_dict(),
            "training_config": result.config.to_dict(),
            "training_limits": result.training_limits.to_dict(),
            "initialization_version": result.initialization_version,
            "main_initialization_version": result.main_initialization_version,
            "parameter_digest": self.parameters.digest,
            "vocabulary_digest": self.vocabulary.digest,
            "vocabulary_size": self.vocabulary.size,
            "parameter_count": self.parameters.parameter_count,
            "canonical_main_parameter_count": self.parameters.architecture.parameter_count,
            "fixed_pad_parameters": 64,
            "training_partition_digest": result.training_partition_digest,
            "model_validation_partition_digest": result.validation_partition_digest,
            "policy_validation_partition_digest": self.policy_validation_digest,
            "training_conversations": result.training_conversations,
            "model_validation_conversations": result.validation_conversations,
            "policy_validation_conversations": self.policy_validation_conversations,
            "training_prefixes": result.training_prefixes,
            "model_validation_prefixes": result.validation_prefixes,
            "policy_validation_prefixes": self.policy_validation_prefixes,
            "policy_reuses_model_validation": False,
            "selected_epoch": result.selected_epoch,
            "history": [asdict(epoch) for epoch in result.history],
            "initial_parameter_sha256": dict(result.initial_parameter_sha256),
            "main_initial_parameter_sha256": dict(result.main_initial_parameter_sha256),
            "changed_parameter_names": list(result.changed_parameter_names),
            "maximum_estimated_workspace_bytes": result.maximum_estimated_workspace_bytes,
            "initialization_workspace_bytes": result.initialization_workspace_bytes,
            "estimated_affine_multiplications": result.estimated_affine_multiplications,
            "estimated_pooling_additions": result.estimated_pooling_additions,
            "estimated_pooling_scalings": result.estimated_pooling_scalings,
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


@dataclass(frozen=True, slots=True)
class AblationForecastPrediction:
    variant: str
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
    pooling_additions: int
    pooling_scalings: int
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


def _admitted_inputs(
    observations: Iterable[ObservedPrefix],
    vocabulary: SequenceVocabulary,
    config: AblationForecastConfig,
    policy: SequencePolicy,
    data: SequenceLimits,
    numeric: NeuralNumericLimits,
    pooling: AblationPoolingLimits,
) -> tuple[EncodedPrefix, ...]:
    encoded_inputs: list[EncodedPrefix] = []
    turns = tokens = source_bytes = affine = pool = 0
    for observation in observations:
        if len(encoded_inputs) >= data.max_conversations:
            raise ValueError("inference collection exceeds max_conversations")
        if type(observation) is not ObservedPrefix:
            raise ValueError("inference requires ObservedPrefix values, not supervised examples")
        observation.validate_limits(data)
        if len(observation.turns) < policy.min_turns:
            raise ValueError("prediction needs configured min_turns complete observations")
        turns += len(observation.turns)
        source_bytes += len(observation.conversation_id.encode("utf-8")) + sum(
            len(turn.id.encode("utf-8")) + len(turn.text.encode("utf-8"))
            for turn in observation.turns
        )
        if turns > data.max_source_turns or source_bytes > data.max_source_bytes:
            raise ValueError("inference collection exceeds aggregate source budgets")
        encoded = encode_observed_prefix(observation, vocabulary, limits=data)
        tokens += encoded.raw_tokens
        work = admit_ablation_turns(
            config.variant,
            config.architecture(vocabulary.size),
            encoded.turns,
            limits=numeric,
            pooling_limits=pooling,
        )
        affine += work.affine_multiplications
        pool += work.pooling_operations
        if tokens > data.max_tokens:
            raise ValueError("inference collection exceeds aggregate token budget")
        if affine > config.max_inference_affine_multiplications:
            raise ValueError("inference collection exceeds max_inference_affine_multiplications")
        if pool > config.max_inference_pooling_operations:
            raise ValueError("inference collection exceeds max_inference_pooling_operations")
        encoded_inputs.append(encoded)
    return tuple(encoded_inputs)


def _dataset_scores(
    data: SequenceForecastDataset, state: AblationForecastState
) -> tuple[float, ...]:
    encoded_inputs = _admitted_inputs(
        data.observations,
        state.vocabulary,
        state.config,
        state.policy,
        state.data_limits,
        state.parameters.limits,
        state.parameters.pooling_limits,
    )
    probabilities = [
        infer_ablation_turns(state.parameters, encoded.turns).probabilities
        for encoded in encoded_inputs
    ]
    return tuple(
        _clip(probabilities[item.conversation_index][item.endpoint]) for item in data.examples
    )


def _prediction(state: AblationForecastState, encoded: EncodedPrefix) -> AblationForecastPrediction:
    result = infer_ablation_turns(state.parameters, encoded.turns)
    probability = _clip(result.probabilities[-1])
    return AblationForecastPrediction(
        state.config.variant,
        result.logits[-1],
        probability,
        probability >= state.threshold,
        state.threshold,
        len(encoded.turns),
        encoded.raw_tokens,
        encoded.retained_tokens,
        encoded.known_tokens,
        encoded.truncated_turns,
        encoded.source_digest,
        state.digest,
        result.work.affine_multiplications,
        result.work.pooling_additions,
        result.work.pooling_scalings,
        result.work.estimated_workspace_bytes,
    )


class AblationEventForecaster:
    """Independently fit a fixed control, requiring separate policy-validation data.

    No optional two-partition reuse, prior trained main parameters, automatic
    retries or optimizer resume. A failed refit preserves the previous candidate.
    """

    def __init__(
        self,
        *,
        config: AblationForecastConfig,
        training_config: NeuralTrainingConfig | None = None,
        policy: SequencePolicy | None = None,
        data_limits: SequenceLimits | None = None,
        numeric_limits: NeuralNumericLimits | None = None,
        pooling_limits: AblationPoolingLimits | None = None,
        training_limits: AblationTrainingLimits | None = None,
    ) -> None:
        self._config = config
        self._training_config = (
            training_config if training_config is not None else NeuralTrainingConfig()
        )
        self._policy = policy if policy is not None else SequencePolicy()
        self._data_limits = data_limits if data_limits is not None else SequenceLimits()
        self._numeric_limits = (
            numeric_limits if numeric_limits is not None else NeuralNumericLimits()
        )
        self._pooling_limits = (
            pooling_limits if pooling_limits is not None else AblationPoolingLimits()
        )
        self._training_limits = (
            training_limits if training_limits is not None else AblationTrainingLimits()
        )
        _settings(self._config, self._policy, self._data_limits, self._numeric_limits)
        for value, kind in (
            (self._training_config, NeuralTrainingConfig),
            (self._pooling_limits, AblationPoolingLimits),
            (self._training_limits, AblationTrainingLimits),
        ):
            if type(value) is not kind:
                raise ValueError("forecaster configuration and limits must be typed contracts")
        self._state: AblationForecastState | None = None

    @property
    def state(self) -> AblationForecastState:
        if self._state is None:
            raise ValueError("ablation event forecaster has not been fitted")
        return self._state

    @property
    def digest(self) -> str:
        return self.state.digest

    @property
    def training_summary(self) -> dict[str, Any]:
        return self.state.summary()

    def save(self, path: str | os.PathLike[str], *, overwrite: bool = False) -> NeuralSaveResult:
        """Publish the distinct inference format, exclusively unless explicitly replaced.

        No optimizer state or executable model object is serialized. The returned
        receipt binds the published bytes and distinguishes cleanup warnings.
        """
        from .neural_ablation_artifact import save_ablation_forecaster

        return save_ablation_forecaster(self, path, overwrite=overwrite)

    @staticmethod
    def load(
        path: str | os.PathLike[str], *, limits: NeuralArtifactLimits | None = None
    ) -> AblationEventForecaster:
        """Restore a closed control model without importing Torch or running model code."""
        from .neural_ablation_artifact import load_ablation_forecaster

        return load_ablation_forecaster(path, limits=limits)

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
        policy_validation: SequenceForecastDataset | Iterable[Conversation],
    ) -> AblationEventForecaster:
        train, validation_data, policy_data = (
            self._data(training),
            self._data(validation),
            self._data(policy_validation),
        )
        if train.group_digests & validation_data.group_digests or policy_data.group_digests & (
            train.group_digests | validation_data.group_digests
        ):
            raise ValueError("all three fitting partitions require distinct groups")
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
        # Policy deployment inputs and both aggregate work quotas are admitted before
        # expensive optimization. Keep these immutable encodings for final scoring.
        policy_inputs = _admitted_inputs(
            policy_data.observations,
            vocabulary,
            self._config,
            self._policy,
            self._data_limits,
            self._numeric_limits,
            self._pooling_limits,
        )
        result = train_ablation_model(
            train,
            validation_data,
            vocabulary,
            self._config.architecture(vocabulary.size),
            variant=self._config.variant,
            config=self._training_config,
            numeric_limits=self._numeric_limits,
            data_limits=self._data_limits,
            pooling_limits=self._pooling_limits,
            training_limits=self._training_limits,
        )
        # A valid result object is not evidence that it belongs to this request.
        # Bind it before policy inference, including future trainer-adapter errors.
        if type(result) is not AblationTrainingResult or (
            result.variant != self._config.variant
            or result.config != self._training_config
            or result.training_limits != self._training_limits
            or result.parameters.limits != self._numeric_limits
            or result.parameters.pooling_limits != self._pooling_limits
            or result.parameters.architecture != self._config.architecture(vocabulary.size)
            or result.vocabulary_digest != vocabulary.digest
            or result.training_partition_digest != train.digest
            or result.validation_partition_digest != validation_data.digest
            or result.training_conversations != len(train.observations)
            or result.validation_conversations != len(validation_data.observations)
            or result.training_prefixes != len(train.examples)
            or result.validation_prefixes != len(validation_data.examples)
        ):
            raise ValueError("training candidate does not match the fitting request")
        probabilities = [
            infer_ablation_turns(result.parameters, encoded.turns).probabilities
            for encoded in policy_inputs
        ]
        scores = tuple(
            _clip(probabilities[item.conversation_index][item.endpoint])
            for item in policy_data.examples
        )
        threshold, accuracy = choose_threshold(_metric_examples(policy_data), scores)
        candidate = AblationForecastState(
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
        )
        self._state = candidate
        return self

    def predict(self, observed: ObservedPrefix) -> AblationForecastPrediction:
        return self.transform((observed,))[0]

    def predict_conversation(self, conversation: Conversation) -> AblationForecastPrediction:
        state = self.state
        return self.predict(
            observed_prefix(conversation, policy=state.policy, limits=state.data_limits)
        )

    def transform(
        self, observations: Iterable[ObservedPrefix]
    ) -> tuple[AblationForecastPrediction, ...]:
        state = self.state
        encoded_inputs = _admitted_inputs(
            observations,
            state.vocabulary,
            state.config,
            state.policy,
            state.data_limits,
            state.parameters.limits,
            state.parameters.pooling_limits,
        )
        return tuple(_prediction(state, encoded) for encoded in encoded_inputs)

    def evaluate(self, heldout: SequenceForecastDataset | Iterable[Conversation]) -> dict[str, Any]:
        state = self.state
        data = self._data(heldout)
        if data.group_digests & (
            state.training_groups | state.validation_groups | state.policy_validation_groups
        ):
            raise ValueError("evaluation groups overlap a fitting partition")
        if not data.examples:
            raise ValueError("evaluation needs eligible forecast prefixes")
        scores = _dataset_scores(data, state)
        return {
            "format": "turnscope.neural-ablation-report.v1",
            "variant": state.config.variant,
            "model_digest": state.digest,
            "partition_digest": data.digest,
            "support": data.audit.to_dict(),
            "metrics": forecast_metrics(_metric_examples(data), scores, state.threshold),
            "policy_reuses_model_validation": False,
            "probability_calibration_claimed": False,
            "whole_repository_parity_claimed": False,
        }
