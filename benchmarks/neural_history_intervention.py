"""Prefix-local order diagnostic for a frozen main forecaster; no fitting.

Private input/model/vocabulary stay in process. Aggregate digests are integrity
bindings, not anonymization. This OOD intervention is not a causal estimate or
the independently trained order-erased comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from turnscope.forecast_metrics import forecast_metrics
from turnscope.models import Conversation
from turnscope.neural_forecast import (
    HierarchicalEventForecaster,
    NeuralForecastState,
    _clip,
    _metric_examples,
)
from turnscope.neural_forecast_data import SequenceForecastDataset
from turnscope.neural_forecast_math import (
    NeuralArchitecture,
    _gru_direction,
    _integer,
    _numpy,
    _sigmoid,
    admit_encoded_turns,
)
from turnscope.neural_token_data import EncodedPrefix, encode_observed_prefix

PERMUTATION_FORMAT = "turnscope.prefix-history-permutation.v1"
PERMUTATION_SEED = "turnscope-history-order/2026-09-12"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def permutation_protocol() -> dict[str, str]:
    return {
        "format": PERMUTATION_FORMAT,
        "seed": PERMUTATION_SEED,
        "ordering": (
            "ascending SHA256(canonical protocol/conversation/endpoint/historical ID), "
            "then original position"
        ),
        "current_turn": "kept last",
        "score": "only final logit attached to original eligible endpoint",
    }


def prefix_history_order(conversation_id: str, prefix_turn_ids: tuple[str, ...]) -> tuple[int, ...]:
    """Order only IDs in this prefix; no labels, model seed or future ID inputs."""
    if type(conversation_id) is not str or not 1 <= len(conversation_id.encode("utf-8")) <= 1024:
        raise ValueError("conversation ID must be bounded nonempty text")
    if type(prefix_turn_ids) is not tuple or not 1 <= len(prefix_turn_ids) <= 256:
        raise ValueError("prefix turn IDs must be a nonempty bounded tuple")
    names = set()
    for name in prefix_turn_ids:
        if type(name) is not str or not 1 <= len(name.encode("utf-8")) <= 1024 or name in names:
            raise ValueError("prefix turn IDs must be unique bounded nonempty text")
        names.add(name)
    endpoint = prefix_turn_ids[-1]
    ranked = []
    for position, identifier in enumerate(prefix_turn_ids[:-1]):
        digest = hashlib.sha256(
            _canonical(
                {
                    "format": PERMUTATION_FORMAT,
                    "seed": PERMUTATION_SEED,
                    "conversation_id": conversation_id,
                    "endpoint_turn_id": endpoint,
                    "historical_turn_id": identifier,
                }
            )
        ).digest()
        ranked.append((digest, position))
    return (*[position for _, position in sorted(ranked)], len(prefix_turn_ids) - 1)


@dataclass(frozen=True, slots=True)
class HistoryInterventionLimits:
    """Bound materialized order indices independently of numerical array quotas."""

    max_permutation_positions: int = 2_000_000

    def __post_init__(self) -> None:
        _integer(self.max_permutation_positions, "max_permutation_positions", 1, 8_000_000)


@dataclass(frozen=True, slots=True)
class _ConversationWork:
    encoded: EncodedPrefix
    endpoints: tuple[int, ...]
    orders: tuple[tuple[int, ...], ...]
    affine_multiplications: int
    estimated_workspace_bytes: int


def _word_work(architecture: NeuralArchitecture, positions: int) -> int:
    return sum(
        positions
        * 6
        * architecture.word_hidden
        * (
            (architecture.embedding_dim if layer == 0 else 2 * architecture.word_hidden)
            + architecture.word_hidden
        )
        for layer in range(architecture.word_layers)
    )


def _turn_work(architecture: NeuralArchitecture, steps: int) -> int:
    return sum(
        steps
        * 3
        * architecture.turn_hidden
        * (
            (2 * architecture.word_hidden if layer == 0 else architecture.turn_hidden)
            + architecture.turn_hidden
        )
        for layer in range(architecture.turn_layers)
    )


def _admit(
    state: NeuralForecastState,
    data: SequenceForecastDataset,
    limits: HistoryInterventionLimits,
) -> tuple[tuple[_ConversationWork, ...], dict[str, Any]]:
    architecture = state.parameters.architecture
    endpoints: dict[int, list[int]] = {}
    for example in data.examples:
        endpoints.setdefault(example.conversation_index, []).append(example.endpoint)
    # Admit the entire logical order index inventory before constructing any order
    # tuples or numerical arrays. Empty/omitted cases cannot turn into scored zeros.
    positions = sum(example.endpoint + 1 for example in data.examples)
    if positions > limits.max_permutation_positions:
        raise ValueError("intervention exceeds max_permutation_positions")
    admitted = []
    work_total = word_total = turn_total = head_total = maximum_workspace = 0
    changed_indices = changed_content = eligible_history = 0
    original_hasher = hashlib.sha256()
    transformed_hasher = hashlib.sha256()
    ordering_hasher = hashlib.sha256()
    raw_tokens = retained_tokens = known_tokens = truncated_turns = unique_turns = 0
    for index, observation in enumerate(data.observations):
        encoded = encode_observed_prefix(observation, state.vocabulary, limits=state.data_limits)
        base = admit_encoded_turns(architecture, encoded.turns, limits=state.parameters.limits)
        selected = tuple(endpoints[index])
        orders = tuple(
            prefix_history_order(
                observation.conversation_id,
                tuple(turn.id for turn in observation.turns[: endpoint + 1]),
            )
            for endpoint in selected
        )
        steps = sum(endpoint + 1 for endpoint in selected)
        word = _word_work(architecture, base.token_positions)
        turn = _turn_work(architecture, steps)
        head = len(selected) * architecture.head_hidden * (architecture.turn_hidden + 1)
        total = word + turn + head
        # Numeric quota applies to this conversation's full diagnostic, not only to
        # one cheap prefix. The collection additionally uses the original global cap.
        if total > state.parameters.limits.max_affine_multiplications:
            raise ValueError("intervention conversation exceeds max_affine_multiplications")
        work_total += total
        if work_total > state.config.max_inference_affine_multiplications:
            raise ValueError("intervention exceeds max_inference_affine_multiplications")
        cache_bytes = len(encoded.turns) * 2 * architecture.word_hidden * 8
        workspace = base.estimated_workspace_bytes + 2 * cache_bytes
        if workspace > state.parameters.limits.max_workspace_bytes:
            raise ValueError("intervention exceeds max_workspace_bytes")
        maximum_workspace = max(maximum_workspace, workspace)
        word_total += word
        turn_total += turn
        head_total += head
        raw_tokens += encoded.raw_tokens
        retained_tokens += encoded.retained_tokens
        known_tokens += encoded.known_tokens
        truncated_turns += encoded.truncated_turns
        unique_turns += len(encoded.turns)
        for endpoint, order in zip(selected, orders, strict=True):
            original = encoded.turns[: endpoint + 1]
            transformed = tuple(encoded.turns[position] for position in order)
            changed_indices += order != tuple(range(endpoint + 1))
            changed_content += transformed != original
            eligible_history += endpoint >= 2
            # One canonical newline-delimited item per existing canonical endpoint.
            # Source/model/vocabulary identity is bound separately in the report.
            original_hasher.update(_canonical(original) + b"\n")
            transformed_hasher.update(_canonical(transformed) + b"\n")
            ordering_hasher.update(_canonical(order) + b"\n")
        admitted.append(_ConversationWork(encoded, selected, orders, total, workspace))
    return tuple(admitted), {
        "eligible_prefixes": len(data.examples),
        "prefixes_with_two_historical_turns": eligible_history,
        "index_order_changed_prefixes": changed_indices,
        "encoded_content_changed_prefixes": changed_content,
        "permutation_positions": positions,
        "unique_completed_turns": unique_turns,
        "word_affine_multiplications": word_total,
        "turn_affine_multiplications": turn_total,
        "head_affine_multiplications": head_total,
        "affine_multiplications": work_total,
        "maximum_estimated_workspace_bytes": maximum_workspace,
        "raw_tokens": raw_tokens,
        "retained_tokens": retained_tokens,
        "known_tokens": known_tokens,
        "truncated_turns": truncated_turns,
        "original_prefix_inventory_sha256": original_hasher.hexdigest(),
        "transformed_prefix_inventory_sha256": transformed_hasher.hexdigest(),
        "ordering_inventory_sha256": ordering_hasher.hexdigest(),
    }


def _cached_scores(
    np: Any, arrays: dict[str, Any], architecture: NeuralArchitecture, record: _ConversationWork
) -> tuple[float, ...]:
    """Private per-conversation cache; never accepts caller-supplied word vectors."""
    vectors = []
    for turn in record.encoded.turns:
        sequence = arrays["embedding.weight"][list(turn)]
        for layer in range(architecture.word_layers):
            forward, final_forward = _gru_direction(
                np, sequence, arrays, "word", layer, architecture.word_hidden, False
            )
            backward, final_backward = _gru_direction(
                np, sequence, arrays, "word", layer, architecture.word_hidden, True
            )
            sequence = np.concatenate((forward, backward), axis=1)
        vectors.append(np.concatenate((final_forward, final_backward)))
    cached = np.stack(vectors)
    scores: list[float] = []
    for order in record.orders:
        states = cached[list(order)]
        for layer in range(architecture.turn_layers):
            states, _ = _gru_direction(
                np, states, arrays, "turn", layer, architecture.turn_hidden, False
            )
        hidden = np.tanh(arrays["head.weight"] @ states[-1] + arrays["head.bias"])
        logit = (arrays["output.weight"] @ hidden + arrays["output.bias"])[0]
        if not math.isfinite(float(logit)):
            raise ValueError("intervention produced a nonfinite logit")
        scores.append(_clip(float(_sigmoid(np, np.asarray(logit)))))
    return tuple(scores)


def evaluate_history_intervention(
    model: HierarchicalEventForecaster,
    heldout: SequenceForecastDataset | Iterable[Conversation],
    *,
    limits: HistoryInterventionLimits | None = None,
) -> dict[str, Any]:
    """Score all original eligible handles after the fixed prefix-local permutation.

    The learned main threshold is unchanged. No data are fitted, recalibrated,
    dropped or emitted; all default primary metric denominators are preserved.
    """
    if type(model) is not HierarchicalEventForecaster:
        raise ValueError("intervention requires a fitted main HierarchicalEventForecaster")
    selected = limits if limits is not None else HistoryInterventionLimits()
    if type(selected) is not HistoryInterventionLimits:
        raise ValueError("limits must be HistoryInterventionLimits")
    state = model.state
    data = model._data(heldout)
    if data.group_digests & (
        state.training_groups | state.validation_groups | state.policy_validation_groups
    ):
        raise ValueError("intervention groups overlap a fitting partition")
    if not data.examples:
        raise ValueError("intervention needs eligible forecast prefixes")
    admitted, audit = _admit(state, data, selected)
    np = _numpy()
    arrays = {name: value.astype(np.float64) for name, value in state.parameters.arrays().items()}
    scores: list[float] = []
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for record in admitted:
            scores.extend(_cached_scores(np, arrays, state.parameters.architecture, record))
    # Dataset examples are canonical conversation/endpoint order; every cached
    # output represents only the final state of its own original eligible prefix.
    if len(scores) != len(data.examples):
        raise ValueError("intervention output inventory differs from eligible endpoints")
    return {
        "format": "turnscope.neural-history-intervention-report.v1",
        "model_digest": state.digest,
        "parameter_digest": state.parameters.digest,
        "vocabulary_digest": state.vocabulary.digest,
        "partition_digest": data.digest,
        "protocol": permutation_protocol(),
        "protocol_sha256": hashlib.sha256(_canonical(permutation_protocol())).hexdigest(),
        "limits": asdict(selected),
        "support": data.audit.to_dict(),
        "audit": audit,
        "prediction_inventory_sha256": hashlib.sha256(_canonical(scores)).hexdigest(),
        "metrics": forecast_metrics(_metric_examples(data), scores, state.threshold),
        "threshold_refitted": False,
        "weights_refitted": False,
        "out_of_distribution_diagnostic": True,
        "causal_effect_claimed": False,
        "probability_calibration_claimed": False,
        "whole_repository_parity_claimed": False,
    }
