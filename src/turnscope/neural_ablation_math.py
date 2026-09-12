"""Closed numerical controls for the fixed one-layer, 64-wide neural experiment.

These are alternate forward functions, never interpretations of main-model
artifacts. NumPy is optional and lazy; inference does not import Torch. Work
estimates bound declared owned arrays/arithmetic, not allocator RSS or FLOPs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .neural_forecast_math import (
    FrozenNeuralTensor,
    NeuralArchitecture,
    NeuralNumericLimits,
    _gru_direction,
    _integer,
    _numpy,
    _sigmoid,
    parameter_shapes,
)

ABLATION_VARIANTS = ("current-turn.v1", "mean-word.v1", "order-erased.v1")
_NUMERICAL_VERSIONS = MappingProxyType(
    {
        "current-turn.v1": "current-turn.reset-after-zero-rzn.v1",
        "mean-word.v1": "mean-word-eos-duplicate.reset-after-rzn.v1",
        "order-erased.v1": "mean-word-eos.mean-prefix.v1",
    }
)


def ablation_numerical_version(variant: str) -> str:
    if type(variant) is not str or variant not in _NUMERICAL_VERSIONS:
        raise ValueError("variant must name one of the three closed ablation versions")
    return _NUMERICAL_VERSIONS[variant]


def ablation_parameter_shapes(
    variant: str, architecture: NeuralArchitecture
) -> dict[str, tuple[int, ...]]:
    """Retain active main-inventory names, rejecting unversioned architectures."""
    ablation_numerical_version(variant)
    if type(architecture) is not NeuralArchitecture or (
        architecture.embedding_dim,
        architecture.word_hidden,
        architecture.turn_hidden,
        architecture.word_layers,
        architecture.turn_layers,
    ) != (64, 64, 64, 1, 1):
        raise ValueError("ablations require the fixed 64-wide, one-layer reference architecture")
    return {
        name: shape
        for name, shape in parameter_shapes(architecture).items()
        if (
            (variant == "current-turn.v1" and name != "turn.weight_hh_l0")
            or (variant == "mean-word.v1" and not name.startswith("word."))
            or (variant == "order-erased.v1" and not name.startswith(("word.", "turn.")))
        )
    }


@dataclass(frozen=True, slots=True)
class AblationPoolingLimits:
    """Separate addition/scaling quota; not disguised as affine multiplications."""

    max_operations: int = 100_000_000

    def __post_init__(self) -> None:
        _integer(self.max_operations, "max_pooling_operations", 1, 1_000_000_000)

    def to_dict(self) -> dict[str, int]:
        return {"max_operations": self.max_operations}


@dataclass(frozen=True, slots=True)
class FrozenAblationParameters:
    variant: str
    architecture: NeuralArchitecture
    tensors: tuple[FrozenNeuralTensor, ...]
    limits: NeuralNumericLimits = field(default_factory=NeuralNumericLimits)
    pooling_limits: AblationPoolingLimits = field(default_factory=AblationPoolingLimits)

    def __post_init__(self) -> None:
        expected = ablation_parameter_shapes(self.variant, self.architecture)
        if type(self.limits) is not NeuralNumericLimits:
            raise ValueError("limits must be NeuralNumericLimits")
        if type(self.pooling_limits) is not AblationPoolingLimits:
            raise ValueError("pooling_limits must be AblationPoolingLimits")
        if self.parameter_count > self.limits.max_parameters:
            raise ValueError("ablation exceeds max_parameters")
        if type(self.tensors) is not tuple or len(self.tensors) != len(expected):
            raise ValueError("tensor inventory must match the ablation")
        for tensor, (name, shape) in zip(self.tensors, expected.items(), strict=True):
            if type(tensor) is not FrozenNeuralTensor or (tensor.name, tensor.shape) != (
                name,
                shape,
            ):
                raise ValueError("tensor order, name or shape differs from the closed inventory")
        np = _numpy()
        for tensor in self.tensors:
            values = np.frombuffer(tensor.data, dtype="<f4")
            if (
                not np.isfinite(values).all()
                or (np.abs(values) > self.limits.max_parameter_magnitude).any()
            ):
                raise ValueError(f"{tensor.name} must contain bounded finite parameters")
        # PAD is a stored constant in this experimental format, not trainable input.
        if any(self.tensors[0].data[: 64 * 4]):
            raise ValueError("the PAD embedding must have canonical positive-zero bytes")

    @property
    def parameter_count(self) -> int:
        return sum(
            math.prod(shape)
            for shape in ablation_parameter_shapes(self.variant, self.architecture).values()
        )

    @classmethod
    def from_arrays(
        cls,
        variant: str,
        architecture: NeuralArchitecture,
        arrays: Mapping[str, Any],
        *,
        limits: NeuralNumericLimits | None = None,
        pooling_limits: AblationPoolingLimits | None = None,
    ) -> FrozenAblationParameters:
        selected = limits if limits is not None else NeuralNumericLimits()
        selected_pooling = pooling_limits if pooling_limits is not None else AblationPoolingLimits()
        if type(selected) is not NeuralNumericLimits:
            raise ValueError("limits must be NeuralNumericLimits")
        if type(selected_pooling) is not AblationPoolingLimits:
            raise ValueError("pooling_limits must be AblationPoolingLimits")
        expected = ablation_parameter_shapes(variant, architecture)
        if sum(math.prod(shape) for shape in expected.values()) > selected.max_parameters:
            raise ValueError("ablation exceeds max_parameters")
        if not isinstance(arrays, Mapping) or len(arrays) != len(expected):
            raise ValueError("array names must match the closed tensor inventory")
        # A caller-controlled keys() view need not describe its actual iterator.
        # Check at most the expected count plus one, before asking for any values.
        names: set[str] = set()
        for name in arrays:
            if type(name) is not str or name not in expected or name in names:
                raise ValueError("array names must match the closed tensor inventory")
            names.add(name)
        if names != set(expected):
            raise ValueError("array names must match the closed tensor inventory")
        np = _numpy()
        admitted = []
        for name, shape in expected.items():
            value = arrays[name]
            _validate_array(np, name, shape, value)
            admitted.append((name, shape, value))
        tensors = []
        # All structure is admitted before copying. Each mapping value is fetched
        # exactly once; recheck the same array after any later mapping callbacks.
        # Concurrent unsynchronized mutation of input arrays is not supported.
        for name, shape, value in admitted:
            _validate_array(np, name, shape, value)
            tensors.append(FrozenNeuralTensor(name, shape, value.tobytes(order="C")))
        return cls(variant, architecture, tuple(tensors), selected, selected_pooling)

    def arrays(self) -> Mapping[str, Any]:
        np = _numpy()
        return MappingProxyType(
            {
                tensor.name: np.frombuffer(tensor.data, dtype="<f4").reshape(tensor.shape)
                for tensor in self.tensors
            }
        )

    @property
    def digest(self) -> str:
        body = {
            "variant": self.variant,
            "numerical_version": ablation_numerical_version(self.variant),
            "architecture": self.architecture.to_dict(),
            "tensors": [
                {
                    "name": tensor.name,
                    "shape": tensor.shape,
                    "sha256": hashlib.sha256(tensor.data).hexdigest(),
                }
                for tensor in self.tensors
            ],
        }
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()


def _validate_array(np: Any, name: str, shape: tuple[int, ...], value: Any) -> None:
    if (
        type(value) is not np.ndarray
        or value.dtype.str != "<f4"
        or value.shape != shape
        or not value.flags.c_contiguous
    ):
        raise ValueError(f"{name} must be a C-contiguous little-endian float32 array")


@dataclass(frozen=True, slots=True)
class AblationInferenceWork:
    turns: int
    token_positions: int
    affine_multiplications: int
    estimated_workspace_bytes: int
    pooling_additions: int
    pooling_scalings: int

    @property
    def pooling_operations(self) -> int:
        return self.pooling_additions + self.pooling_scalings


@dataclass(frozen=True, slots=True)
class AblationInference:
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    turn_vectors: tuple[tuple[float, ...], ...]
    turn_states: tuple[tuple[float, ...], ...]
    work: AblationInferenceWork


def admit_ablation_turns(
    variant: str,
    architecture: NeuralArchitecture,
    turns: tuple[tuple[int, ...], ...],
    *,
    limits: NeuralNumericLimits | None = None,
    pooling_limits: AblationPoolingLimits | None = None,
) -> AblationInferenceWork:
    """Validate shapes, EOS, active storage and separate work caps before NumPy."""
    selected = limits if limits is not None else NeuralNumericLimits()
    pooling = pooling_limits if pooling_limits is not None else AblationPoolingLimits()
    if type(selected) is not NeuralNumericLimits or type(pooling) is not AblationPoolingLimits:
        raise ValueError("admission requires numeric and pooling limits")
    shapes = ablation_parameter_shapes(variant, architecture)
    count = sum(math.prod(shape) for shape in shapes.values())
    if count > selected.max_parameters:
        raise ValueError("ablation exceeds max_parameters")
    if type(turns) is not tuple or not 1 <= len(turns) <= selected.max_observed_turns:
        raise ValueError("turns must be a nonempty bounded tuple")
    positions = 0
    for turn in turns:
        if type(turn) is not tuple or not 1 <= len(turn) <= selected.max_turn_tokens:
            raise ValueError("encoded turn exceeds max_turn_tokens or is not a nonempty tuple")
        positions += len(turn)
        if positions > selected.max_token_positions:
            raise ValueError("encoded input exceeds max_token_positions")
        for index, token in enumerate(turn):
            _integer(token, "token ID", 1, architecture.vocabulary_size - 1)
            if (token == 2) != (index == len(turn) - 1):
                raise ValueError("each turn must contain exactly one final EOS and no PAD")
    units = len(turns)
    affine = units * 32 * (64 + 1)
    additions = scalings = 0
    if variant == "current-turn.v1":
        affine += positions * 6 * 64 * (64 + 64) + units * 3 * 64 * 128
    else:
        # Sum each actual turn from its first vector, then divide by its own length.
        additions = (positions - units) * 64
        scalings = units * 64
        if variant == "mean-word.v1":
            affine += units * 3 * 64 * (128 + 64)
        else:
            additions += (units - 1) * 64
            scalings += units * 64
    if affine > selected.max_affine_multiplications:
        raise ValueError("inference exceeds max_affine_multiplications")
    if additions + scalings > pooling.max_operations:
        raise ValueError("inference exceeds max_pooling_operations")
    # Owned float32 storage plus float64 conversion/temporaries, encoded embeddings,
    # all returned vectors/states and gate work. Python/BLAS caches are excluded.
    word_width = 12 * 64 if variant == "current-turn.v1" else 0
    workspace = 16 * count + 8 * (
        positions * (64 + word_width) + units * (8 * 64 + 4 * 64) + 24 * (64 + 64)
    )
    if workspace > selected.max_workspace_bytes:
        raise ValueError("inference exceeds max_workspace_bytes")
    return AblationInferenceWork(units, positions, affine, workspace, additions, scalings)


def _zero_state_gate(np: Any, vectors: Any, arrays: Mapping[str, Any]) -> Any:
    incoming = vectors @ arrays["turn.weight_ih_l0"].T + arrays["turn.bias_ih_l0"]
    ir, iz, inn = np.split(incoming, 3, axis=1)
    hr, hz, hn = np.split(arrays["turn.bias_hh_l0"], 3)
    reset = _sigmoid(np, ir + hr)
    update = _sigmoid(np, iz + hz)
    return (1 - update) * np.tanh(inn + reset * hn)


def infer_ablation_turns(
    parameters: FrozenAblationParameters,
    turns: tuple[tuple[int, ...], ...],
    *,
    limits: NeuralNumericLimits | None = None,
    pooling_limits: AblationPoolingLimits | None = None,
) -> AblationInference:
    """Return every completed-turn output under the explicit alternate semantics."""
    if type(parameters) is not FrozenAblationParameters:
        raise ValueError("inference requires FrozenAblationParameters")
    selected = limits if limits is not None else parameters.limits
    pooling = pooling_limits if pooling_limits is not None else parameters.pooling_limits
    work = admit_ablation_turns(
        parameters.variant, parameters.architecture, turns, limits=selected, pooling_limits=pooling
    )
    np = _numpy()
    if selected.max_parameter_magnitude < parameters.limits.max_parameter_magnitude and any(
        (np.abs(value) > selected.max_parameter_magnitude).any()
        for value in parameters.arrays().values()
    ):
        raise ValueError("model exceeds max_parameter_magnitude")
    arrays = {name: value.astype(np.float64) for name, value in parameters.arrays().items()}
    vectors = []
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for turn in turns:
            sequence = arrays["embedding.weight"][list(turn)]
            if parameters.variant == "current-turn.v1":
                _, forward = _gru_direction(np, sequence, arrays, "word", 0, 64, False)
                _, backward = _gru_direction(np, sequence, arrays, "word", 0, 64, True)
                vector = np.concatenate((forward, backward))
            else:
                # Explicit order matches the separately reported logical additions.
                vector = sequence[0].copy()
                for row in sequence[1:]:
                    vector += row
                vector /= len(sequence)
                if parameters.variant == "mean-word.v1":
                    vector = np.concatenate((vector, vector))
            vectors.append(vector)
        stacked = np.stack(vectors)
        if parameters.variant == "current-turn.v1":
            contexts = _zero_state_gate(np, stacked, arrays)
        elif parameters.variant == "mean-word.v1":
            contexts, _ = _gru_direction(np, stacked, arrays, "turn", 0, 64, False)
        else:
            contexts = np.empty_like(stacked)
            total = stacked[0].copy()
            contexts[0] = total
            # The first /1 scaling is explicit, as charged in the work inventory.
            contexts[0] /= 1
            for index in range(1, len(stacked)):
                total += stacked[index]
                contexts[index] = total / (index + 1)
        hidden = np.tanh(contexts @ arrays["head.weight"].T + arrays["head.bias"])
        logits = (hidden @ arrays["output.weight"].T + arrays["output.bias"]).ravel()
        if not np.isfinite(logits).all():
            raise ValueError("ablation inference produced nonfinite logits")
        probabilities = _sigmoid(np, logits)
    return AblationInference(
        tuple(float(value) for value in logits),
        tuple(float(value) for value in probabilities),
        tuple(tuple(float(value) for value in row) for row in stacked),
        tuple(tuple(float(value) for value in row) for row in contexts),
        work,
    )
