"""Bounded, Torch-free inference for an explicitly versioned hierarchical GRU.

Importing this module does not import NumPy. Parameters are float32 immutable
bytes; forward arithmetic is float64. This is a numerical primitive, not a
trained forecasting model, calibrated policy, or checkpoint loader.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

NUMERICAL_VERSION = "turnscope.hierarchical-gru.reset-after-rzn.concat-tanh.v1"


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _numpy() -> Any:
    try:
        return importlib.import_module("numpy")
    except ImportError as error:
        raise ImportError("Neural inference requires NumPy; install turnscope[neural]") from error


@dataclass(frozen=True, slots=True)
class NeuralArchitecture:
    """Tensor shapes; vocabulary_size includes PAD=0, UNK=1 and EOS=2."""

    vocabulary_size: int
    embedding_dim: int = 64
    word_hidden: int = 64
    turn_hidden: int = 64
    word_layers: int = 1
    turn_layers: int = 1

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("vocabulary_size", 3, 100_003),
            ("embedding_dim", 1, 256),
            ("word_hidden", 1, 256),
            ("turn_hidden", 2, 256),
            ("word_layers", 1, 2),
            ("turn_layers", 1, 2),
        ):
            _integer(getattr(self, name), name, minimum, maximum)

    @property
    def head_hidden(self) -> int:
        return self.turn_hidden // 2

    @property
    def parameter_count(self) -> int:
        return sum(math.prod(shape) for shape in parameter_shapes(self).values())

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NeuralNumericLimits:
    """Explicit admission quotas, not guarantees about native allocator RSS."""

    max_parameters: int = 4_000_000
    max_observed_turns: int = 64
    max_turn_tokens: int = 129
    max_token_positions: int = 8192
    max_affine_multiplications: int = 1_000_000_000
    max_workspace_bytes: int = 256 * 1024 * 1024
    max_parameter_magnitude: float = 1_000_000.0

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("max_parameters", 1, 16_000_000),
            ("max_observed_turns", 1, 256),
            ("max_turn_tokens", 1, 1025),
            ("max_token_positions", 1, 262_400),
            ("max_affine_multiplications", 1, 10_000_000_000),
            ("max_workspace_bytes", 1, 1024 * 1024 * 1024),
        ):
            _integer(getattr(self, name), name, minimum, maximum)
        value = self.max_parameter_magnitude
        if type(value) not in (int, float) or not 0 < value <= 1e6 or not math.isfinite(value):
            raise ValueError("max_parameter_magnitude must be finite in (0, 1000000]")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def parameter_shapes(architecture: NeuralArchitecture) -> dict[str, tuple[int, ...]]:
    """Closed logical inventory, compatible with explicitly named Torch layers."""
    if type(architecture) is not NeuralArchitecture:
        raise ValueError("architecture must be NeuralArchitecture")
    shapes: dict[str, tuple[int, ...]] = {
        "embedding.weight": (architecture.vocabulary_size, architecture.embedding_dim)
    }
    for level, layers, hidden, first_width, directions in (
        ("word", architecture.word_layers, architecture.word_hidden, architecture.embedding_dim, 2),
        (
            "turn",
            architecture.turn_layers,
            architecture.turn_hidden,
            2 * architecture.word_hidden,
            1,
        ),
    ):
        for layer in range(layers):
            width = first_width if layer == 0 else directions * hidden
            for direction in range(directions):
                suffix = f"_l{layer}" + ("_reverse" if direction else "")
                shapes[f"{level}.weight_ih{suffix}"] = (3 * hidden, width)
                shapes[f"{level}.weight_hh{suffix}"] = (3 * hidden, hidden)
                shapes[f"{level}.bias_ih{suffix}"] = (3 * hidden,)
                shapes[f"{level}.bias_hh{suffix}"] = (3 * hidden,)
    shapes.update(
        {
            "head.weight": (architecture.head_hidden, architecture.turn_hidden),
            "head.bias": (architecture.head_hidden,),
            "output.weight": (1, architecture.head_hidden),
            "output.bias": (1,),
        }
    )
    return shapes


@dataclass(frozen=True, slots=True)
class FrozenNeuralTensor:
    """A closed parameter member; the immutable byte owner cannot be made writable."""

    name: str
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        if type(self.name) is not str or not 1 <= len(self.name) <= 64:
            raise ValueError("tensor name must be bounded text")
        if type(self.shape) is not tuple or not 1 <= len(self.shape) <= 2:
            raise ValueError("tensor shape must have one or two dimensions")
        for dimension in self.shape:
            _integer(dimension, "tensor dimension", 1, 100_003)
        if math.prod(self.shape) > 16_000_000:
            raise ValueError("tensor exceeds hard parameter limit")
        if type(self.data) is not bytes or len(self.data) != 4 * math.prod(self.shape):
            raise ValueError("tensor bytes must match its float32 shape")


@dataclass(frozen=True, slots=True)
class FrozenNeuralParameters:
    architecture: NeuralArchitecture
    tensors: tuple[FrozenNeuralTensor, ...]
    limits: NeuralNumericLimits = NeuralNumericLimits()

    def __post_init__(self) -> None:
        if type(self.limits) is not NeuralNumericLimits:
            raise ValueError("limits must be NeuralNumericLimits")
        expected = parameter_shapes(self.architecture)
        if self.architecture.parameter_count > self.limits.max_parameters:
            raise ValueError("model exceeds max_parameters")
        if type(self.tensors) is not tuple or len(self.tensors) != len(expected):
            raise ValueError("tensor inventory must match the architecture")
        # Shape/byte and aggregate admission precedes any numerical array work.
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

    @classmethod
    def from_arrays(
        cls,
        architecture: NeuralArchitecture,
        arrays: Mapping[str, Any],
        *,
        limits: NeuralNumericLimits | None = None,
    ) -> FrozenNeuralParameters:
        selected = limits if limits is not None else NeuralNumericLimits()
        if type(selected) is not NeuralNumericLimits:
            raise ValueError("limits must be NeuralNumericLimits")
        expected = parameter_shapes(architecture)
        if architecture.parameter_count > selected.max_parameters:
            raise ValueError("model exceeds max_parameters")
        if not isinstance(arrays, Mapping) or arrays.keys() != expected.keys():
            raise ValueError("array names must match the closed tensor inventory")
        np = _numpy()
        tensors = []
        for name, shape in expected.items():
            array = arrays[name]
            if (
                type(array) is not np.ndarray
                or array.dtype.str != "<f4"
                or array.shape != shape
                or not array.flags.c_contiguous
            ):
                raise ValueError(
                    f"{name} must be a C-contiguous little-endian float32 array {shape}"
                )
            # Snapshot the object actually validated, never query a mutable/custom
            # Mapping again and accidentally serialize a different object.
            tensors.append(FrozenNeuralTensor(name, shape, array.tobytes(order="C")))
        return cls(architecture, tuple(tensors), selected)

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
            "numerical_version": NUMERICAL_VERSION,
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


@dataclass(frozen=True, slots=True)
class NeuralInferenceWork:
    turns: int
    token_positions: int
    affine_multiplications: int
    estimated_workspace_bytes: int


@dataclass(frozen=True, slots=True)
class NeuralInference:
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    turn_vectors: tuple[tuple[float, ...], ...]
    turn_states: tuple[tuple[float, ...], ...]
    work: NeuralInferenceWork


def admit_encoded_turns(
    architecture: NeuralArchitecture,
    turns: tuple[tuple[int, ...], ...],
    *,
    limits: NeuralNumericLimits | None = None,
) -> NeuralInferenceWork:
    """Array-free shape/input/work admission, also used before training allocation."""
    limits = limits if limits is not None else NeuralNumericLimits()
    if type(architecture) is not NeuralArchitecture or type(limits) is not NeuralNumericLimits:
        raise ValueError("admission requires architecture and numeric limits")
    if architecture.parameter_count > limits.max_parameters:
        raise ValueError("model exceeds max_parameters")
    if type(turns) is not tuple or not 1 <= len(turns) <= limits.max_observed_turns:
        raise ValueError("turns must be a nonempty bounded tuple")
    positions = 0
    for turn in turns:
        if type(turn) is not tuple or not 1 <= len(turn) <= limits.max_turn_tokens:
            raise ValueError("encoded turn exceeds max_turn_tokens or is not a nonempty tuple")
        positions += len(turn)
        if positions > limits.max_token_positions:
            raise ValueError("encoded input exceeds max_token_positions")
        for index, token in enumerate(turn):
            _integer(token, "token ID", 1, architecture.vocabulary_size - 1)
            if (token == 2) != (index == len(turn) - 1):
                raise ValueError("each turn must contain exactly one final EOS and no PAD")
    # Count affine scalar multiplications exactly (not elementwise gates or FLOPs).
    multiplications = 0
    for layer in range(architecture.word_layers):
        width = architecture.embedding_dim if layer == 0 else 2 * architecture.word_hidden
        multiplications += (
            positions * 2 * 3 * architecture.word_hidden * (width + architecture.word_hidden)
        )
    for layer in range(architecture.turn_layers):
        width = 2 * architecture.word_hidden if layer == 0 else architecture.turn_hidden
        multiplications += (
            len(turns) * 3 * architecture.turn_hidden * (width + architecture.turn_hidden)
        )
    multiplications += len(turns) * architecture.head_hidden * (architecture.turn_hidden + 1)
    if multiplications > limits.max_affine_multiplications:
        raise ValueError("inference exceeds max_affine_multiplications")
    # Conservative explicit owned-array estimate. Python objects, BLAS and allocator
    # caches are not an RSS contract. Whole-model float64 conversion is included.
    workspace = 16 * architecture.parameter_count + 8 * (
        positions * (architecture.embedding_dim + 12 * architecture.word_hidden)
        + len(turns) * (8 * architecture.turn_hidden + 4 * architecture.word_hidden)
        + 24 * (architecture.word_hidden + architecture.turn_hidden)
    )
    if workspace > limits.max_workspace_bytes:
        raise ValueError("inference exceeds max_workspace_bytes")
    return NeuralInferenceWork(len(turns), positions, multiplications, workspace)


def _admit(
    parameters: FrozenNeuralParameters,
    turns: tuple[tuple[int, ...], ...],
    limits: NeuralNumericLimits,
) -> NeuralInferenceWork:
    if type(parameters) is not FrozenNeuralParameters or type(limits) is not NeuralNumericLimits:
        raise ValueError("inference requires frozen parameters and numeric limits")
    work = admit_encoded_turns(parameters.architecture, turns, limits=limits)
    if limits.max_parameter_magnitude < parameters.limits.max_parameter_magnitude:
        # Scan only after cheap shape, input, work and allocation admission.
        # The stored limit is a bound, not evidence that every value reaches it.
        np = _numpy()
        if any(
            (np.abs(value) > limits.max_parameter_magnitude).any()
            for value in parameters.arrays().values()
        ):
            raise ValueError("model exceeds max_parameter_magnitude")
    return work


def _sigmoid(np: Any, values: Any) -> Any:
    exponent = np.exp(-np.abs(values))
    return np.where(values >= 0, 1 / (1 + exponent), exponent / (1 + exponent))


def _gru_step(np: Any, x: Any, previous: Any, weights: tuple[Any, Any, Any, Any]) -> Any:
    """Internal admitted arrays only: reset/update/new and reset-after affine."""
    weight_ih, weight_hh, bias_ih, bias_hh = weights
    incoming = weight_ih @ x + bias_ih
    recurrent = weight_hh @ previous + bias_hh
    ir, iz, inn = np.split(incoming, 3)
    hr, hz, hn = np.split(recurrent, 3)
    reset = _sigmoid(np, ir + hr)
    update = _sigmoid(np, iz + hz)
    new = np.tanh(inn + reset * hn)
    return (1 - update) * new + update * previous


def _gru_direction(
    np: Any,
    sequence: Any,
    arrays: Mapping[str, Any],
    level: str,
    layer: int,
    hidden: int,
    reverse: bool,
) -> tuple[Any, Any]:
    suffix = f"_l{layer}" + ("_reverse" if reverse else "")
    weights = tuple(
        arrays[f"{level}.{kind}{suffix}"]
        for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
    )
    # The four entries are structurally guaranteed by the derived tensor inventory.
    fixed_weights = (weights[0], weights[1], weights[2], weights[3])
    previous = np.zeros(hidden, dtype=np.float64)
    output = np.empty((len(sequence), hidden), dtype=np.float64)
    indices = range(len(sequence) - 1, -1, -1) if reverse else range(len(sequence))
    for index in indices:
        previous = _gru_step(np, sequence[index], previous, fixed_weights)
        output[index] = previous
    return output, previous


def infer_encoded_turns(
    parameters: FrozenNeuralParameters,
    turns: tuple[tuple[int, ...], ...],
    *,
    limits: NeuralNumericLimits | None = None,
) -> NeuralInference:
    """Score every completed turn, without padding, future labels or Torch.

    Eligibility and threshold selection belong to the higher-level forecaster.
    The returned probabilities are not claims of calibrated event likelihood.
    """
    if type(parameters) is not FrozenNeuralParameters:
        raise ValueError("inference requires FrozenNeuralParameters")
    selected = limits if limits is not None else parameters.limits
    work = _admit(parameters, turns, selected)
    np = _numpy()
    architecture = parameters.architecture
    arrays = {name: array.astype(np.float64) for name, array in parameters.arrays().items()}
    turn_vectors = []
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for turn in turns:
            sequence = arrays["embedding.weight"][list(turn)]
            for layer in range(architecture.word_layers):
                forward, forward_last = _gru_direction(
                    np, sequence, arrays, "word", layer, architecture.word_hidden, False
                )
                backward, backward_last = _gru_direction(
                    np, sequence, arrays, "word", layer, architecture.word_hidden, True
                )
                sequence = np.concatenate((forward, backward), axis=1)
            turn_vectors.append(np.concatenate((forward_last, backward_last)))
        contexts = np.stack(turn_vectors)
        for layer in range(architecture.turn_layers):
            contexts, _ = _gru_direction(
                np, contexts, arrays, "turn", layer, architecture.turn_hidden, False
            )
        hidden = np.tanh(contexts @ arrays["head.weight"].T + arrays["head.bias"])
        logits = (hidden @ arrays["output.weight"].T + arrays["output.bias"]).ravel()
        if not np.isfinite(logits).all():
            raise ValueError("neural inference produced nonfinite logits")
        probabilities = _sigmoid(np, logits)
    return NeuralInference(
        tuple(float(value) for value in logits),
        tuple(float(value) for value in probabilities),
        tuple(tuple(float(value) for value in row) for row in turn_vectors),
        tuple(tuple(float(value) for value in row) for row in contexts),
        work,
    )
