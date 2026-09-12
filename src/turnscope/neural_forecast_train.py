"""Optional CPU training of all levels of the ordered hierarchical GRU.

No pretrained weights, network calls, global RNG reseeding, pickle or Torch
checkpoint deserialization. A returned candidate does not yet select an alert
threshold or establish heldout model quality.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Any

from .neural_forecast_data import SequenceForecastDataset, SequenceLimits
from .neural_forecast_math import (
    FrozenNeuralParameters,
    NeuralArchitecture,
    NeuralNumericLimits,
    _integer,
    admit_encoded_turns,
    parameter_shapes,
)
from .neural_token_data import (
    SequenceVocabulary,
    encode_observed_prefix,
    fit_sequence_vocabulary,
)

TRAINING_VERSION = "turnscope.hierarchical-gru.cpu-adam.v1"
INITIALIZATION_VERSION = "local-cpu-generator.uniform-dimension-bound.v1"


def _real(value: Any, name: str, minimum: float, maximum: float) -> float:
    if (
        type(value) not in (float, int)
        or not minimum <= value <= maximum
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite in [{minimum}, {maximum}]")
    return float(value)


@dataclass(frozen=True, slots=True)
class NeuralTrainingConfig:
    min_document_frequency: int = 1
    max_features: int = 10_000
    epochs: int = 8
    patience: int = 3
    batch_conversations: int = 16
    max_batch_token_positions: int = 8192
    max_workspace_bytes: int = 512 * 1024 * 1024
    max_total_affine_multiplications: int = 2_000_000_000_000
    learning_rate: float = 0.001
    gradient_clip: float = 5.0
    seed: int = 0

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("min_document_frequency", 1, 100_000),
            ("max_features", 1, 100_000),
            ("epochs", 1, 100),
            ("patience", 1, 100),
            ("batch_conversations", 1, 128),
            ("max_batch_token_positions", 1, 262_400),
            ("max_workspace_bytes", 1, 2 * 1024 * 1024 * 1024),
            ("max_total_affine_multiplications", 1, 100_000_000_000_000),
            ("seed", 0, 2**32 - 1),
        ):
            _integer(getattr(self, name), name, minimum, maximum)
        _real(self.learning_rate, "learning_rate", 1e-8, 1.0)
        _real(self.gradient_clip, "gradient_clip", 1e-6, 1e6)

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NeuralEpoch:
    epoch: int
    training_loss: float
    validation_loss: float
    optimizer_steps: int
    maximum_unclipped_gradient_norm: float


@dataclass(frozen=True, slots=True)
class NeuralTrainingResult:
    parameters: FrozenNeuralParameters
    config: NeuralTrainingConfig
    history: tuple[NeuralEpoch, ...]
    selected_epoch: int
    initial_parameter_sha256: tuple[tuple[str, str], ...]
    changed_parameter_names: tuple[str, ...]
    training_partition_digest: str
    validation_partition_digest: str
    vocabulary_digest: str
    training_conversations: int
    validation_conversations: int
    training_prefixes: int
    validation_prefixes: int
    maximum_estimated_workspace_bytes: int
    estimated_affine_multiplications: int
    torch_version: str
    numpy_version: str
    torch_threads: int


@dataclass(frozen=True, slots=True)
class _EncodedConversation:
    turns: tuple[tuple[int, ...], ...]
    endpoints: tuple[int, ...]
    label: bool

    @property
    def positions(self) -> int:
        return sum(map(len, self.turns))


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError(
            "CPU neural training requires PyTorch; install turnscope[neural-train]"
        ) from error


def _encode_partition(
    data: SequenceForecastDataset,
    vocabulary: SequenceVocabulary,
    architecture: NeuralArchitecture,
    numeric_limits: NeuralNumericLimits,
    data_limits: SequenceLimits,
) -> tuple[tuple[_EncodedConversation, ...], int]:
    data.validate_limits(data_limits)
    if not data.observations or {example.label for example in data.examples} != {False, True}:
        raise ValueError("each fitting partition requires eligible conversations from both classes")
    by_conversation: dict[int, list[int]] = {}
    labels: dict[int, bool] = {}
    for example in data.examples:
        by_conversation.setdefault(example.conversation_index, []).append(example.endpoint)
        labels[example.conversation_index] = example.label
    result = []
    work = 0
    for index, observation in enumerate(data.observations):
        encoded = encode_observed_prefix(observation, vocabulary, limits=data_limits)
        admission = admit_encoded_turns(architecture, encoded.turns, limits=numeric_limits)
        work += admission.affine_multiplications
        result.append(
            _EncodedConversation(encoded.turns, tuple(by_conversation[index]), labels[index])
        )
    return tuple(result), work


def _workspace(architecture: NeuralArchitecture, batch: tuple[_EncodedConversation, ...]) -> int:
    turns = [turn for conversation in batch for turn in conversation.turns]
    padded_words = max(map(len, turns)) * len(turns)
    padded_turns = max(len(item.turns) for item in batch) * len(batch)
    return 48 * architecture.parameter_count + 4 * (
        padded_words
        * (architecture.embedding_dim + 32 * architecture.word_layers * architecture.word_hidden)
        + padded_turns
        * (16 * architecture.turn_layers * architecture.turn_hidden + 2 * architecture.word_hidden)
        + sum(map(len, turns)) * architecture.embedding_dim * 3
    )


def _batches(
    records: tuple[_EncodedConversation, ...], order: list[int], config: NeuralTrainingConfig
) -> tuple[tuple[int, ...], ...]:
    batches = []
    current: list[int] = []
    positions = 0
    for index in order:
        count = records[index].positions
        if count > config.max_batch_token_positions:
            raise ValueError(
                "one conversation exceeds max_batch_token_positions; nothing is truncated"
            )
        if current and (
            len(current) == config.batch_conversations
            or positions + count > config.max_batch_token_positions
        ):
            batches.append(tuple(current))
            current, positions = [], 0
        current.append(index)
        positions += count
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def make_torch_hierarchy(torch: Any, architecture: NeuralArchitecture, *, seed: int) -> Any:
    """Create explicitly initialized CPU layers without consuming global RNG state."""
    _integer(seed, "seed", 0, 2**32 - 1)
    if type(architecture) is not NeuralArchitecture:
        raise ValueError("architecture must be NeuralArchitecture")
    # Meta constructors allocate no real tensors and consume no CPU RNG values.
    # Every real parameter is then initialized using our own CPU generator.
    module = torch.nn.Module()
    factory = {"device": "meta", "dtype": torch.float32}
    module.add_module(
        "embedding",
        torch.nn.Embedding(
            architecture.vocabulary_size, architecture.embedding_dim, padding_idx=0, **factory
        ),
    )
    module.add_module(
        "word",
        torch.nn.GRU(
            architecture.embedding_dim,
            architecture.word_hidden,
            architecture.word_layers,
            bidirectional=True,
            dropout=0.0,
            **factory,
        ),
    )
    module.add_module(
        "turn",
        torch.nn.GRU(
            2 * architecture.word_hidden,
            architecture.turn_hidden,
            architecture.turn_layers,
            dropout=0.0,
            **factory,
        ),
    )
    module.add_module(
        "head", torch.nn.Linear(architecture.turn_hidden, architecture.head_hidden, **factory)
    )
    module.add_module("output", torch.nn.Linear(architecture.head_hidden, 1, **factory))
    module.to_empty(device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            dimension = {
                "embedding": architecture.embedding_dim,
                "word": architecture.word_hidden,
                "turn": architecture.turn_hidden,
                "head": architecture.turn_hidden,
                "output": architecture.head_hidden,
            }[name.split(".")[0]]
            bound = 1 / math.sqrt(dimension)
            parameter.uniform_(-bound, bound, generator=generator)
        module.embedding.weight[0].zero_()
    return module


def forward_encoded_batch(
    torch: Any, module: Any, batch: tuple[_EncodedConversation, ...]
) -> tuple[Any, ...]:
    """Private admitted batches only; unique turns encoded once, no prefix copies."""
    turns = [turn for conversation in batch for turn in conversation.turns]
    order = sorted(range(len(turns)), key=lambda index: (-len(turns[index]), index))
    inverse = sorted(range(len(order)), key=order.__getitem__)
    lengths = [len(turns[index]) for index in order]
    padded = torch.zeros((max(lengths), len(turns)), dtype=torch.long, device="cpu")
    for column, index in enumerate(order):
        padded[: len(turns[index]), column] = torch.tensor(
            turns[index], dtype=torch.long, device="cpu"
        )
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        module.embedding(padded), lengths, enforce_sorted=True
    )
    _, final = module.word(packed)
    vectors = torch.cat((final[-2], final[-1]), dim=1).index_select(
        0, torch.tensor(inverse, dtype=torch.long, device="cpu")
    )
    parts = list(vectors.split([len(record.turns) for record in batch]))
    context_order = sorted(range(len(parts)), key=lambda index: (-len(parts[index]), index))
    context_inverse = sorted(range(len(context_order)), key=context_order.__getitem__)
    context_lengths = [len(parts[index]) for index in context_order]
    context_padded = torch.nn.utils.rnn.pad_sequence([parts[index] for index in context_order])
    context_packed = torch.nn.utils.rnn.pack_padded_sequence(
        context_padded, context_lengths, enforce_sorted=True
    )
    output, _ = module.turn(context_packed)
    context, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
    context = context.index_select(1, torch.tensor(context_inverse, dtype=torch.long, device="cpu"))
    logits = module.output(torch.tanh(module.head(context))).squeeze(-1)
    return tuple(logits[: len(record.turns), column] for column, record in enumerate(batch))


def _batch_loss(torch: Any, module: Any, batch: tuple[_EncodedConversation, ...]) -> Any:
    logits = forward_encoded_batch(torch, module, batch)
    losses = []
    for record, values in zip(batch, logits, strict=True):
        selected = values[list(record.endpoints)]
        labels = torch.full_like(selected, float(record.label))
        losses.append(torch.nn.functional.binary_cross_entropy_with_logits(selected, labels))
    return torch.stack(losses).mean()


def _validation_loss(
    torch: Any,
    module: Any,
    records: tuple[_EncodedConversation, ...],
    batches: tuple[tuple[int, ...], ...],
) -> float:
    module.eval()
    values = []
    with torch.no_grad():
        for indices in batches:
            loss = float(_batch_loss(torch, module, tuple(records[index] for index in indices)))
            if not math.isfinite(loss):
                raise ValueError("validation produced a nonfinite loss")
            values.append(loss * len(indices))
    return math.fsum(values) / len(records)


def train_sequence_model(
    training: SequenceForecastDataset,
    validation: SequenceForecastDataset,
    vocabulary: SequenceVocabulary,
    architecture: NeuralArchitecture,
    *,
    config: NeuralTrainingConfig | None = None,
    numeric_limits: NeuralNumericLimits | None = None,
    data_limits: SequenceLimits | None = None,
) -> NeuralTrainingResult:
    """Train a private candidate; select earliest lowest validation-loss epoch.

    Batch loss is a mean of per-conversation eligible-prefix means. All retained
    conversations and the final partial batch are included. Bounds include
    padding estimates; native RSS and cross-version bitwise equality are not
    promised. Existing callers/models are never mutated on failure.
    """
    settings = config if config is not None else NeuralTrainingConfig()
    numeric = numeric_limits if numeric_limits is not None else NeuralNumericLimits()
    data = data_limits if data_limits is not None else SequenceLimits()
    if (
        type(settings) is not NeuralTrainingConfig
        or type(numeric) is not NeuralNumericLimits
        or type(data) is not SequenceLimits
    ):
        raise ValueError("training requires typed configuration and limits")
    if (
        type(training) is not SequenceForecastDataset
        or type(validation) is not SequenceForecastDataset
        or type(vocabulary) is not SequenceVocabulary
    ):
        raise ValueError("training requires typed sequence partitions and vocabulary")
    if (
        type(architecture) is not NeuralArchitecture
        or architecture.vocabulary_size != vocabulary.size
    ):
        raise ValueError("architecture must match the vocabulary size")
    if training.group_digests & validation.group_digests:
        raise ValueError("training and model-validation groups overlap")
    if architecture.parameter_count > numeric.max_parameters:
        raise ValueError("model exceeds max_parameters")
    expected_vocabulary = fit_sequence_vocabulary(
        training,
        min_document_frequency=settings.min_document_frequency,
        max_features=settings.max_features,
        max_turn_tokens=vocabulary.max_turn_tokens,
        long_turn_policy=vocabulary.long_turn_policy,
        limits=data,
    )
    if vocabulary != expected_vocabulary:
        raise ValueError("vocabulary must be derived exclusively from the training partition")
    train, train_work = _encode_partition(training, vocabulary, architecture, numeric, data)
    val, val_work = _encode_partition(validation, vocabulary, architecture, numeric, data)
    total_work = settings.epochs * (3 * train_work + val_work)
    if total_work > settings.max_total_affine_multiplications:
        raise ValueError("training exceeds max_total_affine_multiplications")
    # Pre-admit the actual seeded batch schedules for every possible epoch.
    # The multiplier 3 estimates forward+backward work, not measured FLOPs.
    # Reproducible sample ordering is not a cryptographic or secret-generation use.
    rng = random.Random(settings.seed)  # nosec B311
    schedules = []
    maximum_workspace = 0
    for _ in range(settings.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        schedule = _batches(train, order, settings)
        schedules.append(schedule)
        for indices in schedule:
            maximum_workspace = max(
                maximum_workspace,
                _workspace(architecture, tuple(train[index] for index in indices)),
            )
    val_schedule = _batches(val, list(range(len(val))), settings)
    for indices in val_schedule:
        maximum_workspace = max(
            maximum_workspace, _workspace(architecture, tuple(val[index] for index in indices))
        )
    if maximum_workspace > settings.max_workspace_bytes:
        raise ValueError("training exceeds max_workspace_bytes estimate")
    # Numerical dependencies and all parameter allocation occur after admission.
    torch = _torch()
    module = make_torch_hierarchy(torch, architecture, seed=settings.seed)
    if {
        name: tuple(value.shape) for name, value in module.state_dict().items()
    } != parameter_shapes(architecture):
        raise ValueError("Torch parameter inventory differs from the numerical contract")
    initial = tuple(
        (name, hashlib.sha256(value.detach().numpy().tobytes()).hexdigest())
        for name, value in module.state_dict().items()
    )
    optimizer = torch.optim.Adam(
        module.parameters(),
        lr=settings.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    history = []
    best_state = None
    best_loss = math.inf
    selected_epoch = 0
    stale = 0
    for epoch, schedule in enumerate(schedules, start=1):
        module.train()
        epoch_losses = []
        gradient_norm = 0.0
        for indices in schedule:
            batch = tuple(train[index] for index in indices)
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(torch, module, batch)
            value = float(loss.detach())
            if not math.isfinite(value):
                raise ValueError("training produced a nonfinite loss")
            loss.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    module.parameters(),
                    settings.gradient_clip,
                    error_if_nonfinite=True,
                    foreach=False,
                )
            )
            gradient_norm = max(gradient_norm, norm)
            optimizer.step()
            with torch.no_grad():
                if any(
                    not bool(torch.isfinite(parameter).all())
                    or bool((parameter.abs() > numeric.max_parameter_magnitude).any())
                    for parameter in module.parameters()
                ):
                    raise ValueError("optimizer produced nonfinite or oversized parameters")
            epoch_losses.append(value * len(indices))
        validation_loss = _validation_loss(torch, module, val, val_schedule)
        history.append(
            NeuralEpoch(
                epoch,
                math.fsum(epoch_losses) / len(train),
                validation_loss,
                len(schedule),
                gradient_norm,
            )
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            selected_epoch = epoch
            best_state = {
                name: value.detach().clone() for name, value in module.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= settings.patience:
            break
    if best_state is None:
        raise ValueError("training selected no finite epoch")
    arrays = {name: value.numpy().astype("<f4", copy=True) for name, value in best_state.items()}
    frozen = FrozenNeuralParameters.from_arrays(architecture, arrays, limits=numeric)
    final_hashes = {
        tensor.name: hashlib.sha256(tensor.data).hexdigest() for tensor in frozen.tensors
    }
    return NeuralTrainingResult(
        frozen,
        settings,
        tuple(history),
        selected_epoch,
        initial,
        tuple(name for name, digest in initial if final_hashes[name] != digest),
        training.digest,
        validation.digest,
        vocabulary.digest,
        len(train),
        len(val),
        len(training.examples),
        len(validation.examples),
        maximum_workspace,
        total_work,
        str(torch.__version__),
        str(importlib.import_module("numpy").__version__),
        int(torch.get_num_threads()),
    )
