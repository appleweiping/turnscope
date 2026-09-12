"""Private CPU training for three closed, separately fitted neural controls.

No main-model forward function is patched. Initialization projects an untrained
canonical main inventory before optimizer construction; deployment is delegated
to the distinct NumPy ablation contract, not a main-model artifact.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import random
import re
from dataclasses import dataclass
from typing import Any

from .neural_ablation_math import (
    AblationPoolingLimits,
    FrozenAblationParameters,
    ablation_numerical_version,
    ablation_parameter_shapes,
    admit_ablation_turns,
)
from .neural_forecast_data import SequenceForecastDataset, SequenceLimits
from .neural_forecast_math import (
    NeuralArchitecture,
    NeuralNumericLimits,
    _integer,
    parameter_shapes,
)
from .neural_forecast_train import (
    INITIALIZATION_VERSION,
    NeuralEpoch,
    NeuralTrainingConfig,
    _batches,
    _EncodedConversation,
    _real,
    _torch,
    make_torch_hierarchy,
)
from .neural_token_data import (
    SequenceVocabulary,
    encode_observed_prefix,
    fit_sequence_vocabulary,
)

ABLATION_TRAINING_VERSION = "turnscope.neural-ablation.cpu-adam.v1"
ABLATION_INITIALIZATION_VERSION = "canonical-main-subset-init.v1"
_HASH = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class AblationTrainingLimits:
    """Full configured-epoch pooling estimate, independent of affine work."""

    max_total_pooling_operations: int = 100_000_000_000

    def __post_init__(self) -> None:
        _integer(self.max_total_pooling_operations, "max_total_pooling_operations", 1, 10**13)

    def to_dict(self) -> dict[str, int]:
        return {"max_total_pooling_operations": self.max_total_pooling_operations}


def _hashes(module: Any) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, hashlib.sha256(value.detach().numpy().astype("<f4").tobytes()).hexdigest())
        for name, value in module.state_dict().items()
    )


def _validate_hashes(value: Any, names: tuple[str, ...], label: str) -> None:
    if type(value) is not tuple or len(value) != len(names):
        raise ValueError(f"{label} must match the canonical inventory")
    for pair, expected in zip(value, names, strict=True):
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or pair[0] != expected
            or type(pair[1]) is not str
            or _HASH.fullmatch(pair[1]) is None
        ):
            raise ValueError(f"{label} contains a malformed name or digest")


@dataclass(frozen=True, slots=True)
class AblationTrainingResult:
    variant: str
    parameters: FrozenAblationParameters
    config: NeuralTrainingConfig
    training_limits: AblationTrainingLimits
    history: tuple[NeuralEpoch, ...]
    selected_epoch: int
    initial_parameter_sha256: tuple[tuple[str, str], ...]
    main_initial_parameter_sha256: tuple[tuple[str, str], ...]
    changed_parameter_names: tuple[str, ...]
    training_partition_digest: str
    validation_partition_digest: str
    vocabulary_digest: str
    training_conversations: int
    validation_conversations: int
    training_prefixes: int
    validation_prefixes: int
    maximum_estimated_workspace_bytes: int
    initialization_workspace_bytes: int
    estimated_affine_multiplications: int
    estimated_pooling_additions: int
    estimated_pooling_scalings: int
    torch_version: str
    numpy_version: str
    torch_threads: int
    initialization_version: str = ABLATION_INITIALIZATION_VERSION
    main_initialization_version: str = INITIALIZATION_VERSION

    def __post_init__(self) -> None:
        ablation_numerical_version(self.variant)
        if type(self.parameters) is not FrozenAblationParameters:
            raise ValueError("parameters must be FrozenAblationParameters")
        if self.variant != self.parameters.variant:
            raise ValueError("result variant differs from its parameters")
        if (
            type(self.config) is not NeuralTrainingConfig
            or type(self.training_limits) is not AblationTrainingLimits
        ):
            raise ValueError("result requires typed training configuration and limits")
        if (
            self.initialization_version != ABLATION_INITIALIZATION_VERSION
            or self.main_initialization_version != INITIALIZATION_VERSION
        ):
            raise ValueError("result initialization identity is unsupported")
        names = tuple(ablation_parameter_shapes(self.variant, self.parameters.architecture))
        main_names = tuple(parameter_shapes(self.parameters.architecture))
        _validate_hashes(self.initial_parameter_sha256, names, "initial hashes")
        _validate_hashes(self.main_initial_parameter_sha256, main_names, "main initial hashes")
        original = dict(self.main_initial_parameter_sha256)
        if any(original[name] != digest for name, digest in self.initial_parameter_sha256):
            raise ValueError("active initialization must be the canonical main subset")
        final = {
            tensor.name: hashlib.sha256(tensor.data).hexdigest()
            for tensor in self.parameters.tensors
        }
        changed = tuple(
            name for name, digest in self.initial_parameter_sha256 if final[name] != digest
        )
        if (
            type(self.changed_parameter_names) is not tuple
            or self.changed_parameter_names != changed
        ):
            raise ValueError("changed parameter names disagree with selected bytes")
        for name in (
            "training_partition_digest",
            "validation_partition_digest",
            "vocabulary_digest",
        ):
            value = getattr(self, name)
            if type(value) is not str or _HASH.fullmatch(value) is None:
                raise ValueError(f"{name} must be a SHA-256 digest")
        for name, minimum, maximum in (
            ("training_conversations", 2, 100_000),
            ("validation_conversations", 2, 100_000),
            ("training_prefixes", self.training_conversations, 100_000),
            ("validation_prefixes", self.validation_conversations, 100_000),
            ("maximum_estimated_workspace_bytes", 1, self.config.max_workspace_bytes),
            (
                "initialization_workspace_bytes",
                12 * self.parameters.architecture.parameter_count,
                self.maximum_estimated_workspace_bytes,
            ),
            ("estimated_affine_multiplications", 1, self.config.max_total_affine_multiplications),
            ("estimated_pooling_additions", 0, self.training_limits.max_total_pooling_operations),
            ("estimated_pooling_scalings", 0, self.training_limits.max_total_pooling_operations),
            ("torch_threads", 1, 100_000),
        ):
            _integer(getattr(self, name), name, minimum, maximum)
        if self.estimated_pooling_operations > self.training_limits.max_total_pooling_operations:
            raise ValueError("combined pooling work exceeds the training limit")
        if self.maximum_estimated_workspace_bytes < 48 * self.parameters.parameter_count:
            raise ValueError("workspace omits active parameter/optimizer storage")
        if self.variant == "current-turn.v1" and self.estimated_pooling_operations:
            raise ValueError("current-only has no mean pooling work")
        for value in (self.torch_version, self.numpy_version):
            if type(value) is not str or not 1 <= len(value) <= 128 or not value.isascii():
                raise ValueError("runtime versions must be bounded ASCII strings")
        if type(self.history) is not tuple or not 1 <= len(self.history) <= self.config.epochs:
            raise ValueError("history must contain bounded consecutive epochs")
        _integer(self.selected_epoch, "selected_epoch", 1, len(self.history))
        best = math.inf
        earliest = stale = 0
        for position, epoch in enumerate(self.history, start=1):
            if (
                type(epoch) is not NeuralEpoch
                or type(epoch.epoch) is not int
                or epoch.epoch != position
            ):
                raise ValueError("history must contain consecutive typed epochs")
            _real(epoch.training_loss, "training_loss", 0, 1e100)
            _real(epoch.validation_loss, "validation_loss", 0, 1e100)
            _real(epoch.maximum_unclipped_gradient_norm, "gradient_norm", 0, 1e100)
            _integer(
                epoch.optimizer_steps,
                "optimizer_steps",
                math.ceil(self.training_conversations / self.config.batch_conversations),
                self.training_conversations,
            )
            if epoch.validation_loss < best:
                best, earliest, stale = epoch.validation_loss, position, 0
            else:
                stale += 1
            if stale >= self.config.patience and position != len(self.history):
                raise ValueError("history continued beyond the stopping rule")
        if self.selected_epoch != earliest or (
            len(self.history) < self.config.epochs and stale < self.config.patience
        ):
            raise ValueError("selected epoch or early-stop history is inconsistent")

    @property
    def estimated_pooling_operations(self) -> int:
        return self.estimated_pooling_additions + self.estimated_pooling_scalings


def make_torch_ablation(
    torch: Any, variant: str, architecture: NeuralArchitecture, *, seed: int
) -> tuple[Any, tuple[tuple[str, str], ...]]:
    """Canonical private main initialization, then discard inactive parameters."""
    expected = ablation_parameter_shapes(variant, architecture)
    module = make_torch_hierarchy(torch, architecture, seed=seed)
    original = _hashes(module)
    if variant != "current-turn.v1":
        del module.word
    if variant == "order-erased.v1":
        del module.turn
    elif variant == "current-turn.v1":
        old = module.turn
        turn = torch.nn.Module()
        for name in ("weight_ih_l0", "bias_ih_l0", "bias_hh_l0"):
            turn.register_parameter(name, getattr(old, name))
        module.turn = turn
        del old
    if {name: tuple(value.shape) for name, value in module.state_dict().items()} != expected:
        raise ValueError("Torch inventory differs from the closed ablation contract")
    return module, original


def _word_vectors(torch: Any, module: Any, turns: list[tuple[int, ...]], variant: str) -> Any:
    if variant != "current-turn.v1":
        flat = [token for turn in turns for token in turn]
        offsets = [0]
        for turn in turns:
            offsets.append(offsets[-1] + len(turn))
        vectors = torch.nn.functional.embedding_bag(
            torch.tensor(flat, dtype=torch.long, device="cpu"),
            module.embedding.weight,
            torch.tensor(offsets, dtype=torch.long, device="cpu"),
            mode="mean",
            include_last_offset=True,
            padding_idx=0,
        )
        return torch.cat((vectors, vectors), dim=1) if variant == "mean-word.v1" else vectors
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
    return torch.cat((final[-2], final[-1]), dim=1).index_select(
        0, torch.tensor(inverse, dtype=torch.long, device="cpu")
    )


def forward_ablation_batch(
    torch: Any, module: Any, batch: tuple[_EncodedConversation, ...], *, variant: str
) -> tuple[Any, ...]:
    """Private admitted batches: compact turns, no copied or future-filled prefixes."""
    ablation_numerical_version(variant)
    vectors = _word_vectors(torch, module, [turn for item in batch for turn in item.turns], variant)
    parts = list(vectors.split([len(item.turns) for item in batch]))
    if variant == "current-turn.v1":
        incoming = torch.nn.functional.linear(
            vectors, module.turn.weight_ih_l0, module.turn.bias_ih_l0
        )
        ir, iz, inn = incoming.chunk(3, dim=1)
        hr, hz, hn = module.turn.bias_hh_l0.chunk(3)
        states = (1 - torch.sigmoid(iz + hz)) * torch.tanh(inn + torch.sigmoid(ir + hr) * hn)
        contexts = list(states.split([len(item.turns) for item in batch]))
    elif variant == "order-erased.v1":
        contexts = [
            part.cumsum(0)
            / torch.arange(1, len(part) + 1, dtype=torch.float32, device="cpu").unsqueeze(1)
            for part in parts
        ]
    else:
        order = sorted(range(len(parts)), key=lambda index: (-len(parts[index]), index))
        inverse = sorted(range(len(order)), key=order.__getitem__)
        padded = torch.nn.utils.rnn.pad_sequence([parts[index] for index in order])
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            padded, [len(parts[index]) for index in order], enforce_sorted=True
        )
        output, _ = module.turn(packed)
        states, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        states = states.index_select(1, torch.tensor(inverse, dtype=torch.long, device="cpu"))
        contexts = [states[: len(item.turns), column] for column, item in enumerate(batch)]
    return tuple(
        module.output(torch.tanh(module.head(context))).squeeze(-1) for context in contexts
    )


def _batch_loss(
    torch: Any, module: Any, batch: tuple[_EncodedConversation, ...], variant: str
) -> Any:
    logits = forward_ablation_batch(torch, module, batch, variant=variant)
    return torch.stack(
        [
            torch.nn.functional.binary_cross_entropy_with_logits(
                values[list(item.endpoints)],
                torch.full_like(values[list(item.endpoints)], float(item.label)),
            )
            for item, values in zip(batch, logits, strict=True)
        ]
    ).mean()


def _validation_loss(
    torch: Any,
    module: Any,
    records: tuple[_EncodedConversation, ...],
    batches: tuple[tuple[int, ...], ...],
    variant: str,
) -> float:
    module.eval()
    values = []
    with torch.no_grad():
        for indices in batches:
            loss = float(
                _batch_loss(torch, module, tuple(records[index] for index in indices), variant)
            )
            if not math.isfinite(loss):
                raise ValueError("validation produced a nonfinite loss")
            values.append(loss * len(indices))
    return math.fsum(values) / len(records)


def _encode_partition(
    data: SequenceForecastDataset,
    vocabulary: SequenceVocabulary,
    architecture: NeuralArchitecture,
    variant: str,
    numeric: NeuralNumericLimits,
    limits: SequenceLimits,
    pooling: AblationPoolingLimits,
) -> tuple[tuple[_EncodedConversation, ...], tuple[int, int, int]]:
    data.validate_limits(limits)
    if not data.observations or {example.label for example in data.examples} != {False, True}:
        raise ValueError("each fitting partition requires eligible conversations from both classes")
    endpoints: dict[int, list[int]] = {}
    labels: dict[int, bool] = {}
    for example in data.examples:
        endpoints.setdefault(example.conversation_index, []).append(example.endpoint)
        labels[example.conversation_index] = example.label
    result = []
    affine = additions = scalings = 0
    for index, observation in enumerate(data.observations):
        encoded = encode_observed_prefix(observation, vocabulary, limits=limits)
        work = admit_ablation_turns(
            variant, architecture, encoded.turns, limits=numeric, pooling_limits=pooling
        )
        affine += work.affine_multiplications
        additions += work.pooling_additions
        scalings += work.pooling_scalings
        result.append(_EncodedConversation(encoded.turns, tuple(endpoints[index]), labels[index]))
    return tuple(result), (affine, additions, scalings)


def _workspace(
    variant: str, architecture: NeuralArchitecture, batch: tuple[_EncodedConversation, ...]
) -> int:
    active = sum(
        math.prod(shape) for shape in ablation_parameter_shapes(variant, architecture).values()
    )
    turns = [turn for item in batch for turn in item.turns]
    positions = sum(map(len, turns))
    padded_words = max(map(len, turns)) * len(turns)
    padded_turns = max(len(item.turns) for item in batch) * len(batch)
    # Float32 parameter/gradient/Adam/selected copies plus explicit activation
    # allowances. Integer token, offset and length buffers are included separately.
    if variant == "current-turn.v1":
        activations = padded_words * (64 + 32 * 64) + len(turns) * 20 * 64
    elif variant == "mean-word.v1":
        activations = positions * 3 * 64 + padded_turns * (16 * 64 + 128)
    else:
        activations = positions * 3 * 64 + len(turns) * 8 * 64
    return (
        48 * active + 4 * activations + 8 * (padded_words + positions + 4 * len(turns) + len(batch))
    )


def train_ablation_model(
    training: SequenceForecastDataset,
    validation: SequenceForecastDataset,
    vocabulary: SequenceVocabulary,
    architecture: NeuralArchitecture,
    *,
    variant: str,
    config: NeuralTrainingConfig | None = None,
    numeric_limits: NeuralNumericLimits | None = None,
    data_limits: SequenceLimits | None = None,
    pooling_limits: AblationPoolingLimits | None = None,
    training_limits: AblationTrainingLimits | None = None,
) -> AblationTrainingResult:
    """Fit a private alternate encoder, selecting its earliest best validation epoch.

    All data/configuration, full possible-epoch schedules and resource estimates
    are admitted before Torch import or parameter allocation. No provider, file,
    threshold selection, global reseeding or process-setting change occurs.
    """
    settings = config if config is not None else NeuralTrainingConfig()
    numeric = numeric_limits if numeric_limits is not None else NeuralNumericLimits()
    data = data_limits if data_limits is not None else SequenceLimits()
    pooling = pooling_limits if pooling_limits is not None else AblationPoolingLimits()
    budget = training_limits if training_limits is not None else AblationTrainingLimits()
    for typed_value, kind in (
        (settings, NeuralTrainingConfig),
        (numeric, NeuralNumericLimits),
        (data, SequenceLimits),
        (pooling, AblationPoolingLimits),
        (budget, AblationTrainingLimits),
        (training, SequenceForecastDataset),
        (validation, SequenceForecastDataset),
        (vocabulary, SequenceVocabulary),
    ):
        if type(typed_value) is not kind:
            raise ValueError(
                "training requires typed partitions, vocabulary, configuration and limits"
            )
    ablation_parameter_shapes(variant, architecture)
    if architecture.vocabulary_size != vocabulary.size:
        raise ValueError("architecture must match the vocabulary size")
    if training.group_digests & validation.group_digests:
        raise ValueError("training and model-validation groups overlap")
    # The full canonical initializer exists transiently, even for a small mode.
    if architecture.parameter_count > numeric.max_parameters:
        raise ValueError("canonical main initializer exceeds max_parameters")
    expected = fit_sequence_vocabulary(
        training,
        min_document_frequency=settings.min_document_frequency,
        max_features=settings.max_features,
        max_turn_tokens=vocabulary.max_turn_tokens,
        long_turn_policy=vocabulary.long_turn_policy,
        limits=data,
    )
    if vocabulary != expected:
        raise ValueError("vocabulary must be derived exclusively from the training partition")
    train, train_work = _encode_partition(
        training, vocabulary, architecture, variant, numeric, data, pooling
    )
    val, val_work = _encode_partition(
        validation, vocabulary, architecture, variant, numeric, data, pooling
    )
    total_affine, total_additions, total_scalings = (
        settings.epochs * (3 * left + right)
        for left, right in zip(train_work, val_work, strict=True)
    )
    if total_affine > settings.max_total_affine_multiplications:
        raise ValueError("training exceeds max_total_affine_multiplications")
    if total_additions + total_scalings > budget.max_total_pooling_operations:
        raise ValueError("training exceeds max_total_pooling_operations")
    # Local reproducible sample ordering is not a secret-generation use.
    rng = random.Random(settings.seed)  # nosec B311
    schedules = []
    initial_workspace = 12 * architecture.parameter_count
    maximum_workspace = initial_workspace
    for _ in range(settings.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        schedule = _batches(train, order, settings)
        schedules.append(schedule)
        for indices in schedule:
            maximum_workspace = max(
                maximum_workspace,
                _workspace(variant, architecture, tuple(train[index] for index in indices)),
            )
    val_schedule = _batches(val, list(range(len(val))), settings)
    for indices in val_schedule:
        maximum_workspace = max(
            maximum_workspace,
            _workspace(variant, architecture, tuple(val[index] for index in indices)),
        )
    if maximum_workspace > settings.max_workspace_bytes:
        raise ValueError("training exceeds max_workspace_bytes estimate")
    torch = _torch()
    module, main_initial = make_torch_ablation(torch, variant, architecture, seed=settings.seed)
    with torch.no_grad():
        if any(
            not bool(torch.isfinite(parameter).all())
            or bool((parameter.abs() > numeric.max_parameter_magnitude).any())
            for parameter in module.parameters()
        ):
            raise ValueError("initialized parameters exceed finite/magnitude limits")
    initial = _hashes(module)
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
    selected_epoch = stale = 0
    for epoch, schedule in enumerate(schedules, start=1):
        module.train()
        losses = []
        gradient_norm = 0.0
        for indices in schedule:
            batch = tuple(train[index] for index in indices)
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(torch, module, batch, variant)
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
            losses.append(value * len(indices))
        validation_loss = _validation_loss(torch, module, val, val_schedule, variant)
        history.append(
            NeuralEpoch(
                epoch, math.fsum(losses) / len(train), validation_loss, len(schedule), gradient_norm
            )
        )
        if validation_loss < best_loss:
            best_loss, selected_epoch, stale = validation_loss, epoch, 0
            best_state = {
                name: value.detach().clone() for name, value in module.state_dict().items()
            }
        else:
            stale += 1
        if stale >= settings.patience:
            break
    if best_state is None:
        raise ValueError("training selected no finite epoch")
    frozen = FrozenAblationParameters.from_arrays(
        variant,
        architecture,
        {name: value.numpy().astype("<f4", copy=True) for name, value in best_state.items()},
        limits=numeric,
        pooling_limits=pooling,
    )
    final = {tensor.name: hashlib.sha256(tensor.data).hexdigest() for tensor in frozen.tensors}
    return AblationTrainingResult(
        variant,
        frozen,
        settings,
        budget,
        tuple(history),
        selected_epoch,
        initial,
        main_initial,
        tuple(name for name, digest in initial if final[name] != digest),
        training.digest,
        validation.digest,
        vocabulary.digest,
        len(train),
        len(val),
        len(training.examples),
        len(validation.examples),
        maximum_workspace,
        initial_workspace,
        total_affine,
        total_additions,
        total_scalings,
        str(torch.__version__),
        str(importlib.import_module("numpy").__version__),
        int(torch.get_num_threads()),
    )
