"""Distinct, closed inference-only archives for the three neural controls.

The canonical container primitives are shared with the main forecaster. Neither
loader guesses a mode or converts a trained main artifact into a control.
"""

from __future__ import annotations

import math
import os
import struct
import tempfile
from pathlib import Path
from typing import Any

from . import neural_forecast_artifact as _base
from .neural_ablation import AblationEventForecaster, AblationForecastConfig, AblationForecastState
from .neural_ablation_math import (
    AblationInferenceWork,
    AblationPoolingLimits,
    FrozenAblationParameters,
    ablation_numerical_version,
    ablation_parameter_shapes,
    admit_ablation_turns,
)
from .neural_ablation_train import (
    ABLATION_INITIALIZATION_VERSION,
    ABLATION_TRAINING_VERSION,
    AblationTrainingLimits,
    AblationTrainingResult,
)
from .neural_forecast_artifact import NeuralArtifactError, NeuralArtifactLimits, NeuralSaveResult
from .neural_forecast_data import SequenceLimits, SequencePolicy
from .neural_forecast_math import FrozenNeuralTensor, NeuralNumericLimits, parameter_shapes
from .neural_forecast_train import NeuralEpoch, NeuralTrainingConfig
from .neural_token_data import SequenceVocabulary

ABLATION_ARTIFACT_FORMAT = "turnscope.neural-ablation-artifact.v1"
_SUMMARY_FIELDS = _base._SUMMARY_FIELDS | {
    "variant",
    "numerical_version",
    "reference_architecture",
    "pooling_limits",
    "training_limits",
    "initialization_version",
    "main_initialization_version",
    "canonical_main_parameter_count",
    "fixed_pad_parameters",
    "main_initial_parameter_sha256",
    "initialization_workspace_bytes",
    "estimated_pooling_additions",
    "estimated_pooling_scalings",
}


def _tensors(
    manifest: dict[str, Any],
    members: dict[str, memoryview],
    shapes: dict[str, tuple[int, ...]],
    limits: NeuralArtifactLimits,
) -> tuple[FrozenNeuralTensor, ...]:
    descriptors = _base._array(manifest["tensors"], len(shapes), minimum=len(shapes))
    if list(members) != ["manifest.json", *(f"tensors/{name}.npy" for name in shapes)]:
        raise NeuralArtifactError("archive inventory differs from the declared ablation")
    admitted = []
    for (name, shape), descriptor in zip(shapes.items(), descriptors, strict=True):
        item = _base._object(
            descriptor,
            {"name", "member", "dtype", "shape", "data_bytes", "data_sha256", "member_sha256"},
        )
        member = f"tensors/{name}.npy"
        if (
            item["name"] != name
            or item["member"] != member
            or item["dtype"] != "<f4"
            or _base._canonical(item["shape"], limits) != _base._canonical(list(shape), limits)
            or _base._integer(item["data_bytes"], 4, 4 * limits.max_parameters)
            != 4 * math.prod(shape)
        ):
            raise NeuralArtifactError("tensor descriptor differs from the derived ablation shape")
        payload = members[member]
        header = _base._npy_header(shape)
        if len(payload) != len(header) + item["data_bytes"] or payload[: len(header)] != header:
            raise NeuralArtifactError("tensor must have the exact canonical NPY 1.0 header")
        raw = payload[len(header) :]
        if _base._sha(raw) != _base._digest(item["data_sha256"]) or _base._sha(
            payload
        ) != _base._digest(item["member_sha256"]):
            raise NeuralArtifactError("tensor integrity digest differs")
        admitted.append((name, shape, raw))
    # Inventory, every header and every digest are checked before tensor copies.
    for _, _, raw in admitted:
        for (value,) in struct.iter_unpack("<f", raw):
            if not math.isfinite(value) or abs(value) > limits.max_parameter_magnitude:
                raise NeuralArtifactError("tensor exceeds the artifact magnitude budget")
    return tuple(FrozenNeuralTensor(name, shape, bytes(raw)) for name, shape, raw in admitted)


def _work(
    summary: dict[str, Any],
    config: AblationForecastConfig,
    training: NeuralTrainingConfig,
    training_limits: AblationTrainingLimits,
    policy: SequencePolicy,
    minimum: AblationInferenceWork,
    supports: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> None:
    # Each distinct eligible endpoint requires a distinct completed turn; every
    # conversation also has at least min_turns-1 preceding turns. EOS-only turns
    # give a lower bound without claiming to recover omitted source lengths.
    lower = []
    for conversations, prefixes in supports:
        turns = prefixes + conversations * (policy.min_turns - 1)
        affine = turns * (minimum.affine_multiplications // policy.min_turns)
        additions = (turns - conversations) * 64 if config.variant == "order-erased.v1" else 0
        scalings = turns * (minimum.pooling_scalings // policy.min_turns)
        lower.append((affine, additions, scalings))
    for index, (field, cap) in enumerate(
        (
            ("estimated_affine_multiplications", training.max_total_affine_multiplications),
            ("estimated_pooling_additions", training_limits.max_total_pooling_operations),
            ("estimated_pooling_scalings", training_limits.max_total_pooling_operations),
        )
    ):
        value = _base._integer(summary[field], 0, cap)
        required = (3 * lower[0][index] + lower[1][index]) * training.epochs
        if value % training.epochs or value < required:
            raise NeuralArtifactError("training work omits configured epochs or minimum support")
    if (
        lower[2][0] > config.max_inference_affine_multiplications
        or lower[2][1] + lower[2][2] > config.max_inference_pooling_operations
    ):
        raise NeuralArtifactError("policy validation support exceeds an inference work budget")


def _restore(
    manifest: dict[str, Any], members: dict[str, memoryview], limits: NeuralArtifactLimits
) -> AblationEventForecaster:
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    if _base._digest(manifest["sha256"]) != _base._sha(_base._canonical(body, limits)):
        raise NeuralArtifactError("manifest integrity digest differs")
    for key, expected in (
        ("format", ABLATION_ARTIFACT_FORMAT),
        ("training_version", ABLATION_TRAINING_VERSION),
        ("initialization_version", ABLATION_INITIALIZATION_VERSION),
    ):
        if manifest[key] != expected:
            raise NeuralArtifactError("unsupported ablation artifact version")
    values = _base._object(
        manifest["state"],
        {
            "summary",
            "vocabulary",
            "training_groups",
            "model_validation_groups",
            "policy_validation_groups",
        },
    )
    summary = _base._object(values["summary"], _SUMMARY_FIELDS)
    config = _base._typed(AblationForecastConfig, summary["config"])
    if manifest["numerical_version"] != ablation_numerical_version(config.variant):
        raise NeuralArtifactError("artifact numerical version differs from its variant")
    policy = SequencePolicy.from_dict(summary["eligibility_policy"])
    data = SequenceLimits.from_dict(summary["data_limits"])
    numeric = _base._typed(NeuralNumericLimits, summary["numeric_limits"])
    pooling = _base._typed(AblationPoolingLimits, summary["pooling_limits"])
    training_config = _base._typed(NeuralTrainingConfig, summary["training_config"])
    training_limits = _base._typed(AblationTrainingLimits, summary["training_limits"])
    vocabulary = SequenceVocabulary.from_dict(values["vocabulary"])
    vocabulary.validate_limits(data)
    architecture = config.architecture(vocabulary.size)
    shapes = ablation_parameter_shapes(config.variant, architecture)
    if architecture.parameter_count > numeric.max_parameters:
        raise NeuralArtifactError("canonical main initializer exceeds the saved parameter budget")
    if sum(math.prod(shape) for shape in shapes.values()) > min(
        limits.max_parameters, numeric.max_parameters
    ):
        raise NeuralArtifactError("artifact exceeds the active parameter budget")
    minimum = admit_ablation_turns(
        config.variant,
        architecture,
        tuple((2,) for _ in range(policy.min_turns)),
        limits=numeric,
        pooling_limits=pooling,
    )
    parameters = FrozenAblationParameters(
        config.variant, architecture, _tensors(manifest, members, shapes, limits), numeric, pooling
    )
    train_count, train_prefixes = _base._support(summary, "training", data, numeric, policy)
    val_count, val_prefixes = _base._support(summary, "model_validation", data, numeric, policy)
    policy_count, policy_prefixes = _base._support(
        summary, "policy_validation", data, numeric, policy
    )
    groups = [
        _base._group_set(values[name + "_groups"], data.max_groups, count)
        for name, count in (
            ("training", train_count),
            ("model_validation", val_count),
            ("policy_validation", policy_count),
        )
    ]
    if summary["policy_reuses_model_validation"] is not False:
        raise NeuralArtifactError("ablations require three distinct fitting partitions")
    initial = _base._object(summary["initial_parameter_sha256"], set(shapes))
    main_shapes = parameter_shapes(architecture)
    main_initial = _base._object(summary["main_initial_parameter_sha256"], set(main_shapes))
    _work(
        summary,
        config,
        training_config,
        training_limits,
        policy,
        minimum,
        ((train_count, train_prefixes), (val_count, val_prefixes), (policy_count, policy_prefixes)),
    )
    _base._real(summary["threshold"], 0, 1)
    _base._real(summary["policy_validation_balanced_accuracy"], 0.5, 1)
    result = AblationTrainingResult(
        variant=config.variant,
        parameters=parameters,
        config=training_config,
        training_limits=training_limits,
        history=_base._history(summary, training_config, train_count),
        selected_epoch=summary["selected_epoch"],
        initial_parameter_sha256=tuple((name, _base._digest(initial[name])) for name in shapes),
        main_initial_parameter_sha256=tuple(
            (name, _base._digest(main_initial[name])) for name in main_shapes
        ),
        changed_parameter_names=tuple(
            _base._array(summary["changed_parameter_names"], len(shapes))
        ),
        training_partition_digest=_base._digest(summary["training_partition_digest"]),
        validation_partition_digest=_base._digest(summary["model_validation_partition_digest"]),
        vocabulary_digest=vocabulary.digest,
        training_conversations=train_count,
        validation_conversations=val_count,
        training_prefixes=train_prefixes,
        validation_prefixes=val_prefixes,
        maximum_estimated_workspace_bytes=summary["maximum_estimated_workspace_bytes"],
        initialization_workspace_bytes=summary["initialization_workspace_bytes"],
        estimated_affine_multiplications=summary["estimated_affine_multiplications"],
        estimated_pooling_additions=summary["estimated_pooling_additions"],
        estimated_pooling_scalings=summary["estimated_pooling_scalings"],
        torch_version=summary["torch_version"],
        numpy_version=summary["numpy_version"],
        torch_threads=summary["torch_threads"],
        initialization_version=summary["initialization_version"],
        main_initialization_version=summary["main_initialization_version"],
    )
    state = AblationForecastState(
        config,
        policy,
        data,
        vocabulary,
        result,
        summary["threshold"],
        summary["policy_validation_balanced_accuracy"],
        groups[0],
        groups[1],
        groups[2],
        _base._digest(summary["policy_validation_partition_digest"]),
        policy_count,
        policy_prefixes,
    )
    if _base._canonical(state.summary(), limits) != _base._canonical(
        summary, limits
    ) or state.digest != _base._digest(manifest["model_digest"]):
        raise NeuralArtifactError("restored ablation state or identity differs from its manifest")
    model = AblationEventForecaster(
        config=config,
        training_config=training_config,
        policy=policy,
        data_limits=data,
        numeric_limits=numeric,
        pooling_limits=pooling,
        training_limits=training_limits,
    )
    model._state = state
    return model


def _encode(model: AblationEventForecaster, limits: NeuralArtifactLimits) -> bytes:
    if type(model) is not AblationEventForecaster or type(model.state) is not AblationForecastState:
        raise NeuralArtifactError("save requires a fitted closed ablation forecaster")
    state = model.state
    for value, kind in (
        (state.config, AblationForecastConfig),
        (state.policy, SequencePolicy),
        (state.data_limits, SequenceLimits),
        (state.vocabulary, SequenceVocabulary),
        (state.training, AblationTrainingResult),
    ):
        if type(value) is not kind:
            raise NeuralArtifactError("source model must contain closed typed contracts")
    result = state.training
    if (
        type(result.config) is not NeuralTrainingConfig
        or type(state.parameters) is not FrozenAblationParameters
    ):
        raise NeuralArtifactError("source training state must use closed typed contracts")
    parameters = state.parameters
    shapes = ablation_parameter_shapes(state.config.variant, parameters.architecture)
    if sum(math.prod(shape) for shape in shapes.values()) > limits.max_parameters:
        raise NeuralArtifactError("artifact exceeds the active parameter budget")
    if type(parameters.tensors) is not tuple or len(parameters.tensors) != len(shapes):
        raise NeuralArtifactError("source tensor inventory differs from the ablation")
    tensor_bytes = 0
    for tensor, (name, shape) in zip(parameters.tensors, shapes.items(), strict=True):
        if (
            type(tensor) is not FrozenNeuralTensor
            or tensor.name != name
            or tensor.shape != shape
            or type(tensor.data) is not bytes
            or len(tensor.data) != 4 * math.prod(shape)
        ):
            raise NeuralArtifactError("source tensor differs from the closed descriptor")
        tensor_bytes += len(_base._npy_header(shape)) + len(tensor.data)
    if tensor_bytes > limits.max_file_bytes:
        raise NeuralArtifactError("source tensors exceed the artifact file budget")
    for inventory, maximum in (
        (result.history, result.config.epochs),
        (result.initial_parameter_sha256, len(shapes)),
        (result.main_initial_parameter_sha256, len(parameter_shapes(parameters.architecture))),
        (result.changed_parameter_names, len(shapes)),
    ):
        if type(inventory) is not tuple or len(inventory) > maximum:
            raise NeuralArtifactError("source training inventory is not bounded and immutable")
    if any(type(epoch) is not NeuralEpoch for epoch in result.history):
        raise NeuralArtifactError("source history must contain typed epochs")
    # Validate tuple ordering/duplicates before summary() turns hash pairs into
    # JSON objects, where duplicate names could otherwise be silently collapsed.
    result.__post_init__()
    state.vocabulary.validate_limits(state.data_limits)
    groups = (state.training_groups, state.validation_groups, state.policy_validation_groups)
    if any(
        type(group) is not frozenset or len(group) > state.data_limits.max_groups
        for group in groups
    ):
        raise NeuralArtifactError("source group inventory is not bounded and immutable")
    if result.vocabulary_digest != state.vocabulary.digest:
        raise NeuralArtifactError("source vocabulary and training identity differ")
    descriptors = []
    members = []
    for tensor in parameters.tensors:
        name = f"tensors/{tensor.name}.npy"
        payload = _base._npy_header(tensor.shape) + tensor.data
        members.append((name, payload))
        descriptors.append(
            {
                "name": tensor.name,
                "member": name,
                "dtype": "<f4",
                "shape": list(tensor.shape),
                "data_bytes": len(tensor.data),
                "data_sha256": _base._sha(tensor.data),
                "member_sha256": _base._sha(payload),
            }
        )
    manifest = {
        "format": ABLATION_ARTIFACT_FORMAT,
        "numerical_version": ablation_numerical_version(state.config.variant),
        "training_version": ABLATION_TRAINING_VERSION,
        "initialization_version": ABLATION_INITIALIZATION_VERSION,
        "model_digest": state.digest,
        "state": {
            "summary": state.summary(),
            "vocabulary": state.vocabulary.to_dict(),
            "training_groups": sorted(groups[0]),
            "model_validation_groups": sorted(groups[1]),
            "policy_validation_groups": sorted(groups[2]),
        },
        "tensors": descriptors,
    }
    manifest["sha256"] = _base._sha(_base._canonical(manifest, limits))
    encoded = _base._canonical(manifest, limits)
    raw = _base._archive([("manifest.json", encoded), *members], limits)
    _restore(_base._parse_manifest(encoded, limits), _base._members(raw, limits), limits)
    return raw


def load_ablation_forecaster(
    path: str | os.PathLike[str], *, limits: NeuralArtifactLimits | None = None
) -> AblationEventForecaster:
    """Restore validated inference data, without Torch, extraction or generic loaders."""
    selected = _base._limits(limits)
    members = _base._members(_base._read(Path(path), selected), selected)
    try:
        return _restore(
            _base._parse_manifest(members["manifest.json"], selected), members, selected
        )
    except NeuralArtifactError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
        raise NeuralArtifactError("invalid ablation artifact contract") from error


def save_ablation_forecaster(
    model: AblationEventForecaster, path: str | os.PathLike[str], *, overwrite: bool = False
) -> NeuralSaveResult:
    """Validate before exclusive publication; overwrite requires an explicit opt-in."""
    if type(overwrite) is not bool:
        raise NeuralArtifactError("overwrite must be an explicit boolean")
    target = Path(os.path.abspath(path))
    previous = _base._target(target)
    if previous is not None and not overwrite:
        raise FileExistsError("artifact output already exists")
    try:
        raw = _encode(model, NeuralArtifactLimits())
    except NeuralArtifactError:
        raise
    except (
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        OverflowError,
        RecursionError,
    ) as error:
        raise NeuralArtifactError("invalid source ablation model state") from error
    temporary: Path | None = None
    published = False
    warning = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=".turnscope-ablation-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            if stream.write(raw) != len(raw):
                raise OSError("artifact temporary file was not completely written")
            stream.flush()
            os.fsync(stream.fileno())
        current = _base._target(target)
        if overwrite:
            if (previous is None) != (current is None) or (
                previous is not None
                and current is not None
                and _base._identity(previous) != _base._identity(current)
            ):
                raise NeuralArtifactError("artifact output changed before replacement")
            os.replace(temporary, target)
            temporary = None
        else:
            os.link(temporary, target)
        published = True
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                if published:
                    warning = (
                        "Artifact was published; a private temporary file could not be removed."
                    )
    return NeuralSaveResult(str(target), _base._sha(raw), len(raw), warning)
