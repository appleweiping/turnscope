"""Closed, deterministic, NumPy-only inference artifacts (never optimizer resume)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
import tempfile
import zlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .neural_forecast import (
    HierarchicalEventForecaster,
    NeuralForecastConfig,
    NeuralForecastState,
)
from .neural_forecast_data import SequenceLimits, SequencePolicy
from .neural_forecast_math import (
    NUMERICAL_VERSION,
    FrozenNeuralParameters,
    FrozenNeuralTensor,
    NeuralNumericLimits,
    admit_encoded_turns,
    parameter_shapes,
)
from .neural_forecast_train import (
    INITIALIZATION_VERSION,
    TRAINING_VERSION,
    NeuralEpoch,
    NeuralTrainingConfig,
    NeuralTrainingResult,
)
from .neural_token_data import SequenceVocabulary

ARTIFACT_FORMAT = "turnscope.neural-forecast-artifact.v1"
_MIB = 1024 * 1024
_LOCAL = struct.Struct("<4s5H3L2H")
_CENTRAL = struct.Struct("<4s6H3L5H2L")
_END = struct.Struct("<4s4H2LH")
_ATTR = (stat.S_IFREG | 0o600) << 16
_SUMMARY_FIELDS = {
    "format",
    "config",
    "eligibility_policy",
    "data_limits",
    "numeric_limits",
    "training_config",
    "parameter_digest",
    "vocabulary_digest",
    "vocabulary_size",
    "parameter_count",
    "training_partition_digest",
    "model_validation_partition_digest",
    "policy_validation_partition_digest",
    "training_conversations",
    "model_validation_conversations",
    "policy_validation_conversations",
    "training_prefixes",
    "model_validation_prefixes",
    "policy_validation_prefixes",
    "policy_reuses_model_validation",
    "selected_epoch",
    "history",
    "initial_parameter_sha256",
    "changed_parameter_names",
    "maximum_estimated_workspace_bytes",
    "estimated_affine_multiplications",
    "torch_version",
    "numpy_version",
    "torch_threads",
    "threshold",
    "threshold_rule",
    "threshold_objective",
    "policy_validation_balanced_accuracy",
    "probability_floor",
    "probability_calibration_claimed",
}


class NeuralArtifactError(ValueError):
    """Malformed, inconsistent, or over-budget inference artifact."""


def _integer(value: Any, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise NeuralArtifactError("artifact integer is outside its declared range")
    return value


def _real(value: Any, minimum: float, maximum: float) -> float:
    if (
        type(value) not in (int, float)
        or not minimum <= value <= maximum
        or not math.isfinite(value)
    ):
        raise NeuralArtifactError("artifact number must be finite and in range")
    return float(value)


@dataclass(frozen=True, slots=True)
class NeuralArtifactLimits:
    """Admission limits, not changes to a saved model's numerical policy."""

    max_file_bytes: int = 64 * _MIB
    max_manifest_bytes: int = 4 * _MIB
    max_manifest_nodes: int = 200_000
    max_manifest_depth: int = 16
    max_parameters: int = 16_000_000
    max_parameter_magnitude: float = 1e6

    def __post_init__(self) -> None:
        for name, ceiling in (
            ("max_file_bytes", 64 * _MIB),
            ("max_manifest_bytes", 4 * _MIB),
            ("max_manifest_nodes", 200_000),
            ("max_manifest_depth", 16),
            ("max_parameters", 16_000_000),
        ):
            _integer(getattr(self, name), 1, ceiling)
        magnitude = _real(self.max_parameter_magnitude, 0.0, 1e6)
        if magnitude == 0:
            raise NeuralArtifactError("artifact parameter magnitude limit must be positive")


@dataclass(frozen=True, slots=True)
class NeuralSaveResult:
    path: str
    sha256: str
    bytes_written: int
    cleanup_warning: str | None = None


def _limits(limits: NeuralArtifactLimits | None) -> NeuralArtifactLimits:
    if limits is None:
        return NeuralArtifactLimits()
    if type(limits) is not NeuralArtifactLimits:
        raise NeuralArtifactError("limits must be a NeuralArtifactLimits instance")
    return limits


def _sha(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: Any) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise NeuralArtifactError("artifact digest must be lowercase SHA-256")
    return value


def _object(value: Any, names: set[str]) -> dict[str, Any]:
    if type(value) is not dict or len(value) != len(names) or value.keys() != names:
        raise NeuralArtifactError("artifact object has missing or unknown fields")
    return value


def _array(value: Any, maximum: int, *, minimum: int = 0) -> list[Any]:
    if type(value) is not list or not minimum <= len(value) <= maximum:
        raise NeuralArtifactError("artifact array has invalid length or type")
    return value


def _string_bytes(value: str, maximum: int) -> int:
    size = 2
    for character in value:
        code = ord(character)
        if 0xD800 <= code <= 0xDFFF:
            raise NeuralArtifactError("artifact strings cannot contain surrogate code points")
        if character in '"\\\b\f\n\r\t':
            size += 2
        elif code < 32:
            size += 6
        else:
            size += 1 if code < 128 else 2 if code < 2048 else 3 if code < 65536 else 4
        if size > maximum:
            raise NeuralArtifactError("manifest exceeds its encoded byte budget")
    return size


def _canonical(value: Any, limits: NeuralArtifactLimits) -> bytes:
    """Count the entire pending tree and encoded size before JSON allocation."""
    stack = [(value, 1)]
    nodes = size = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > limits.max_manifest_nodes or depth > limits.max_manifest_depth:
            raise NeuralArtifactError("manifest exceeds its node or depth budget")
        kind = type(item)
        if kind is dict:
            pending = 2 * len(item)
            if pending > limits.max_manifest_nodes - nodes - len(stack):
                raise NeuralArtifactError("manifest exceeds its node budget")
            size += 2 + max(0, len(item) - 1) + len(item)
            for key, child in item.items():
                if type(key) is not str:
                    raise NeuralArtifactError("manifest keys must be strings")
                stack.append((key, depth + 1))
                stack.append((child, depth + 1))
        elif kind in (list, tuple):
            if len(item) > limits.max_manifest_nodes - nodes - len(stack):
                raise NeuralArtifactError("manifest exceeds its node budget")
            size += 2 + max(0, len(item) - 1)
            stack.extend((child, depth + 1) for child in item)
        elif kind is str:
            size += _string_bytes(item, limits.max_manifest_bytes - size)
        elif kind is bool:
            size += 4 if item else 5
        elif item is None:
            size += 4
        elif kind is int:
            _integer(item, -(2**63 - 1), 2**63 - 1)
            size += len(str(item))
        elif kind is float:
            if not math.isfinite(item):
                raise NeuralArtifactError("manifest numbers must be finite")
            size += len(repr(item))
        else:
            raise NeuralArtifactError("manifest must contain only closed JSON values")
        if size > limits.max_manifest_bytes:
            raise NeuralArtifactError("manifest exceeds its encoded byte budget")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _lexical_admission(text: str, limits: NeuralArtifactLimits) -> None:
    # Count containers/scalars/keys before json.loads builds a tree. Full JSON
    # grammar validation remains the decoder's job; no regular-expression parser.
    depth = nodes = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in " \t\r\n,:]}":
            if char in "]}":
                depth -= 1
            index += 1
            continue
        nodes += 1
        if nodes > limits.max_manifest_nodes:
            raise NeuralArtifactError("manifest exceeds its node budget")
        if char in "[{":
            depth += 1
            if depth > limits.max_manifest_depth:
                raise NeuralArtifactError("manifest exceeds its depth budget")
            index += 1
        elif char == '"':
            if depth + 1 > limits.max_manifest_depth:
                raise NeuralArtifactError("manifest exceeds its depth budget")
            index += 1
            while index < len(text):
                if text[index] == "\\":
                    index += 2
                elif text[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
        else:
            if depth + 1 > limits.max_manifest_depth:
                raise NeuralArtifactError("manifest exceeds its depth budget")
            start = index
            while index < len(text) and text[index] not in " \t\r\n,:[]{}":
                index += 1
            if index - start > 128:
                raise NeuralArtifactError("manifest numeric token is too long")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise NeuralArtifactError("manifest has duplicate keys")
        result[name] = value
    return result


def _parse_integer(text: str) -> int:
    if len(text) > 20:
        raise NeuralArtifactError("manifest integer is too long")
    return _integer(int(text), -(2**63 - 1), 2**63 - 1)


def _nonfinite(text: str) -> Any:
    raise NeuralArtifactError("manifest numbers must be finite")


def _parse_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise NeuralArtifactError("manifest numbers must be finite")
    return value


def _parse_manifest(raw: bytes | memoryview, limits: NeuralArtifactLimits) -> dict[str, Any]:
    if len(raw) > limits.max_manifest_bytes:
        raise NeuralArtifactError("manifest exceeds its byte budget")
    try:
        text = bytes(raw).decode("utf-8")
        _lexical_admission(text, limits)
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_int=_parse_integer,
            parse_float=_parse_float,
            parse_constant=_nonfinite,
        )
        if _canonical(value, limits) != raw:
            raise NeuralArtifactError("manifest must use canonical UTF-8 JSON encoding")
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise NeuralArtifactError("invalid manifest JSON") from error
    return _object(
        value,
        {
            "format",
            "numerical_version",
            "training_version",
            "initialization_version",
            "model_digest",
            "state",
            "tensors",
            "sha256",
        },
    )


def _npy_header(shape: tuple[int, ...]) -> bytes:
    text = "{'descr': '<f4', 'fortran_order': False, 'shape': " + str(shape) + ", }"
    header = text.encode("ascii")
    header += b" " * ((-10 - len(header) - 1) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header


def _local(name: bytes, data: bytes | memoryview) -> bytes:
    return (
        _LOCAL.pack(
            b"PK\x03\x04", 20, 0, 0, 0, 33, zlib.crc32(data), len(data), len(data), len(name), 0
        )
        + name
    )


def _central(name: bytes, data: bytes | memoryview, offset: int) -> bytes:
    return (
        _CENTRAL.pack(
            b"PK\x01\x02",
            788,
            20,
            0,
            0,
            0,
            33,
            zlib.crc32(data),
            len(data),
            len(data),
            len(name),
            0,
            0,
            0,
            0,
            _ATTR,
            offset,
        )
        + name
    )


def _archive(members: list[tuple[str, bytes]], limits: NeuralArtifactLimits) -> bytes:
    if not 1 <= len(members) <= 32:
        raise NeuralArtifactError("archive member count exceeds budget")
    total = _END.size + sum(
        _LOCAL.size + _CENTRAL.size + 2 * len(name) + len(data) for name, data in members
    )
    if total > limits.max_file_bytes:
        raise NeuralArtifactError("archive exceeds its file byte budget")
    local: list[bytes] = []
    central: list[bytes] = []
    offset = 0
    for name, data in members:
        encoded = name.encode("ascii")
        header = _local(encoded, data)
        central.append(_central(encoded, data, offset))
        local.extend((header, data))
        offset += len(header) + len(data)
    directory = b"".join(central)
    return b"".join(
        (
            *local,
            directory,
            _END.pack(b"PK\x05\x06", 0, 0, len(members), len(members), len(directory), offset, 0),
        )
    )


def _members(raw: bytes, limits: NeuralArtifactLimits) -> dict[str, memoryview]:
    if not _END.size <= len(raw) <= limits.max_file_bytes:
        raise NeuralArtifactError("archive is truncated or over budget")
    end = _END.unpack_from(raw, len(raw) - _END.size)
    _, disk, start_disk, count, total_count, directory_bytes, directory_start, comment = end
    if (
        end[0] != b"PK\x05\x06"
        or disk
        or start_disk
        or comment
        or not 1 <= count <= 32
        or count != total_count
        or directory_start + directory_bytes != len(raw) - _END.size
    ):
        raise NeuralArtifactError("archive end record is not canonical")
    cursor, expected_offset = directory_start, 0
    result: dict[str, memoryview] = {}
    for _ in range(count):
        if cursor + _CENTRAL.size > len(raw) - _END.size:
            raise NeuralArtifactError("truncated archive directory")
        record = _CENTRAL.unpack_from(raw, cursor)
        size, name_length, offset = record[9], record[10], record[16]
        if (
            not 1 <= name_length <= 128
            or cursor + _CENTRAL.size + name_length > len(raw) - _END.size
        ):
            raise NeuralArtifactError("invalid archive member name")
        name_bytes = raw[cursor + _CENTRAL.size : cursor + _CENTRAL.size + name_length]
        try:
            name = name_bytes.decode("ascii")
        except UnicodeError as error:
            raise NeuralArtifactError("archive names must be ASCII") from error
        if (
            name in result
            or name.startswith("/")
            or ".." in name
            or "\\" in name
            or ":" in name
            or offset != expected_offset
        ):
            raise NeuralArtifactError("archive contains duplicate, unsafe, or overlapping members")
        data_start = offset + _LOCAL.size + name_length
        data_end = data_start + size
        if data_end > directory_start:
            raise NeuralArtifactError("archive member overlaps the directory")
        data = memoryview(raw)[data_start:data_end]
        # Byte-for-byte headers enforce stored/regular/no flags, extras, comments,
        # descriptor, links, traversal, alternate names, or hidden local members.
        header = _local(name_bytes, data)
        directory = _central(name_bytes, data, offset)
        if raw[offset:data_start] != header or raw[cursor : cursor + len(directory)] != directory:
            raise NeuralArtifactError("archive member metadata, size, or CRC is not canonical")
        result[name] = data
        expected_offset = data_end
        cursor += len(directory)
    if expected_offset != directory_start or cursor != len(raw) - _END.size:
        raise NeuralArtifactError("archive contains unindexed bytes")
    if next(iter(result)) != "manifest.json":
        raise NeuralArtifactError("archive must start with its manifest")
    return result


def _typed(cls: Any, value: Any) -> Any:
    obj = _object(value, {item.name for item in fields(cls) if item.init})
    return cls(**obj)


def _group_set(value: Any, maximum: int, conversations: int) -> frozenset[str]:
    items = _array(value, maximum, minimum=conversations)
    for digest in items:
        _digest(digest)
    if items != sorted(set(items)):
        raise NeuralArtifactError("group digest inventory must be sorted and unique")
    return frozenset(items)


def _history(
    summary: dict[str, Any], config: NeuralTrainingConfig, conversations: int
) -> tuple[NeuralEpoch, ...]:
    records = _array(summary["history"], config.epochs, minimum=1)
    history = []
    best_loss, best_epoch, stale = math.inf, 0, 0
    minimum_steps = (conversations + config.batch_conversations - 1) // config.batch_conversations
    for index, value in enumerate(records, 1):
        if stale >= config.patience:
            raise NeuralArtifactError("training history continues after early stopping")
        record = _object(value, {item.name for item in fields(NeuralEpoch)})
        if _integer(record["epoch"], 1, config.epochs) != index:
            raise NeuralArtifactError("training epochs must be consecutive")
        for key in ("training_loss", "validation_loss", "maximum_unclipped_gradient_norm"):
            _real(record[key], 0, 1e300)
        _integer(record["optimizer_steps"], minimum_steps, conversations)
        history.append(NeuralEpoch(**record))
        if record["validation_loss"] < best_loss:
            best_loss, best_epoch, stale = record["validation_loss"], index, 0
        else:
            stale += 1
    if len(records) < config.epochs and stale < config.patience:
        raise NeuralArtifactError("training history ends before its declared stopping policy")
    if _integer(summary["selected_epoch"], 1, len(records)) != best_epoch:
        raise NeuralArtifactError("selected epoch must be the earliest minimum validation loss")
    return tuple(history)


def _support(
    summary: dict[str, Any],
    name: str,
    data: SequenceLimits,
    numeric: NeuralNumericLimits,
    policy: SequencePolicy,
) -> tuple[int, int]:
    conversations = _integer(summary[name + "_conversations"], 2, data.max_conversations)
    per_conversation = (
        min(data.max_observed_turns, numeric.max_observed_turns) - policy.min_turns + 1
    )
    prefixes = _integer(
        summary[name + "_prefixes"],
        conversations,
        min(data.max_prefixes, conversations * per_conversation),
    )
    if prefixes + conversations * (policy.min_turns - 1) > data.max_source_turns:
        raise NeuralArtifactError("declared support exceeds the source-turn budget")
    return conversations, prefixes


def _restore(
    manifest: dict[str, Any], members: dict[str, memoryview], limits: NeuralArtifactLimits
) -> HierarchicalEventForecaster:
    claimed_hash = _digest(manifest["sha256"])
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    if claimed_hash != _sha(_canonical(body, limits)):
        raise NeuralArtifactError("manifest integrity digest differs")
    for key, expected in (
        ("format", ARTIFACT_FORMAT),
        ("numerical_version", NUMERICAL_VERSION),
        ("training_version", TRAINING_VERSION),
        ("initialization_version", INITIALIZATION_VERSION),
    ):
        if manifest[key] != expected:
            raise NeuralArtifactError("unsupported neural artifact version")
    state_data = _object(
        manifest["state"],
        {
            "summary",
            "vocabulary",
            "training_groups",
            "model_validation_groups",
            "policy_validation_groups",
        },
    )
    summary = _object(state_data["summary"], _SUMMARY_FIELDS)
    config = _typed(NeuralForecastConfig, summary["config"])
    policy = SequencePolicy.from_dict(summary["eligibility_policy"])
    data = SequenceLimits.from_dict(summary["data_limits"])
    numeric = _typed(NeuralNumericLimits, summary["numeric_limits"])
    training_config = _typed(NeuralTrainingConfig, summary["training_config"])
    vocabulary = SequenceVocabulary.from_dict(state_data["vocabulary"])
    vocabulary.validate_limits(data)
    if (
        vocabulary.max_turn_tokens != config.max_turn_tokens
        or vocabulary.long_turn_policy != config.long_turn_policy
        or len(vocabulary.tokens) > training_config.max_features
        or any(
            frequency < training_config.min_document_frequency
            for frequency in vocabulary.document_frequencies
        )
    ):
        raise NeuralArtifactError(
            "vocabulary differs from the model's declared training/token policy"
        )
    architecture = config.architecture(vocabulary.size)
    if architecture.parameter_count > min(limits.max_parameters, numeric.max_parameters):
        raise NeuralArtifactError("artifact exceeds the parameter budget")
    minimum_inference = admit_encoded_turns(
        architecture, tuple((2,) for _ in range(policy.min_turns)), limits=numeric
    )
    shapes = parameter_shapes(architecture)
    descriptors = _array(manifest["tensors"], len(shapes), minimum=len(shapes))
    expected_members = ["manifest.json", *(f"tensors/{name}.npy" for name in shapes)]
    if list(members) != expected_members:
        raise NeuralArtifactError("archive member inventory differs from the architecture")
    tensors = []
    for (name, shape), descriptor in zip(shapes.items(), descriptors, strict=True):
        item = _object(
            descriptor,
            {"name", "member", "dtype", "shape", "data_bytes", "data_sha256", "member_sha256"},
        )
        member = f"tensors/{name}.npy"
        if (
            item["name"] != name
            or item["member"] != member
            or item["dtype"] != "<f4"
            or _canonical(item["shape"], limits) != _canonical(list(shape), limits)
            or _integer(item["data_bytes"], 4, 4 * limits.max_parameters) != 4 * math.prod(shape)
        ):
            raise NeuralArtifactError("tensor descriptor differs from the derived shape")
        payload = members[member]
        header = _npy_header(shape)
        if len(payload) != len(header) + item["data_bytes"] or payload[: len(header)] != header:
            raise NeuralArtifactError("tensor must have the exact canonical NPY 1.0 header")
        raw = payload[len(header) :]
        if _sha(raw) != _digest(item["data_sha256"]) or _sha(payload) != _digest(
            item["member_sha256"]
        ):
            raise NeuralArtifactError("tensor integrity digest differs")
        # Check the additional caller cap without replacing the persisted limits.
        if limits.max_parameter_magnitude < numeric.max_parameter_magnitude:
            for (value,) in struct.iter_unpack("<f", raw):
                if not math.isfinite(value) or abs(value) > limits.max_parameter_magnitude:
                    raise NeuralArtifactError("tensor exceeds the artifact magnitude budget")
        tensors.append(FrozenNeuralTensor(name, shape, bytes(raw)))
    parameters = FrozenNeuralParameters(architecture, tuple(tensors), numeric)
    train_conversations, train_prefixes = _support(summary, "training", data, numeric, policy)
    val_conversations, val_prefixes = _support(summary, "model_validation", data, numeric, policy)
    policy_conversations, policy_prefixes = _support(
        summary, "policy_validation", data, numeric, policy
    )
    if vocabulary.documents != train_conversations:
        raise NeuralArtifactError("vocabulary document count differs from training support")
    train_groups = _group_set(state_data["training_groups"], data.max_groups, train_conversations)
    val_groups = _group_set(
        state_data["model_validation_groups"], data.max_groups, val_conversations
    )
    policy_groups = _group_set(
        state_data["policy_validation_groups"], data.max_groups, policy_conversations
    )
    train_digest = _digest(summary["training_partition_digest"])
    val_digest = _digest(summary["model_validation_partition_digest"])
    policy_digest = _digest(summary["policy_validation_partition_digest"])
    reuse = summary["policy_reuses_model_validation"]
    if type(reuse) is not bool or train_groups & val_groups or train_digest == val_digest:
        raise NeuralArtifactError("training and validation partition declarations conflict")
    if reuse:
        if (
            policy_groups != val_groups
            or policy_digest != val_digest
            or (policy_conversations, policy_prefixes) != (val_conversations, val_prefixes)
        ):
            raise NeuralArtifactError(
                "reused policy validation must exactly match model validation"
            )
    elif policy_groups & (train_groups | val_groups) or policy_digest in (train_digest, val_digest):
        raise NeuralArtifactError("independent policy validation overlaps fitting declarations")
    history = _history(summary, training_config, train_conversations)
    initial = _object(summary["initial_parameter_sha256"], set(shapes))
    initial_pairs = tuple((name, _digest(initial[name])) for name in shapes)
    changed = tuple(_array(summary["changed_parameter_names"], len(shapes)))
    expected_changed = tuple(
        tensor.name for tensor in parameters.tensors if _sha(tensor.data) != initial[tensor.name]
    )
    if changed != expected_changed:
        raise NeuralArtifactError("changed tensor inventory differs from initial/selected hashes")
    workspace = _integer(
        summary["maximum_estimated_workspace_bytes"],
        48 * architecture.parameter_count,
        training_config.max_workspace_bytes,
    )
    work = _integer(
        summary["estimated_affine_multiplications"],
        1,
        training_config.max_total_affine_multiplications,
    )
    if work % training_config.epochs:
        raise NeuralArtifactError("training work estimate must cover all configured epochs")
    if (
        work
        < (3 * train_conversations + val_conversations)
        * training_config.epochs
        * minimum_inference.affine_multiplications
    ):
        raise NeuralArtifactError("training work estimate is below its minimum declared support")
    if (
        config.max_inference_affine_multiplications
        < policy_conversations * minimum_inference.affine_multiplications
    ):
        raise NeuralArtifactError("policy validation support exceeds its inference work budget")
    for key in ("torch_version", "numpy_version"):
        value = summary[key]
        if (
            type(value) is not str
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}", value) is None
        ):
            raise NeuralArtifactError("invalid dependency version declaration")
    threads = _integer(summary["torch_threads"], 1, 65536)
    _real(summary["threshold"], 0, 1)
    _real(summary["policy_validation_balanced_accuracy"], 0.5, 1)
    result = NeuralTrainingResult(
        parameters,
        training_config,
        history,
        summary["selected_epoch"],
        initial_pairs,
        changed,
        train_digest,
        val_digest,
        vocabulary.digest,
        train_conversations,
        val_conversations,
        train_prefixes,
        val_prefixes,
        workspace,
        work,
        summary["torch_version"],
        summary["numpy_version"],
        threads,
    )
    state = NeuralForecastState(
        config,
        policy,
        data,
        vocabulary,
        result,
        summary["threshold"],
        summary["policy_validation_balanced_accuracy"],
        train_groups,
        val_groups,
        policy_groups,
        policy_digest,
        policy_conversations,
        policy_prefixes,
        reuse,
    )
    if _canonical(state.summary(), limits) != _canonical(
        summary, limits
    ) or state.digest != _digest(manifest["model_digest"]):
        raise NeuralArtifactError("restored model state or identity differs from its manifest")
    model = HierarchicalEventForecaster(
        config=config,
        training_config=training_config,
        policy=policy,
        data_limits=data,
        numeric_limits=numeric,
    )
    model._state = state
    return model


def _encode(model: HierarchicalEventForecaster, limits: NeuralArtifactLimits) -> bytes:
    if (
        type(model) is not HierarchicalEventForecaster
        or type(model.state) is not NeuralForecastState
    ):
        raise NeuralArtifactError("save requires a fitted closed neural forecaster")
    state = model.state
    for value, cls in (
        (state.config, NeuralForecastConfig),
        (state.policy, SequencePolicy),
        (state.data_limits, SequenceLimits),
        (state.vocabulary, SequenceVocabulary),
        (state.training, NeuralTrainingResult),
    ):
        if type(value) is not cls:
            raise NeuralArtifactError("model state must use closed typed contracts")
    result = state.training
    if (
        type(result.config) is not NeuralTrainingConfig
        or type(state.parameters) is not FrozenNeuralParameters
    ):
        raise NeuralArtifactError("model state must use closed typed training contracts")
    if (
        type(result.history) is not tuple
        or not 1 <= len(result.history) <= result.config.epochs
        or any(type(item) is not NeuralEpoch for item in result.history)
        or type(result.initial_parameter_sha256) is not tuple
        or len(result.initial_parameter_sha256) != len(state.parameters.tensors)
        or any(
            type(item) is not tuple or len(item) != 2 for item in result.initial_parameter_sha256
        )
        or tuple(item[0] for item in result.initial_parameter_sha256)
        != tuple(parameter_shapes(state.parameters.architecture))
        or type(result.changed_parameter_names) is not tuple
        or any(
            type(value) is not frozenset
            for value in (
                state.training_groups,
                state.validation_groups,
                state.policy_validation_groups,
            )
        )
        or any(
            len(value) > state.data_limits.max_groups
            for value in (
                state.training_groups,
                state.validation_groups,
                state.policy_validation_groups,
            )
        )
        or result.vocabulary_digest != state.vocabulary.digest
    ):
        raise NeuralArtifactError("source model contains inconsistent typed training state")
    if state.parameters.architecture.parameter_count > limits.max_parameters:
        raise NeuralArtifactError("artifact exceeds the parameter budget")
    members = []
    descriptors = []
    total_tensor_bytes = sum(
        len(_npy_header(tensor.shape)) + len(tensor.data) for tensor in state.parameters.tensors
    )
    if total_tensor_bytes > limits.max_file_bytes:
        raise NeuralArtifactError("tensor members exceed the artifact byte budget")
    for tensor in state.parameters.tensors:
        name = f"tensors/{tensor.name}.npy"
        header = _npy_header(tensor.shape)
        payload = header + tensor.data
        members.append((name, payload))
        descriptors.append(
            {
                "name": tensor.name,
                "member": name,
                "dtype": "<f4",
                "shape": list(tensor.shape),
                "data_bytes": len(tensor.data),
                "data_sha256": _sha(tensor.data),
                "member_sha256": _sha(payload),
            }
        )
    manifest = {
        "format": ARTIFACT_FORMAT,
        "numerical_version": NUMERICAL_VERSION,
        "training_version": TRAINING_VERSION,
        "initialization_version": INITIALIZATION_VERSION,
        "model_digest": state.digest,
        "state": {
            "summary": state.summary(),
            "vocabulary": state.vocabulary.to_dict(),
            "training_groups": sorted(state.training_groups),
            "model_validation_groups": sorted(state.validation_groups),
            "policy_validation_groups": sorted(state.policy_validation_groups),
        },
        "tensors": descriptors,
    }
    manifest["sha256"] = _sha(_canonical(manifest, limits))
    raw_manifest = _canonical(manifest, limits)
    raw = _archive([("manifest.json", raw_manifest), *members], limits)
    # Saving arbitrary manually assembled state must obey the same semantic
    # validator as loading. Never publish first and discover inconsistency later.
    _restore(_parse_manifest(raw_manifest, limits), _members(raw, limits), limits)
    return raw


def _identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _read(path: Path, limits: NeuralArtifactLimits) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limits.max_file_bytes:
        raise NeuralArtifactError("artifact must be a bounded regular file, not a symlink")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(before):
            raise NeuralArtifactError("artifact changed while being opened")
        raw = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
    current = path.lstat()
    if (
        len(raw) != before.st_size
        or _identity(after) != _identity(before)
        or _identity(current) != _identity(before)
        or not stat.S_ISREG(current.st_mode)
    ):
        raise NeuralArtifactError("artifact changed while being read")
    return raw


def load_neural_forecaster(
    path: str | os.PathLike[str], *, limits: NeuralArtifactLimits | None = None
) -> HierarchicalEventForecaster:
    """Read only fixed inference data; never extract files or deserialize code."""
    settings = _limits(limits)
    members = _members(_read(Path(path), settings), settings)
    try:
        return _restore(_parse_manifest(members["manifest.json"], settings), members, settings)
    except NeuralArtifactError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
        raise NeuralArtifactError("invalid neural artifact contract") from error


def _target(path: Path) -> os.stat_result | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise NeuralArtifactError("artifact output cannot be a symlink or non-regular file")
    return info


def save_neural_forecaster(
    model: HierarchicalEventForecaster, path: str | os.PathLike[str], *, overwrite: bool = False
) -> NeuralSaveResult:
    """Publish a validated artifact atomically, exclusively unless explicitly replaced."""
    if type(overwrite) is not bool:
        raise NeuralArtifactError("overwrite must be an explicit boolean")
    target = Path(os.path.abspath(path))
    previous = _target(target)
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
        raise NeuralArtifactError("invalid source neural model state") from error
    temporary: Path | None = None
    published = False
    warning = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=".turnscope-neural-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            if stream.write(raw) != len(raw):
                raise OSError("artifact temporary file was not completely written")
            stream.flush()
            os.fsync(stream.fileno())
        current = _target(target)
        if overwrite:
            if (previous is None) != (current is None) or (
                previous is not None
                and current is not None
                and _identity(previous) != _identity(current)
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
                # Before publication retain the original failure, not a cleanup
                # error that misleadingly changes the operation's outcome.
    return NeuralSaveResult(str(target), _sha(raw), len(raw), warning)
