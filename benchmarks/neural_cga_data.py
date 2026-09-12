"""Read-only pinned CGA-WIKI ingestion and label-independent neural partitions.

This benchmark adapter never trains, downloads, extracts, or interprets parses.
The returned conversation records and any derived models must remain private.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import struct
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from turnscope.models import Conversation, Utterance
from turnscope.neural_forecast_data import (
    SequenceLimits,
    SequencePolicy,
    prepare_sequence_forecasts,
)

ARCHIVE_SHA256 = "84e2d1ac60a3269251b5e175e549fc65cec875fdc926f99172b5b70d3ca1b122"
SEED = "turnscope-neural-v1/2026-09-12"
PREFIX = "conversations-gone-awry-corpus/"
_MIB = 1024 * 1024
_END = struct.Struct("<4s4H2LH")
_METADATA_FIELDS = {
    "page_title",
    "page_id",
    "pair_id",
    "conversation_has_personal_attack",
    "verified",
    "pair_verified",
    "annotation_year",
    "split",
}
_ROW_FIELDS = {"id", "conversation_id", "text", "speaker", "meta", "reply-to", "timestamp"}
_META_REQUIRED = {"is_section_header", "comment_has_personal_attack", "parsed"}
_META_ALLOWED = _META_REQUIRED | {"toxicity"}
_SPLITS = ("train", "val", "test")
_OUTPUTS = ("training", "validation", "policy_validation", "test")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def protocol() -> dict[str, Any]:
    """Return a fresh exact protocol declaration, without source identifiers."""
    return {
        "format": "turnscope.neural-cga-partition.v1",
        "archive_sha256": ARCHIVE_SHA256,
        "seed": SEED,
        "official_training": "train",
        "official_test": "test",
        "validation_components": (
            "shared page OR reciprocal pair across every official val conversation"
        ),
        "component_digest": (
            "SHA256(UTF8 compact sorted-key JSON {seed,conversation_ids:sorted(ids)})"
        ),
        "bucket": "int(component_digest,16) % 5",
        "policy_validation_buckets": [0, 1],
        "model_validation_buckets": [2, 3, 4],
        "eligibility_before_partitioning": False,
        "label_dependent_retries": False,
        "ordering": "ascending finite UTC timestamp, then exact utterance ID",
        "event": "human comment_has_personal_attack on non-header utterances",
        "header": "human is_section_header excluded by SequencePolicy",
        "reply_edges": "ignored; invalid pointers never remove an utterance",
        "features": "text only; no speaker, toxicity, parse, page title, or label features",
    }


def protocol_digest() -> str:
    return _digest(protocol())


@dataclass(frozen=True, slots=True)
class NeuralCgaLimits:
    max_archive_bytes: int = 128 * _MIB
    max_members: int = 64
    max_directory_bytes: int = 64 * 1024
    max_metadata_bytes: int = 4 * _MIB
    max_utterance_bytes: int = 256 * _MIB
    max_row_bytes: int = 4 * _MIB
    max_conversations: int = 10_000
    max_utterances: int = 50_000
    max_json_nodes: int = 1_000_000
    max_json_depth: int = 32
    max_object_fields: int = 64
    max_identifier_bytes: int = 1024
    max_text_bytes: int = _MIB
    max_total_text_bytes: int = 32 * _MIB

    def __post_init__(self) -> None:
        maxima = (
            128 * _MIB,
            64,
            64 * 1024,
            4 * _MIB,
            256 * _MIB,
            4 * _MIB,
            10_000,
            50_000,
            1_000_000,
            32,
            64,
            1024,
            _MIB,
            32 * _MIB,
        )
        for (name, value), maximum in zip(asdict(self).items(), maxima, strict=True):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"invalid CGA limit: {name}")


def _limits(value: NeuralCgaLimits | None) -> NeuralCgaLimits:
    if value is None:
        return NeuralCgaLimits()
    if type(value) is not NeuralCgaLimits:
        raise ValueError("CGA limits must use the closed typed contract")
    return value


@dataclass(frozen=True, slots=True)
class NeuralCgaPartitions:
    training: tuple[Conversation, ...]
    validation: tuple[Conversation, ...]
    policy_validation: tuple[Conversation, ...]
    test: tuple[Conversation, ...]
    _audit_json: bytes = field(repr=False)

    @property
    def audit(self) -> dict[str, Any]:
        return self.audit_dict()

    def audit_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = json.loads(self._audit_json)
        return value

    def prepared_audit(
        self, *, policy: SequencePolicy | None = None, limits: SequenceLimits | None = None
    ) -> dict[str, Any]:
        """Prepare only immutable causal handles; no vocabulary fitting or model calls."""
        result = {}
        group_sets = []
        for name in _OUTPUTS:
            data = prepare_sequence_forecasts(getattr(self, name), policy=policy, limits=limits)
            result[name] = {
                "partition_digest": data.digest,
                "group_count": len(data.group_digests),
                "audit": data.audit.to_dict(),
            }
            group_sets.append(data.group_digests)
        if any(a & b for i, a in enumerate(group_sets) for b in group_sets[i + 1 :]):
            raise ValueError("prepared neural partitions share declared groups")
        return result


class _LegacyNaN:
    __slots__ = ()


_NAN_PARENT = _LegacyNaN()


def _constant(text: str) -> _LegacyNaN:
    if text != "NaN":
        raise ValueError("nonfinite source JSON value")
    return _NAN_PARENT


def _float(text: str) -> float:
    number = float(text)
    if not math.isfinite(number):
        raise ValueError("nonfinite source JSON number")
    return number


def _integer(text: str) -> int:
    if len(text) > 20:
        raise ValueError("source JSON integer exceeds its bound")
    value = int(text)
    if not -(2**63) <= value < 2**63:
        raise ValueError("source JSON integer exceeds its bound")
    return value


def _text(value: Any, maximum: int, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value.strip()) or len(value) > maximum:
        raise ValueError("source text has invalid type or size")
    size = 0
    try:
        for offset in range(0, len(value), 4096):
            size += len(value[offset : offset + 4096].encode("utf-8"))
            if size > maximum:
                raise ValueError("source text exceeds its UTF-8 byte bound")
    except UnicodeError as error:
        raise ValueError("source text contains invalid Unicode") from error
    return value


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, child in pairs:
        if key in value:
            raise ValueError("duplicate source JSON key")
        value[key] = child
    return value


def _lexical(raw: bytes, limits: NeuralCgaLimits, *, metadata: bool) -> str:
    """Bound JSON tree admission before the decoder allocates containers."""
    text = raw.decode("utf-8")
    nodes = depth = index = 0
    object_fields: dict[int, int] = {}
    awaiting_key: set[int] = set()
    while index < len(text):
        char = text[index]
        if char in " \t\r\n,:]}":
            if char == ":":
                awaiting_key.discard(depth)
            elif char == "," and depth in object_fields:
                awaiting_key.add(depth)
            elif char == "}":
                object_fields.pop(depth, None)
                awaiting_key.discard(depth)
            depth -= char in "]}"
            index += 1
            continue
        nodes += 1
        if nodes > limits.max_json_nodes:
            raise ValueError("source JSON exceeds its node bound")
        if char in "[{":
            depth += 1
            if char == "{":
                object_fields[depth] = 0
                awaiting_key.add(depth)
            if depth > limits.max_json_depth:
                raise ValueError("source JSON exceeds its depth bound")
            index += 1
        else:
            if depth + 1 > limits.max_json_depth:
                raise ValueError("source JSON exceeds its depth bound")
            if char == '"':
                if depth in awaiting_key:
                    object_fields[depth] += 1
                    cap = (
                        limits.max_conversations
                        if metadata and depth == 1
                        else limits.max_object_fields
                    )
                    if object_fields[depth] > cap:
                        raise ValueError("source JSON exceeds its object field bound")
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
                start = index
                while index < len(text) and text[index] not in " \t\r\n,:[]{}":
                    index += 1
                if index - start > 128:
                    raise ValueError("source JSON token exceeds its bound")
    return text


def _tree(value: Any, limits: NeuralCgaLimits, *, metadata: bool) -> None:
    stack: list[tuple[Any, int, str | None]] = [(value, 1, None)]
    nodes = 0
    encoded_size = 0
    byte_bound = limits.max_metadata_bytes if metadata else limits.max_row_bytes
    while stack:
        item, depth, key = stack.pop()
        nodes += 1
        if nodes > limits.max_json_nodes or depth > limits.max_json_depth:
            raise ValueError("source JSON exceeds its node/depth bound")
        if item is _NAN_PARENT:
            if metadata or depth != 2 or key != "reply-to":
                raise ValueError("legacy NaN is allowed only in top-level reply-to")
            encoded_size += 3
        elif type(item) is dict:
            bound = (
                limits.max_conversations if metadata and depth == 1 else limits.max_object_fields
            )
            if len(item) > bound or 2 * len(item) > limits.max_json_nodes - nodes - len(stack):
                raise ValueError("source JSON object exceeds its field/node bound")
            encoded_size += 2 + max(0, len(item) - 1) + len(item)
            for name, child in item.items():
                if type(name) is not str:
                    raise ValueError("source JSON object keys must be strings")
                stack.append((name, depth + 1, None))
                stack.append((child, depth + 1, name))
        elif type(item) is list:
            if len(item) > limits.max_json_nodes - nodes - len(stack):
                raise ValueError("source JSON array exceeds its node bound")
            encoded_size += 2 + max(0, len(item) - 1)
            stack.extend((child, depth + 1, None) for child in item)
        elif type(item) is str:
            _text(item, byte_bound, empty=True)
            encoded_size += 2
            for offset in range(0, len(item), 4096):
                chunk = item[offset : offset + 4096]
                encoded_size += len(chunk.encode("utf-8"))
                encoded_size += sum(
                    1 if char in '\\"\b\f\n\r\t' else 5 if ord(char) < 32 else 0 for char in chunk
                )
                if encoded_size > byte_bound:
                    raise ValueError("source JSON exceeds its aggregate encoded byte bound")
        elif type(item) is int:
            if not -(2**63) <= item < 2**63:
                raise ValueError("source integer exceeds its bound")
            encoded_size += len(str(item))
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("nonfinite source numeric value")
            encoded_size += len(repr(item))
        elif item is not None and type(item) is not bool:
            raise ValueError("source must contain closed JSON values")
        else:
            encoded_size += 5 if item is False else 4
        if encoded_size > byte_bound:
            raise ValueError("source JSON exceeds its aggregate encoded byte bound")


def _json(raw: bytes, limits: NeuralCgaLimits, *, metadata: bool = False) -> Any:
    bound = limits.max_metadata_bytes if metadata else limits.max_row_bytes
    if len(raw) > bound:
        raise ValueError("source JSON exceeds its encoded byte bound")
    try:
        value = json.loads(
            _lexical(raw, limits, metadata=metadata),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
            parse_float=_float,
            parse_int=_integer,
        )
        _tree(value, limits, metadata=metadata)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid source JSON") from error
    return value


def _shape(value: Any, names: set[str]) -> dict[str, Any]:
    if type(value) is not dict or len(value) != len(names) or value.keys() != names:
        raise ValueError("source object has unknown or missing fields")
    return value


def _timestamp(value: Any) -> datetime:
    if (
        type(value) not in (int, float)
        or not -62135596800 <= value < 253402300800
        or not math.isfinite(value)
    ):
        raise ValueError("source timestamp must be finite UTC seconds in datetime range")
    try:
        return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=value)
    except (OverflowError, ValueError) as error:
        raise ValueError("source timestamp is outside the UTC datetime range") from error


def _component_assignments(metadata: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    ids = sorted(key for key, value in metadata.items() if value["split"] == "val")
    parents = {key: key for key in ids}

    def find(key: str) -> str:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def union(first: str, second: str) -> None:
        a, b = find(first), find(second)
        if a != b:
            parents[max(a, b)] = min(a, b)

    pages: dict[int, str] = {}
    for key in ids:
        value = metadata[key]
        union(key, value["pair_id"])
        page = value["page_id"]
        if page in pages:
            union(key, pages[page])
        else:
            pages[page] = key
    components: dict[str, list[str]] = defaultdict(list)
    for key in ids:
        components[find(key)].append(key)
    assignments = {}
    bucket_components: Counter[int] = Counter()
    bucket_conversations: Counter[int] = Counter()
    component_bindings = []
    for keys in sorted(components.values()):
        digest = _digest({"seed": SEED, "conversation_ids": keys})
        bucket = int(digest, 16) % 5
        output = "policy_validation" if bucket in (0, 1) else "validation"
        assignments.update(dict.fromkeys(keys, output))
        bucket_components[bucket] += 1
        bucket_conversations[bucket] += len(keys)
        component_bindings.append({"sha256": digest, "bucket": bucket, "conversations": len(keys)})
    return assignments, {
        "components": len(components),
        "largest_component_conversations": max(map(len, components.values()), default=0),
        "component_counts_by_bucket": {str(i): bucket_components[i] for i in range(5)},
        "conversation_counts_by_bucket": {str(i): bucket_conversations[i] for i in range(5)},
        "component_inventory_sha256": _digest(component_bindings),
    }


def partition_neural_cga(
    metadata: Any, records: Iterable[Any], *, limits: NeuralCgaLimits | None = None
) -> NeuralCgaPartitions:
    """Validate authored/native rows, then split all validation components before eligibility.

    This entry point does not claim an archive origin. Only load_neural_cga pins
    source bytes. Native values must already be closed JSON (or the private NaN
    sentinel emitted by the strict source parser).
    """
    return _partition_neural_cga(metadata, records, _limits(limits), validated_json=False)


def _partition_neural_cga(
    metadata: Any,
    records: Iterable[Any],
    bounds: NeuralCgaLimits,
    *,
    validated_json: bool,
) -> NeuralCgaPartitions:
    # Only the local file loader uses validated_json=True, immediately after its
    # strict bounded _json boundary. The public native-data adapter always
    # validates the complete tree, including inert fields, independently.
    if type(metadata) is not dict or not 1 <= len(metadata) <= bounds.max_conversations:
        raise ValueError("source conversation inventory exceeds its bound")
    if not validated_json:
        _tree(metadata, bounds, metadata=True)
    page_splits: dict[int, str] = {}
    for identifier, raw in metadata.items():
        _text(identifier, bounds.max_identifier_bytes)
        value = _shape(raw, _METADATA_FIELDS)
        for name in ("page_title", "annotation_year"):
            _text(value[name], bounds.max_identifier_bytes)
        _text(value["pair_id"], bounds.max_identifier_bytes)
        for name in ("conversation_has_personal_attack", "verified", "pair_verified"):
            if type(value[name]) is not bool:
                raise ValueError("source conversation labels must be boolean")
        page, split = value["page_id"], value["split"]
        if (
            type(page) is not int
            or not 0 <= page < 2**63
            or type(split) is not str
            or split not in _SPLITS
        ):
            raise ValueError("invalid page identity or official source split")
        if page in page_splits and page_splits[page] != split:
            raise ValueError("page groups overlap official splits")
        page_splits[page] = split
    for identifier, value in metadata.items():
        partner = value["pair_id"]
        if (
            partner == identifier
            or partner not in metadata
            or metadata[partner]["pair_id"] != identifier
            or metadata[partner]["split"] != value["split"]
        ):
            raise ValueError(
                "source pairs must be reciprocal, nonself, and within one official split"
            )
    # Native metadata fields are now closed scalars. A user-supplied record
    # iterator cannot subsequently mutate their validated split/label declarations.
    metadata = {key: value.copy() for key, value in metadata.items()}
    assignments, components = _component_assignments(metadata)
    by_conversation: dict[str, list[Utterance]] = defaultdict(list)
    seen: set[str] = set()
    headers = legacy_nan = text_bytes = 0
    input_hasher = hashlib.sha256()
    # Binding is normalized below to timestamp/ID order, not input JSONL order.
    for value in records:
        if len(seen) >= bounds.max_utterances:
            raise ValueError("source utterance inventory exceeds its bound")
        if not validated_json:
            _tree(value, bounds, metadata=False)
        row = _shape(value, _ROW_FIELDS)
        identifier = _text(row["id"], bounds.max_identifier_bytes)
        conversation = _text(row["conversation_id"], bounds.max_identifier_bytes)
        if identifier in seen or conversation not in metadata:
            raise ValueError("duplicate utterance or unknown source conversation")
        seen.add(identifier)
        _text(row["speaker"], bounds.max_identifier_bytes)
        text = _text(row["text"], bounds.max_text_bytes, empty=True)
        text_bytes += len(text.encode("utf-8"))
        if text_bytes > bounds.max_total_text_bytes:
            raise ValueError("source feature text exceeds its aggregate byte bound")
        meta = row["meta"]
        if (
            type(meta) is not dict
            or len(meta) not in (3, 4)
            or not _META_REQUIRED <= meta.keys() <= _META_ALLOWED
            or type(meta["parsed"]) is not list
        ):
            raise ValueError("invalid native utterance metadata schema")
        for name in ("is_section_header", "comment_has_personal_attack"):
            if type(meta[name]) is not bool:
                raise ValueError("human event/header annotations must be boolean")
        if "toxicity" in meta and type(meta["toxicity"]) not in (int, float):
            raise ValueError("ignored toxicity must retain its native numeric type")
        parent = row["reply-to"]
        if parent is _NAN_PARENT:
            legacy_nan += 1
        elif parent is not None:
            _text(parent, bounds.max_identifier_bytes)
        headers += meta["is_section_header"]
        by_conversation[conversation].append(
            Utterance(
                identifier,
                "speaker",
                text,
                _timestamp(row["timestamp"]),
                metadata={
                    "event": meta["comment_has_personal_attack"],
                    "is_section_header": meta["is_section_header"],
                },
            )
        )
    if set(by_conversation) != metadata.keys():
        raise ValueError("source metadata and observed conversation inventories differ")
    outputs: dict[str, list[Conversation]] = {name: [] for name in _OUTPUTS}
    official_labels: dict[str, Counter[str]] = {name: Counter() for name in _SPLITS}
    output_labels: dict[str, Counter[str]] = {name: Counter() for name in _OUTPUTS}
    group_outputs: dict[str, str] = {}
    for identifier, value in sorted(metadata.items()):
        turns = sorted(by_conversation[identifier], key=lambda turn: (turn.timestamp, turn.id))
        label = any(
            turn.metadata["event"] for turn in turns if not turn.metadata["is_section_header"]
        )
        if label != value["conversation_has_personal_attack"]:
            raise ValueError("conversation label disagrees with nonheader human utterance events")
        split = value["split"]
        output = (
            assignments[identifier]
            if split == "val"
            else "training"
            if split == "train"
            else "test"
        )
        groups = ["page:" + str(value["page_id"]), "pair:" + min(identifier, value["pair_id"])]
        for group in groups:
            if group in group_outputs and group_outputs[group] != output:
                raise ValueError("page/pair isolation failed in derived partitions")
            group_outputs[group] = output
        outputs[output].append(
            Conversation(identifier, turns, metadata={"forecast_groups": groups})
        )
        official_labels[split]["positive" if label else "negative"] += 1
        output_labels[output]["positive" if label else "negative"] += 1
        input_hasher.update(
            _canonical(
                {
                    "conversation_id": identifier,
                    "output": output,
                    "groups": groups,
                    "turns": [
                        [
                            turn.id,
                            turn.timestamp.isoformat(),
                            turn.text,
                            turn.metadata["event"],
                            turn.metadata["is_section_header"],
                        ]
                        for turn in turns
                    ],
                }
            )
        )
        input_hasher.update(b"\n")
    audit = {
        "format": "turnscope.neural-cga-source-audit.v1",
        "protocol": protocol(),
        "protocol_sha256": protocol_digest(),
        "source_archive_verified": False,
        "raw_conversations": len(metadata),
        "raw_utterances": len(seen),
        "section_headers": headers,
        "source_text_bytes": text_bytes,
        "legacy_reply_to_nan": legacy_nan,
        "reply_edge_filtering": False,
        "other_nonfinite_values": "rejected",
        "official_conversation_counts": dict(Counter(v["split"] for v in metadata.values())),
        "official_label_counts": {name: dict(official_labels[name]) for name in _SPLITS},
        "partition_conversation_counts": {name: len(outputs[name]) for name in _OUTPUTS},
        "partition_label_counts": {name: dict(output_labels[name]) for name in _OUTPUTS},
        "page_groups": len(page_splits),
        "pair_groups": len(metadata) // 2,
        "cross_partition_page_pair_groups": 0,
        "validation_components": components,
        "normalized_partition_input_sha256": input_hasher.hexdigest(),
        "assignment_sha256": _digest([[key, assignments[key]] for key in sorted(assignments)]),
        "limits": asdict(bounds),
        "raw_identifiers_and_text_published": False,
    }
    return NeuralCgaPartitions(
        tuple(outputs["training"]),
        tuple(outputs["validation"]),
        tuple(outputs["policy_validation"]),
        tuple(outputs["test"]),
        _canonical(audit),
    )


def _archive_admission(stream: BinaryIO, size: int, limits: NeuralCgaLimits) -> None:
    stream.seek(max(0, size - 65557))
    tail = stream.read(65557)
    position = tail.rfind(b"PK\x05\x06")
    if position < 0 or len(tail) - position < _END.size:
        raise ValueError("source archive has no bounded ZIP end record")
    end = _END.unpack_from(tail, position)
    _, disk, start_disk, count, total, directory_bytes, directory_start, comment = end
    absolute_end = size - len(tail) + position
    if (
        disk
        or start_disk
        or not 1 <= count == total <= limits.max_members
        or directory_bytes > limits.max_directory_bytes
        or directory_start + directory_bytes != absolute_end
        or position + _END.size + comment != len(tail)
    ):
        raise ValueError(
            "source archive exceeds member/directory bounds or uses unsupported ZIP framing"
        )
    stream.seek(0)


def load_neural_cga(
    path: str | os.PathLike[str], *, limits: NeuralCgaLimits | None = None
) -> NeuralCgaPartitions:
    """Verify the fixed local archive, then read only two bounded required members."""
    bounds = _limits(limits)
    source_path = Path(path)
    before = source_path.lstat()
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= bounds.max_archive_bytes:
        raise ValueError("source archive must be a bounded regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    with os.fdopen(os.open(source_path, flags), "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ):
            raise ValueError("source archive changed while opening")
        total = 0
        while chunk := stream.read(min(_MIB, bounds.max_archive_bytes + 1 - total)):
            total += len(chunk)
            if total > bounds.max_archive_bytes:
                raise ValueError("source archive exceeds its encoded byte bound")
            digest.update(chunk)
        if digest.hexdigest() != ARCHIVE_SHA256:
            raise ValueError("source archive SHA-256 differs from the pinned CGA release")
        _archive_admission(stream, total, bounds)
        with ZipFile(stream) as archive:
            infos = archive.infolist()
            if len(infos) > bounds.max_members or len({info.filename for info in infos}) != len(
                infos
            ):
                raise ValueError("duplicate or excessive source archive members")
            for info in infos:
                if (
                    info.flag_bits & 1
                    or stat.S_ISLNK(info.external_attr >> 16)
                    or ".." in info.filename.split("/")
                    or info.filename.startswith("/")
                    or "\\" in info.filename
                    or ":" in info.filename
                ):
                    raise ValueError("unsafe source archive member metadata")
            metadata_info = archive.getinfo(PREFIX + "conversations.json")
            utterance_info = archive.getinfo(PREFIX + "utterances.jsonl")
            for info, bound in (
                (metadata_info, bounds.max_metadata_bytes),
                (utterance_info, bounds.max_utterance_bytes),
            ):
                if (
                    info.is_dir()
                    or info.compress_type not in (ZIP_STORED, ZIP_DEFLATED)
                    or info.file_size > bound
                ):
                    raise ValueError("required source member exceeds its expanded byte bound")
            with archive.open(metadata_info) as member:
                metadata_raw = member.read(bounds.max_metadata_bytes + 1)
            if len(metadata_raw) != metadata_info.file_size:
                raise ValueError("metadata member length differs from its declaration")
            metadata = _json(metadata_raw, bounds, metadata=True)
            utterance_digest = hashlib.sha256()
            consumed = 0
            with archive.open(utterance_info) as member:

                def rows() -> Iterable[Any]:
                    nonlocal consumed
                    while raw := member.readline(bounds.max_row_bytes + 1):
                        consumed += len(raw)
                        if len(raw) > bounds.max_row_bytes or consumed > bounds.max_utterance_bytes:
                            raise ValueError("source JSONL exceeds row/expanded stream bounds")
                        utterance_digest.update(raw)
                        yield _json(raw, bounds)

                result = _partition_neural_cga(metadata, rows(), bounds, validated_json=True)
            if consumed != utterance_info.file_size:
                raise ValueError("utterance member length differs from its declaration")
        after = os.fstat(stream.fileno())
    current = source_path.lstat()
    if any(
        (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        for value in (after, current)
    ):
        raise ValueError("source archive changed while reading")
    audit = result.audit_dict()
    audit.update(
        {
            "source_archive_verified": True,
            "archive_sha256": digest.hexdigest(),
            "archive_bytes": total,
            "metadata_bytes": len(metadata_raw),
            "utterances_bytes": consumed,
            "metadata_sha256": hashlib.sha256(metadata_raw).hexdigest(),
            "utterances_sha256": utterance_digest.hexdigest(),
            "read_members": 2,
            "ignored_members": len(infos) - 2,
            "ignored_members_are_not_expanded": True,
        }
    )
    return NeuralCgaPartitions(
        result.training, result.validation, result.policy_validation, result.test, _canonical(audit)
    )
