"""Pinned CGA reply-context retrieval; engineering evidence, not dialogue-act accuracy.

The protocol is fixed before model evaluation. Raw text, IDs, features and fitted
parameters are private and must not be included in the published aggregate report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import sys
import time
import tracemalloc
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Any
from zipfile import ZipFile

ARCHIVE_SHA256 = "84e2d1ac60a3269251b5e175e549fc65cec875fdc926f99172b5b70d3ca1b122"
_PREFIX = "conversations-gone-awry-corpus/"
_SPLITS = ("train", "val", "test")
_MAX_ROW_BYTES = 16 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
_MAX_METADATA_BYTES = 4 * 1024 * 1024
_MAX_UTTERANCE_BYTES = 256 * 1024 * 1024
NodeKey = tuple[str, str]
Edge = tuple[NodeKey, NodeKey]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, name: str, *, empty: bool = False, maximum: int = 1_000_000) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be valid text")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as error:
        raise ValueError(f"{name} must contain valid Unicode") from error
    if size > maximum:
        raise ValueError(f"{name} exceeds its byte limit")
    return value


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _constant(_value: str) -> Any:
    raise ValueError("nonfinite JSON value")


def _float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number")
    return number


def _json(raw: bytes) -> Any:
    return json.loads(
        raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=_constant, parse_float=_float
    )


@dataclass(frozen=True)
class _LegacyNonfinite:
    literal: str


_LEGACY_NAN_PARENT = _LegacyNonfinite("NaN")


def _source_record_json(raw: bytes) -> Any:
    """Quarantine this pinned source's top-level reply-to NaN; never repair it.

    This is deliberately NOT the public model JSON parser or general JSON
    compatibility. Literal NaN elsewhere, Infinity and numeric overflow fail.
    """
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_object,
        parse_constant=_LegacyNonfinite,
        parse_float=_float,
    )
    stack: list[tuple[tuple[Any, ...], Any]] = [((), value)]
    while stack:
        path, item = stack.pop()
        if isinstance(item, _LegacyNonfinite):
            if path != ("reply-to",) or item.literal != "NaN":
                raise ValueError("nonfinite value outside the source's legacy NaN parent field")
            value["reply-to"] = _LEGACY_NAN_PARENT
        elif isinstance(item, dict):
            stack.extend(((*path, key), child) for key, child in item.items())
        elif isinstance(item, list):
            stack.extend(((*path, index), child) for index, child in enumerate(item))
    return value


@dataclass(frozen=True)
class BenchmarkProtocol:
    seed: int = 20260909
    n_components: int = 16
    drop_first_component: bool = True
    n_clusters: int = 8
    max_features: int = 256
    min_document_frequency: int = 3
    max_queries: int = 256
    max_candidates: int = 1024
    ks: tuple[int, ...] = (1, 5, 10)
    max_shuffle_attempts: int = 128

    def __post_init__(self) -> None:
        for name, maximum in (
            ("seed", 2**32 - 1),
            ("n_components", 256),
            ("n_clusters", 64),
            ("max_features", 1024),
            ("min_document_frequency", 50_000),
            ("max_queries", 1024),
            ("max_candidates", 10_000),
            ("max_shuffle_attempts", 128),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"invalid protocol {name}")
        if type(self.drop_first_component) is not bool:
            raise ValueError("drop_first_component must be boolean")
        if (
            not isinstance(self.ks, tuple)
            or not self.ks
            or any(type(k) is not int or not 1 <= k <= 1000 for k in self.ks)
            or tuple(sorted(set(self.ks))) != self.ks
        ):
            raise ValueError("retrieval cutoffs must be distinct increasing positive integers")

    @property
    def digest(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class CgaNode:
    key: NodeKey
    text: str
    reply_to: str | _LegacyNonfinite | None
    is_header: bool


@dataclass(frozen=True)
class CgaGroup:
    split: str
    page_id: int
    pair_id: str


@dataclass(frozen=True)
class CgaData:
    nodes: Mapping[NodeKey, CgaNode]
    groups: Mapping[str, CgaGroup]
    forward_edges: tuple[Edge, ...]
    audit: Mapping[str, int]
    fold_audit: Mapping[str, Mapping[str, int]]

    def catalog(self, split: str) -> tuple[CgaNode, ...]:
        if split not in _SPLITS:
            raise ValueError("unknown corpus split")
        return tuple(
            node
            for key, node in sorted(self.nodes.items())
            if self.groups[key[0]].split == split and not node.is_header
        )

    def edges(self, split: str, direction: str = "forward") -> tuple[Edge, ...]:
        if split not in _SPLITS or direction not in ("forward", "backward"):
            raise ValueError("unknown split or relation direction")
        edges = tuple(edge for edge in self.forward_edges if self.groups[edge[0][0]].split == split)
        return edges if direction == "forward" else tuple(sorted((b, a) for a, b in edges))

    @property
    def input_digest(self) -> str:
        return _digest(
            {
                "records": [
                    [
                        list(key),
                        node.text,
                        {"legacy_nonfinite": "NaN"}
                        if node.reply_to is _LEGACY_NAN_PARENT
                        else node.reply_to,
                        node.is_header,
                    ]
                    for key, node in sorted(self.nodes.items())
                ],
                "groups": [[key, asdict(group)] for key, group in sorted(self.groups.items())],
                "accepted_edges": self.forward_edges,
            }
        )


def _cycle_nodes(parents: Mapping[NodeKey, NodeKey]) -> set[NodeKey]:
    """Return precisely cycle members, not acyclic tails entering a cycle."""
    done: set[NodeKey] = set()
    cyclic: set[NodeKey] = set()
    for start in sorted(parents):
        if start in done:
            continue
        path: list[NodeKey] = []
        positions: dict[NodeKey, int] = {}
        node = start
        while node in parents and node not in done and node not in positions:
            positions[node] = len(path)
            path.append(node)
            node = parents[node]
        if node in positions:
            cyclic.update(path[positions[node] :])
        done.update(path)
    return cyclic


def audit_cga(metadata: Any, records: Iterable[Any]) -> CgaData:
    """Validate official grouping; classify and exclude invalid/header relations.

    No source text or reply pointer is rewritten. Supervision, toxicity, parses,
    speaker IDs and timestamps are not retained as model features.
    """
    if not isinstance(metadata, dict) or not 1 <= len(metadata) <= 10_000:
        raise ValueError("CGA metadata must contain a bounded conversation inventory")
    groups: dict[str, CgaGroup] = {}
    page_splits: dict[int, set[str]] = defaultdict(set)
    for identifier, value in metadata.items():
        _text(identifier, "conversation ID", maximum=1000)
        if not isinstance(value, dict):
            raise ValueError("CGA conversation metadata must be an object")
        split = value.get("split")
        partner = _text(value.get("pair_id"), "paired conversation ID", maximum=1000)
        page_id = value.get("page_id")
        if split not in _SPLITS or type(page_id) is not int or not 0 <= page_id < 2**63:
            raise ValueError("invalid official split or page ID")
        if (
            partner == identifier
            or partner not in metadata
            or not isinstance(metadata[partner], dict)
        ):
            raise ValueError("missing or self-paired conversation")
        if (
            metadata[partner].get("pair_id") != identifier
            or metadata[partner].get("split") != split
        ):
            raise ValueError("nonreciprocal or cross-split matched pair")
        groups[identifier] = CgaGroup(split, page_id, min(identifier, partner))
        page_splits[page_id].add(split)
    if any(len(values) != 1 for values in page_splits.values()):
        raise ValueError("page groups overlap official splits")
    nodes: dict[NodeKey, CgaNode] = {}
    by_id: dict[str, NodeKey] = {}
    fold: dict[str, Counter[str]] = {split: Counter() for split in _SPLITS}
    for value in records:
        if len(nodes) >= 50_000:
            raise ValueError("CGA record count exceeds its bound")
        if not isinstance(value, dict):
            raise ValueError("CGA utterance must be an object")
        identifier = _text(value.get("id"), "utterance ID", maximum=1000)
        conversation = _text(value.get("conversation_id"), "conversation ID", maximum=1000)
        text = _text(value.get("text"), "utterance text", empty=True)
        if identifier in by_id or conversation not in groups:
            raise ValueError("duplicate utterance ID or unknown conversation")
        if "reply-to" not in value:
            raise ValueError("original reply-to field is required; adjacency is not a substitute")
        parent = value["reply-to"]
        if parent is not None and parent is not _LEGACY_NAN_PARENT:
            _text(parent, "reply-to ID", maximum=1000)
        meta = value.get("meta")
        if not isinstance(meta, dict) or type(meta.get("is_section_header")) is not bool:
            raise ValueError("CGA header annotation must be boolean")
        key = (conversation, identifier)
        node = CgaNode(key, text, parent, meta["is_section_header"])
        nodes[key], by_id[identifier] = node, key
        fold[groups[conversation].split]["nodes"] += 1
        fold[groups[conversation].split]["header_nodes"] += node.is_header
    if {key[0] for key in nodes} != set(groups):
        raise ValueError("conversation metadata and observed records differ")
    # Check relation integrity before filtering headers, with a single primary
    # reason per raw edge. Removing a cycle edge never invents a replacement.
    candidates: dict[NodeKey, NodeKey] = {}
    for key, node in sorted(nodes.items()):
        counts = fold[groups[key[0]].split]
        if node.reply_to is None:
            counts["null_parent"] += 1
            continue
        counts["nonnull_edges"] += 1
        if node.reply_to is _LEGACY_NAN_PARENT:
            counts["legacy_nan_parent"] += 1
            continue
        parent_key = by_id.get(node.reply_to)
        if parent_key is None:
            counts["dangling_edges"] += 1
        elif parent_key == key:
            counts["self_edges"] += 1
        elif parent_key[0] != key[0]:
            counts["cross_conversation_edges"] += 1
        else:
            candidates[key] = parent_key
    cyclic = _cycle_nodes(candidates)
    accepted = []
    for key, parent_key in sorted(candidates.items()):
        counts = fold[groups[key[0]].split]
        if key in cyclic:
            counts["cycle_edges"] += 1
        elif nodes[key].is_header or nodes[parent_key].is_header:
            counts["header_edges"] += 1
        else:
            accepted.append((parent_key, key))
            counts["accepted_edges"] += 1
    fields = (
        "nodes",
        "header_nodes",
        "null_parent",
        "nonnull_edges",
        "legacy_nan_parent",
        "dangling_edges",
        "self_edges",
        "cross_conversation_edges",
        "cycle_edges",
        "header_edges",
        "accepted_edges",
    )
    frozen = {
        split: MappingProxyType({name: fold[split][name] for name in fields}) for split in _SPLITS
    }
    totals = {name: sum(fold[split][name] for split in _SPLITS) for name in fields}
    if totals["nonnull_edges"] != sum(totals[name] for name in fields[4:]):
        raise RuntimeError("edge exclusion inventory does not reconcile")
    return CgaData(
        MappingProxyType(nodes),
        MappingProxyType(groups),
        tuple(sorted(accepted)),
        MappingProxyType(totals),
        MappingProxyType(frozen),
    )


def load_cga_archive(path: Path) -> tuple[CgaData, dict[str, Any]]:
    """Read only the pinned archive's metadata and utterance stream; never extract."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > _MAX_ARCHIVE_BYTES:
                raise ValueError("CGA archive exceeds its encoded byte bound")
            digest.update(chunk)
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("CGA archive SHA-256 differs from the pinned release")
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("duplicate archive member")
        meta_info = archive.getinfo(_PREFIX + "conversations.json")
        utterance_info = archive.getinfo(_PREFIX + "utterances.jsonl")
        if (
            meta_info.file_size > _MAX_METADATA_BYTES
            or utterance_info.file_size > _MAX_UTTERANCE_BYTES
        ):
            raise ValueError("CGA archive member exceeds its expanded byte bound")
        metadata_bytes = archive.read(meta_info)
        utterance_hash = hashlib.sha256()
        with archive.open(utterance_info) as stream:

            def records() -> Iterable[Any]:
                total = 0
                while raw := stream.readline(_MAX_ROW_BYTES + 1):
                    total += len(raw)
                    if len(raw) > _MAX_ROW_BYTES or total > _MAX_UTTERANCE_BYTES:
                        raise ValueError("CGA JSONL row or stream exceeds its byte bound")
                    utterance_hash.update(raw)
                    yield _source_record_json(raw)

            data = audit_cga(_json(metadata_bytes), records())
    legacy_nodes = {key for key, node in data.nodes.items() if node.reply_to is _LEGACY_NAN_PARENT}
    return data, {
        "archive_sha256": digest.hexdigest(),
        "archive_bytes": size,
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "utterances_sha256": utterance_hash.hexdigest(),
        "model_input_sha256": data.input_digest,
        "audit": dict(data.audit),
        "fold_audit": {s: dict(v) for s, v in data.fold_audit.items()},
        "conversation_counts": dict(Counter(group.split for group in data.groups.values())),
        "cross_split_page_groups": 0,
        "cross_split_pair_groups": 0,
        "source_json_policy": {
            "strict_json": data.audit["legacy_nan_parent"] == 0,
            "legacy_nonfinite_path": ["reply-to"],
            "legacy_nonfinite_literal": "NaN",
            "legacy_nonfinite_count": data.audit["legacy_nan_parent"],
            "legacy_parent_nodes_with_nonheader_text_retained": sum(
                not data.nodes[key].is_header for key in legacy_nodes
            ),
            "accepted_edges_from_legacy_parent_nodes": sum(
                source in legacy_nodes for source, _ in data.forward_edges
            ),
            "handling": "quarantine parent edge, retain node text and valid incoming edges",
            "other_nonfinite_values": "rejected",
        },
        "edge_policy": (
            "classify null/legacy-NaN-parent/dangling/self/cross-conversation/cycle/header; "
            "retain other authentic parent-to-reply edges"
        ),
    }


@dataclass(frozen=True)
class RetrievalQuery:
    key: NodeKey
    positives: tuple[NodeKey, ...]


@dataclass(frozen=True)
class RetrievalTask:
    split: str
    direction: str
    queries: tuple[RetrievalQuery, ...]
    candidates: tuple[NodeKey, ...]
    eligible_queries: int
    available_candidates: int
    protocol_digest: str

    @property
    def digest(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class ShuffledTraining:
    records: Mapping[NodeKey, str]
    edges: tuple[Edge, ...] | None
    report: Mapping[str, Any]


def shuffled_training(data: CgaData, protocol: BenchmarkProtocol) -> ShuffledTraining:
    """Degree-preserving, structurally conditioned null; never inspect scores.

    The synthetic scope permits target reassignment across original conversations.
    Original composite IDs remain opaque JSON strings, not delimiter concatenation.
    """
    catalog = data.catalog("train")
    original = data.edges("train")
    if not original:
        raise ValueError("training null requires authentic edges")
    reindex = {node.key: ("synthetic-null", _canonical(node.key).decode()) for node in catalog}
    records = MappingProxyType({reindex[node.key]: node.text for node in catalog})
    source = [reindex[a] for a, _ in original]
    target = [reindex[b] for _, b in original]
    rng = random.Random(protocol.seed)
    rejected: Counter[str] = Counter()
    chosen = None
    attempts = 0
    for _attempt in range(protocol.max_shuffle_attempts):
        attempts += 1
        proposal = target.copy()
        rng.shuffle(proposal)
        edges = tuple(sorted(zip(source, proposal, strict=True)))
        if any(a == b for a, b in edges):
            rejected["self_edges"] += 1
        elif len(set(edges)) != len(edges):
            rejected["duplicate_edges"] += 1
        # Authentic reply graphs have at most one incoming parent per target.
        # The target multiset is preserved, so this functional cycle check applies.
        elif _cycle_nodes({b: a for a, b in edges}):
            rejected["cycles"] += 1
        else:
            chosen = edges
            break
    original_reindexed = {(reindex[a], reindex[b]) for a, b in original}
    original_contexts = {key for edge in original for key in edge}
    chosen_contexts = {key for edge in chosen for key in edge} if chosen is not None else None
    catalog_identity = _digest([[node.key, node.text] for node in catalog])
    context_identity = _digest([[key, data.nodes[key].text] for key in sorted(original_contexts)])
    if chosen is not None:
        if (
            Counter(a for a, _ in chosen) != Counter(source)
            or Counter(b for _, b in chosen) != Counter(target)
            or chosen_contexts != {reindex[key] for key in original_contexts}
        ):
            raise RuntimeError("null degree or shared-context inventory changed")
        decoded_catalog = sorted(
            [[tuple(json.loads(key[1])), text] for key, text in records.items()]
        )
        decoded_contexts = sorted(
            [[tuple(json.loads(key[1])), records[key]] for key in chosen_contexts]
        )
        if (
            _digest(decoded_catalog) != catalog_identity
            or _digest(decoded_contexts) != context_identity
        ):
            raise RuntimeError("null changed original training text or shared context input")
    return ShuffledTraining(
        records,
        chosen,
        MappingProxyType(
            {
                "status": "available" if chosen is not None else "unavailable",
                "scope": "synthetic-null, opaque original composite IDs",
                "policy": (
                    "target multiset permutation conditioned on no self/duplicate/cyclic edges"
                ),
                "uniform_over_all_mappings": False,
                "seed": protocol.seed,
                "attempt_limit": protocol.max_shuffle_attempts,
                "attempts": attempts,
                "rejected_attempts": dict(rejected),
                "unchanged_edge_fraction": (
                    len(set(chosen) & original_reindexed) / len(original)
                    if chosen is not None
                    else None
                ),
                "training_catalog_sha256": catalog_identity,
                "shared_context_input_sha256": context_identity,
                "same_text_and_context_inventory_verified": chosen is not None,
                "degrees_preserved": chosen is not None,
            }
        ),
    )


_TOKEN = re.compile(r"[\w]+(?:['-][\w]+)*", re.UNICODE)


class LexicalBaseline:
    """Independent train-utterance-fitted relative-TF/IDF cosine baseline."""

    def __init__(self, catalog: tuple[CgaNode, ...], protocol: BenchmarkProtocol) -> None:
        if not catalog:
            raise ValueError("lexical baseline requires training utterances")
        frequency: Counter[str] = Counter()
        for node in catalog:
            frequency.update({m.group(0).casefold() for m in _TOKEN.finditer(node.text)})
        selected = sorted(
            (term for term, count in frequency.items() if count >= protocol.min_document_frequency),
            key=lambda term: (-frequency[term], term),
        )[: protocol.max_features]
        self.idf = MappingProxyType(
            {
                term: math.log((1 + len(catalog)) / (1 + frequency[term])) + 1
                for term in sorted(selected)
            }
        )
        self.training_records = len(catalog)

    @property
    def digest(self) -> str:
        return _digest([self.training_records, dict(self.idf)])

    def project(self, text: str) -> tuple[float, ...] | None:
        counts = Counter(m.group(0).casefold() for m in _TOKEN.finditer(text))
        values = tuple(counts[term] * idf for term, idf in self.idf.items())
        length = math.hypot(*values)
        return tuple(value / length for value in values) if length else None


def cosine(left: tuple[float, ...] | None, right: tuple[float, ...] | None) -> float | None:
    if left is None or right is None:
        return None
    if not left or len(left) != len(right):
        raise ValueError("cosine vectors require matching nonempty dimensions")
    # The numeric model and baseline return unit vectors. Validate rather than
    # rewarding malformed projections with an arbitrary clipped dot product.
    for value in (left, right):
        if any(type(x) not in (int, float) or not -1.00000001 <= x <= 1.00000001 for x in value):
            raise ValueError("invalid unit-vector coordinate")
        if not math.isclose(math.hypot(*value), 1, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("retrieval projections must be unit length")
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(left, right, strict=True))))


def mean_projection(vectors: Iterable[tuple[float, ...] | None]) -> tuple[float, ...] | None:
    available = [value for value in vectors if value is not None]
    if not available:
        return None
    dimensions = len(available[0])
    if any(len(value) != dimensions for value in available):
        raise ValueError("mean projection dimensions differ")
    values = tuple(
        math.fsum(value[i] for value in available) / len(available) for i in range(dimensions)
    )
    length = math.hypot(*values)
    return tuple(value / length for value in values) if length else None


def make_retrieval_task(
    data: CgaData, split: str, direction: str, protocol: BenchmarkProtocol
) -> RetrievalTask:
    if split not in ("val", "test"):
        raise ValueError("retrieval evaluation requires an official heldout split")
    positives: dict[NodeKey, set[NodeKey]] = defaultdict(set)
    for source, target in data.edges(split, direction):
        positives[source].add(target)
    if not positives:
        raise ValueError("no eligible heldout relation queries")

    def order(kind: str, key: NodeKey) -> tuple[str, NodeKey]:
        return _digest([kind, protocol.seed, direction, key]), key

    selected = sorted(positives, key=lambda key: order("query", key))[: protocol.max_queries]
    required = {target for key in selected for target in positives[key]}
    if len(required) > protocol.max_candidates:
        raise ValueError(
            "all positive contexts exceed the candidate budget; no positives truncated"
        )
    pool = {node.key for node in data.catalog(split)}
    remaining = sorted(pool - required, key=lambda key: order("candidate", key))
    candidates = tuple(sorted(required | set(remaining[: protocol.max_candidates - len(required)])))
    queries = tuple(RetrievalQuery(key, tuple(sorted(positives[key]))) for key in selected)
    return RetrievalTask(
        split, direction, queries, candidates, len(positives), len(pool), protocol.digest
    )


def ranking_metrics(
    scores: Mapping[NodeKey, float | None], positives: tuple[NodeKey, ...], ks: tuple[int, ...]
) -> dict[str, Any]:
    """Multi-positive rank; exact score ties use ascending composite IDs.

    Undefined candidates are unrankable, never removed from the positive recall
    denominator. The complete candidate inventory must be checked by the caller.
    """
    if (
        not isinstance(scores, Mapping)
        or not 1 <= len(scores) <= 10_000
        or any(
            not isinstance(key, tuple)
            or len(key) != 2
            or any(not isinstance(item, str) for item in key)
            for key in scores
        )
        or not isinstance(ks, tuple)
        or not ks
        or any(type(k) is not int or not 1 <= k <= 1000 for k in ks)
        or tuple(sorted(set(ks))) != ks
    ):
        raise ValueError("invalid bounded candidate inventory or retrieval cutoffs")
    if not positives or len(set(positives)) != len(positives) or not set(positives) <= set(scores):
        raise ValueError("positive contexts must be unique and present in the candidate inventory")
    available = []
    for key, score in scores.items():
        if score is not None:
            if type(score) not in (int, float) or not -1 <= score <= 1 or not math.isfinite(score):
                raise ValueError("retrieval similarity must be finite in [-1,1] or undefined")
            available.append((key, float(score)))
    ranking = [key for key, _ in sorted(available, key=lambda item: (-item[1], item[0]))]
    ranks = [i + 1 for i, key in enumerate(ranking) if key in positives]
    return {
        "reciprocal_rank": 1.0 / min(ranks) if ranks else 0.0,
        "recall": {str(k): sum(rank <= k for rank in ranks) / len(positives) for k in ks},
        "candidates": len(scores),
        "scored_candidates": len(available),
        "positives": len(positives),
        "scored_positives": len(ranks),
    }


ScoreQuery = Callable[[NodeKey, tuple[NodeKey, ...]], Mapping[NodeKey, float | None] | None]


def evaluate_retrieval(
    data: CgaData, task: RetrievalTask, scorer: ScoreQuery, protocol: BenchmarkProtocol
) -> dict[str, Any]:
    """Aggregate all selected queries, including zero-credit undefined queries."""
    if task != make_retrieval_task(data, task.split, task.direction, protocol):
        raise ValueError("retrieval task must exactly match its frozen data/protocol")
    rr: list[float] = []
    recalls: dict[str, list[float]] = {str(k): [] for k in protocol.ks}
    page_values: dict[int, list[float]] = defaultdict(list)
    pair_values: dict[str, list[float]] = defaultdict(list)
    page_recalls: dict[str, dict[int, list[float]]] = {
        str(k): defaultdict(list) for k in protocol.ks
    }
    pair_recalls: dict[str, dict[str, list[float]]] = {
        str(k): defaultdict(list) for k in protocol.ks
    }
    counts: Counter[str] = Counter()
    conditional = []
    for query in task.queries:
        candidates = tuple(key for key in task.candidates if key != query.key)
        scores = scorer(query.key, candidates)
        if scores is not None and (
            not isinstance(scores, Mapping) or set(scores) != set(candidates)
        ):
            raise ValueError("scorer must return the exact fixed candidate inventory")
        metrics = ranking_metrics(
            dict.fromkeys(candidates) if scores is None else scores, query.positives, protocol.ks
        )
        counts["undefined_queries"] += scores is None
        counts["scored_queries"] += scores is not None
        for field in ("candidates", "scored_candidates", "positives", "scored_positives"):
            counts[field] += metrics[field]
        value = metrics["reciprocal_rank"]
        rr.append(value)
        if scores is not None:
            conditional.append(value)
        for k in recalls:
            recalls[k].append(metrics["recall"][k])
        group = data.groups[query.key[0]]
        page_values[group.page_id].append(value)
        pair_values[group.pair_id].append(value)
        for k in recalls:
            page_recalls[k][group.page_id].append(metrics["recall"][k])
            pair_recalls[k][group.pair_id].append(metrics["recall"][k])

    def mean(values: list[float]) -> float:
        return math.fsum(values) / len(values)

    return {
        "task_sha256": task.digest,
        "split": task.split,
        "direction": task.direction,
        "eligible_queries": task.eligible_queries,
        "selected_queries": len(task.queries),
        "available_candidates": task.available_candidates,
        "candidate_pool": len(task.candidates),
        "query_coverage": counts["scored_queries"] / len(task.queries),
        "zero_credit_mrr": mean(rr),
        "scorable_mrr": mean(conditional) if conditional else None,
        "zero_credit_recall": {k: mean(values) for k, values in recalls.items()},
        "page_macro_zero_credit_mrr": mean([mean(values) for values in page_values.values()]),
        "pair_macro_zero_credit_mrr": mean([mean(values) for values in pair_values.values()]),
        "page_macro_zero_credit_recall": {
            k: mean([mean(values) for values in groups.values()])
            for k, groups in page_recalls.items()
        },
        "pair_macro_zero_credit_recall": {
            k: mean([mean(values) for values in groups.values()])
            for k, groups in pair_recalls.items()
        },
        "page_groups": len(page_values),
        "pair_groups": len(pair_values),
        **dict(counts),
    }


def _vector_scorer(
    queries: Mapping[NodeKey, tuple[float, ...] | None],
    candidates: Mapping[NodeKey, tuple[float, ...] | None],
) -> ScoreQuery:
    # Freeze and validate each representation once, not once per candidate pair.
    # Otherwise lexical cosine would repeat hundreds of coordinate checks for
    # every pair in the bounded 256 x 1024 task.
    def freeze(
        values: Mapping[NodeKey, tuple[float, ...] | None],
    ) -> dict[NodeKey, tuple[float, ...] | None]:
        result = {}
        for key, vector in values.items():
            frozen = None if vector is None else tuple(vector)
            if frozen is not None:
                cosine(frozen, frozen)
            result[key] = frozen
        return result

    frozen_queries, frozen_candidates = freeze(queries), freeze(candidates)
    dimensions = {
        len(v) for v in (*frozen_queries.values(), *frozen_candidates.values()) if v is not None
    }
    if len(dimensions) > 1:
        raise ValueError("query and candidate vector dimensions differ")

    def score(query: NodeKey, keys: tuple[NodeKey, ...]) -> Mapping[NodeKey, float | None] | None:
        vector = frozen_queries[query]
        if vector is None:
            return None
        return {
            key: (
                None
                if frozen_candidates[key] is None
                else max(
                    -1.0,
                    min(
                        1.0,
                        math.fsum(
                            a * b for a, b in zip(vector, frozen_candidates[key], strict=True)
                        ),
                    ),
                )
            )
            for key in keys
        }

    return score


def evaluate_models(data: CgaData, protocol: BenchmarkProtocol) -> dict[str, Any]:
    """Fit full official training once per main/null model; never select on scores."""
    from turnscope.dual_context import ContextEdge, ContextRecord, DualContextModel

    # Build every task before fitting or evaluating anything. The edge-grounded
    # pool policy is intentionally visible, not a deployment retrieval claim.
    tasks = tuple(
        make_retrieval_task(data, split, direction, protocol)
        for split in ("val", "test")
        for direction in ("forward", "backward")
    )
    nodes = data.catalog("train")
    edges = data.edges("train")
    null = shuffled_training(data, protocol)
    config = {
        "n_components": protocol.n_components,
        "drop_first": protocol.drop_first_component,
        "min_document_frequency": protocol.min_document_frequency,
        "max_features": protocol.max_features,
        "n_clusters": protocol.n_clusters,
        "max_kmeans_iterations": 100,
        "seed": protocol.seed,
    }
    models = {}
    fitted = {}
    for name, catalog, training_edges in (
        ("dual_context", {node.key: node.text for node in nodes}, edges),
        ("shuffled_relation_null", null.records, null.edges),
    ):
        if training_edges is None:
            continue
        model = DualContextModel(**config)
        started = time.perf_counter()
        if tracemalloc.is_tracing():
            raise ValueError("benchmark requires ownership of its fit memory measurement")
        tracemalloc.start()
        try:
            model.fit(
                (ContextRecord(*key, text) for key, text in sorted(catalog.items())),
                (ContextEdge(a[0], a[1], b[1]) for a, b in training_edges),
                (ContextEdge(b[0], b[1], a[1]) for a, b in training_edges),
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        fit_seconds = time.perf_counter() - started
        artifact = model.to_dict()
        restored = DualContextModel.from_dict(artifact)
        if restored.digest != model.digest:
            raise RuntimeError("fitted artifact roundtrip changed model identity")
        models[name] = restored
        fitted[name] = {
            "model_sha256": model.digest,
            "tokenizer": artifact["tokenizer"],
            "fit_seconds": fit_seconds,
            "fit_python_tracked_peak_bytes": peak,
            "artifact_roundtrip_verified": True,
            "training": model.training_summary(),
        }
    main = models["dual_context"]
    if "shuffled_relation_null" in models:
        shuffled = models["shuffled_relation_null"]
        if (
            main.state.vocabulary != shuffled.state.vocabulary
            or main.state.document_frequencies != shuffled.state.document_frequencies
            or len(main.state.singular_values) != len(shuffled.state.singular_values)
            or any(
                not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-11)
                for a, b in zip(main.state.column_norms, shuffled.state.column_norms, strict=True)
            )
            or any(
                not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-11)
                for a, b in zip(
                    main.state.singular_values, shuffled.state.singular_values, strict=True
                )
            )
        ):
            raise RuntimeError("null changed shared TFIDF/SVD input statistics")
    baseline_started = time.perf_counter()
    lexical = LexicalBaseline(nodes, protocol)
    lexical_identity = lexical.digest
    all_context_keys = {key for edge in edges for key in edge}
    train_contexts = {
        key: main.project_context(data.nodes[key].text).vector for key in all_context_keys
    }
    means = {
        direction: mean_projection(train_contexts[b] for _, b in data.edges("train", direction))
        for direction in ("forward", "backward")
    }
    baseline_seconds = time.perf_counter() - baseline_started
    reports: dict[str, list[dict[str, Any]]] = defaultdict(list)
    geometry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evaluation_start = time.perf_counter()
    for task in tasks:
        query_keys = tuple(query.key for query in task.queries)
        lex_candidates = {key: lexical.project(data.nodes[key].text) for key in task.candidates}
        lex_queries = {key: lexical.project(data.nodes[key].text) for key in query_keys}
        reports["lexical_tfidf"].append(
            evaluate_retrieval(data, task, _vector_scorer(lex_queries, lex_candidates), protocol)
        )
        for name, model in models.items():
            candidates = {
                key: model.project_context(data.nodes[key].text).vector for key in task.candidates
            }
            predictions = {key: model.predict(data.nodes[key].text) for key in query_keys}
            queries = {
                key: getattr(prediction, task.direction).vector
                for key, prediction in predictions.items()
            }
            reports[name].append(
                evaluate_retrieval(data, task, _vector_scorer(queries, candidates), protocol)
            )
            cluster_counts: Counter[int] = Counter()
            reasons: Counter[str] = Counter()
            ranges, shifts, orientations = [], [], []
            for prediction in predictions.values():
                directional = getattr(prediction, task.direction)
                reasons[directional.reason] += 1
                if directional.cluster_id is not None:
                    cluster_counts[directional.cluster_id] += 1
                if directional.range is not None:
                    ranges.append(directional.range)
                if prediction.shift is not None:
                    shifts.append(prediction.shift)
                if prediction.orientation is not None:
                    orientations.append(prediction.orientation)

            def descriptive(values: list[float], total: int = len(query_keys)) -> dict[str, Any]:
                return {
                    "defined": len(values),
                    "total": total,
                    "mean": math.fsum(values) / len(values) if values else None,
                }

            geometry[name].append(
                {
                    "split": task.split,
                    "direction": task.direction,
                    "projection_reasons": dict(reasons),
                    "cluster_assignment_counts": {
                        str(k): v for k, v in sorted(cluster_counts.items())
                    },
                    "range": descriptive(ranges),
                    "shift": descriptive(shifts),
                    "orientation": descriptive(orientations),
                    "has_gold_cluster_labels": False,
                }
            )
            if name == "dual_context":
                reports["training_context_mean"].append(
                    evaluate_retrieval(
                        data,
                        task,
                        _vector_scorer(
                            dict.fromkeys(query_keys, means[task.direction]), candidates
                        ),
                        protocol,
                    )
                )
    for name, model in models.items():
        if model.digest != fitted[name]["model_sha256"]:
            raise RuntimeError("evaluation changed frozen model parameters")
    if lexical.digest != lexical_identity:
        raise RuntimeError("heldout evaluation changed lexical baseline")
    return {
        "model_config": config,
        "fitted_models": fitted,
        "lexical_baseline": {
            "fitted_sha256": lexical_identity,
            "training_records": lexical.training_records,
            "features": len(lexical.idf),
            "column_normalization": False,
        },
        "training_mean": {
            "weighting": "one vote per authentic training edge target occurrence",
            "query_independent": True,
            "defined": {name: value is not None for name, value in means.items()},
        },
        "null_control": dict(null.report),
        "retrieval": dict(reports),
        "descriptive_geometry_not_accuracy": dict(geometry),
        "baseline_fit_and_training_context_projection_seconds": baseline_seconds,
        "retrieval_scoring_seconds": time.perf_counter() - evaluation_start,
        "all_frozen_parameters_unchanged": True,
    }


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = (
        *sorted((root / "src/turnscope").glob("*.py")),
        Path(__file__),
        root / "pyproject.toml",
    )
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def build_report(archive: Path, protocol: BenchmarkProtocol) -> dict[str, Any]:
    started = time.perf_counter()
    before = _source_hashes()
    data, source = load_cga_archive(archive)
    parsing_seconds = time.perf_counter() - started
    result = evaluate_models(data, protocol)
    if _source_hashes() != before:
        raise RuntimeError("runtime or benchmark source changed during evaluation")
    return {
        "format": "turnscope.dual-context-benchmark.v1",
        "kind": "real-source-fixed-candidate-reply-context-retrieval",
        "dataset": "CGA-WIKI",
        "dataset_source": "https://convokit.cornell.edu/documentation/awry.html",
        "source": source,
        "protocol": asdict(protocol),
        "protocol_sha256": protocol.digest,
        "candidate_policy": (
            "relation-gold-inclusive bounded pool, fixed before scoring; not full-corpus retrieval"
        ),
        "fitting_scope": (
            "all nonheader official training utterances; authentic retained train edges only"
        ),
        "test_used_for_this_task_tuning": False,
        "test_previously_used_for_other_task": (
            "earlier lexical personal-attack forecasting benchmark"
        ),
        "cluster_accuracy_evaluated": False,
        "quality_claim": (
            "observed retrieval evidence only; not ConvoKit parity or human dialogue-act validity"
        ),
        "redistribution": (
            "aggregate-only report; source archive, texts, IDs, vocabulary "
            "and fitted parameters remain local"
        ),
        "license_boundary": "dataset source availability is not an additional redistribution grant",
        "measurement_scope": (
            "fit function Python-tracked allocations, excludes parsing/external artifact roundtrip "
            "and is not process/native RSS"
        ),
        "source_sha256_before_and_after": before,
        "source_unchanged": True,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": importlib.metadata.version("numpy"),
        "unicode_database_version": unicodedata.unidata_version,
        "source_hashing_and_parsing_seconds": parsing_seconds,
        "end_to_end_seconds": time.perf_counter() - started,
        **result,
    }


def _diagnostic(value: Mapping[str, Any]) -> None:
    with suppress(OSError, UnicodeError, ValueError):
        print(json.dumps(dict(value), sort_keys=True), file=sys.stderr)


def _publish_report(destination: Path, payload: bytes) -> None:
    """Publish a fully written file via exclusive hard link; never replace a path."""
    temporary = None
    try:
        with NamedTemporaryFile(
            prefix=".dual-context-report-", dir=destination.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            # Only the exact fresh helper-owned file; never delete the destination.
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    published = False
    try:
        # Exclusive creation also protects source hardlinks, symlinks and races.
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("output must be a new path")
        if args.output.resolve() == args.archive.resolve() or not args.output.parent.is_dir():
            raise ValueError("output must differ from source and have an existing parent")
        result = build_report(args.archive, BenchmarkProtocol())
        payload = _canonical(result) + b"\n"
        if len(payload) > 4 * 1024 * 1024:
            raise ValueError("aggregate report exceeds 4 MiB")
        _publish_report(args.output, payload)
        published = True
        print(
            json.dumps(
                {"status": "completed", "report_sha256": hashlib.sha256(payload).hexdigest()},
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError, RuntimeError, TypeError, OverflowError, ImportError) as error:
        _diagnostic(
            {
                "status": "completed" if published else "failed",
                "report_published": published,
                "error_type": type(error).__name__,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
