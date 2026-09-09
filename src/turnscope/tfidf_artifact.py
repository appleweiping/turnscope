"""Strict, portable fitted TF-IDF artifact encoding."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path

_FIELDS = {"format", "version", "tokenizer", "documents", "min_df", "max_features", "features"}
_TOKENIZER = "unicode-word-apostrophe-hyphen-casefold-v1"


def _digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def encode_model(
    documents: int, min_df: int, max_features: int | None, frequencies: Mapping[str, int]
) -> dict[str, object]:
    value: dict[str, object] = {
        "format": "turnscope-tfidf",
        "version": 1,
        "tokenizer": _TOKENIZER,
        "documents": documents,
        "min_df": min_df,
        "max_features": max_features,
        "features": [{"term": term, "df": frequency} for term, frequency in frequencies.items()],
    }
    return {**value, "sha256": _digest(value)}


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def decode_model(value: object) -> tuple[int, int, int | None, dict[str, int]]:
    if not isinstance(value, dict) or set(value) != _FIELDS | {"sha256"}:
        raise ValueError("invalid TF-IDF artifact fields")
    if value["format"] != "turnscope-tfidf":
        raise ValueError("unsupported TF-IDF artifact format")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("unsupported TF-IDF artifact version")
    if value["tokenizer"] != _TOKENIZER:
        raise ValueError("unsupported TF-IDF tokenizer")
    documents = _positive_int(value["documents"], "documents")
    min_df = _positive_int(value["min_df"], "min_df")
    max_features = (
        None
        if value["max_features"] is None
        else _positive_int(value["max_features"], "max_features")
    )
    features = value["features"]
    if not isinstance(features, list):
        raise ValueError("features must be a list")
    frequencies: dict[str, int] = {}
    for feature in features:
        if not isinstance(feature, dict) or set(feature) != {"term", "df"}:
            raise ValueError("invalid TF-IDF feature fields")
        term = feature["term"]
        if (
            not isinstance(term, str)
            or not term
            or term.casefold() != term
            or any(character.isspace() for character in term)
            or term in frequencies
        ):
            raise ValueError("feature terms must be unique non-empty casefolded tokens")
        frequency = _positive_int(feature["df"], "feature df")
        if not min_df <= frequency <= documents:
            raise ValueError("feature df must be between min_df and documents")
        frequencies[term] = frequency
    if max_features is not None and len(frequencies) > max_features:
        raise ValueError("features exceed max_features")
    if list(frequencies) != sorted(frequencies, key=lambda term: (-frequencies[term], term)):
        raise ValueError("features must be ordered by decreasing df, then term")
    if value["sha256"] != _digest({key: value[key] for key in _FIELDS}):
        raise ValueError("TF-IDF artifact checksum mismatch")
    return documents, min_df, max_features, frequencies


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def load_model(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load TF-IDF model: {error}") from error


def save_model(value: dict[str, object], path: Path) -> None:
    rendered = (
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(rendered)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
