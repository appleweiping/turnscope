"""Closed JSON schema and bounded loading for SVD-plus-ridge context models."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

from .expected_context import ExpectedContextModel, ExpectedContextState, _integer
from .io import parse_json_value
from .transformers import TfidfVectorizer

MAX_CONTEXT_MODEL_BYTES = 32 * 1024 * 1024
_FORMAT = "turnscope.expected-context.svd-ridge.v1"
_CONFIG = {
    "n_components",
    "regularization",
    "relation",
    "min_document_frequency",
    "max_features",
    "max_training_pairs",
    "max_dense_cells",
}
_STATE = {
    "basis",
    "coefficients",
    "source_mean",
    "context_mean",
    "singular_values",
    "training_pairs",
    "estimated_dense_cells",
    "training_mse",
    "mean_baseline_mse",
}


def _checksum(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def encode_context_model(model: ExpectedContextModel) -> dict[str, object]:
    state = model.state
    if model._source_model is None or model._context_model is None:
        raise ValueError("expected-context model has not been fitted")
    payload: dict[str, object] = {
        "format": _FORMAT,
        "config": {
            "n_components": model._components,
            "regularization": model._regularization,
            "relation": model._relation,
            "min_document_frequency": model._min_df,
            "max_features": model._max_features,
            "max_training_pairs": model._max_pairs,
            "max_dense_cells": model._max_cells,
        },
        "source_tfidf": model._source_model.to_dict(),
        "context_tfidf": model._context_model.to_dict(),
        "state": {
            "basis": [list(row) for row in state.basis],
            "coefficients": [list(row) for row in state.coefficients],
            "source_mean": list(state.source_mean),
            "context_mean": list(state.context_mean),
            "singular_values": list(state.singular_values),
            "training_pairs": state.training_pairs,
            "estimated_dense_cells": state.estimated_dense_cells,
            "training_mse": state.training_mse,
            "mean_baseline_mse": state.mean_baseline_mse,
        },
    }
    return {**payload, "sha256": _checksum(payload)}


def _object(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"invalid {name} fields")
    return value


def _number(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("context parameters must be finite numbers")
    try:
        result = float(value)  # type: ignore[arg-type]
    except OverflowError as error:
        raise ValueError("context parameters must be finite numbers") from error
    if not math.isfinite(result):
        raise ValueError("context parameters must be finite numbers")
    return result


def _vector(value: object, width: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"invalid {name} dimension")
    return tuple(_number(item) for item in value)


def _matrix(value: object, rows: int, columns: int, name: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"invalid {name} row count")
    return tuple(_vector(row, columns, name) for row in value)


def decode_context_model(value: object) -> ExpectedContextModel:
    payload = _object(
        value, {"format", "config", "source_tfidf", "context_tfidf", "state", "sha256"}, "model"
    )
    if payload["format"] != _FORMAT:
        raise ValueError("unsupported expected-context model format")
    config = _object(payload["config"], _CONFIG, "configuration")
    config = {**config, "regularization": _number(config["regularization"])}
    model = ExpectedContextModel(**config)
    state = _object(payload["state"], _STATE, "state")
    pairs = _integer(state["training_pairs"], "training_pairs", model._max_pairs)

    def vectorizer(raw: object) -> TfidfVectorizer:
        if not isinstance(raw, dict) or not isinstance(raw.get("features"), list):
            raise ValueError("invalid nested TF-IDF model")
        _integer(raw.get("documents"), "TF-IDF documents", pairs)
        if (
            not 1 <= len(raw["features"]) <= model._max_features
            or raw.get("min_df") != model._min_df
            or raw.get("max_features") != model._max_features
        ):
            raise ValueError("TF-IDF model does not match expected-context configuration")
        return TfidfVectorizer.from_dict(raw)

    source, context = vectorizer(payload["source_tfidf"]), vectorizer(payload["context_tfidf"])
    p, q = len(source.state.vocabulary), len(context.state.vocabulary)
    singular = state["singular_values"]
    if not isinstance(singular, list) or not 1 <= len(singular) <= min(
        model._components, q, context.state.documents
    ):
        raise ValueError("invalid retained context dimensions")
    k = len(singular)
    singular_values = _vector(singular, k, "singular_values")
    if (
        any(value <= 0 for value in singular_values)
        or list(singular_values) != sorted(singular_values, reverse=True)
        or math.fsum(value * value for value in singular_values) > context.state.documents + 1e-6
    ):
        raise ValueError("invalid context singular values")
    basis = _matrix(state["basis"], q, k, "basis")
    if any(abs(value) > 1 + 1e-8 for row in basis for value in row):
        raise ValueError("context basis must have orthonormal columns")
    for first in range(k):
        pivot = max(range(q), key=lambda row: abs(basis[row][first]))
        if basis[pivot][first] < 0:
            raise ValueError("context basis signs must be canonical")
        for second in range(first + 1):
            product = math.fsum(row[first] * row[second] for row in basis)
            if not math.isclose(product, float(first == second), abs_tol=1e-7):
                raise ValueError("context basis must have orthonormal columns")
    coefficients = _matrix(state["coefficients"], p, k, "coefficients")
    source_mean = _vector(state["source_mean"], p, "source_mean")
    context_mean = _vector(state["context_mean"], k, "context_mean")
    if (
        any(not 0 <= value <= 1 + 1e-8 for value in source_mean)
        or math.fsum(value * value for value in source_mean) > 1 + 1e-7
        or any(abs(value) > 1 + 1e-8 for value in context_mean)
        or math.fsum(value * value for value in context_mean) > 1 + 1e-7
    ):
        raise ValueError("invalid normalized training means")
    training_mse, baseline_mse = _number(state["training_mse"]), _number(state["mean_baseline_mse"])
    if not 0 <= training_mse <= baseline_mse + 1e-8 or not 0 <= baseline_mse <= 1 + 1e-8:
        raise ValueError("invalid training error summaries")
    estimated_k = min(model._components, context.state.documents, q)
    cells = (
        4 * pairs * p
        + 4 * context.state.documents * q
        + 3 * p * p
        + 3 * pairs * estimated_k
        + 2 * q * estimated_k
    )
    if (
        type(state["estimated_dense_cells"]) is not int
        or state["estimated_dense_cells"] != cells
        or cells > model._max_cells
    ):
        raise ValueError("invalid dense fitting workspace budget")
    if payload["sha256"] != _checksum(
        {key: item for key, item in payload.items() if key != "sha256"}
    ):
        raise ValueError("expected-context artifact checksum mismatch")
    model._source_model, model._context_model = source, context
    model._state = ExpectedContextState(
        source.state,
        context.state,
        basis,
        coefficients,
        source_mean,
        context_mean,
        singular_values,
        pairs,
        cells,
        training_mse,
        baseline_mse,
    )
    return model


def load_context_model(path: Path) -> object:
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_CONTEXT_MODEL_BYTES + 1)
        if len(data) > MAX_CONTEXT_MODEL_BYTES:
            raise ValueError("expected-context model exceeds 32 MiB")
        return parse_json_value(data.decode("utf-8"), location="expected-context model")
    except (OSError, UnicodeError, RecursionError) as error:
        raise ValueError(f"cannot load expected-context model: {error}") from error


def save_context_model(value: dict[str, object], path: Path) -> None:
    """Atomically save only artifacts that fit the loader's exact byte budget."""
    data = (
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")
    if len(data) > MAX_CONTEXT_MODEL_BYTES:
        raise ValueError("expected-context model exceeds 32 MiB")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
