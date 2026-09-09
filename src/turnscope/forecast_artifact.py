"""Strict, bounded JSON artifacts for cumulative prefix forecasters."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .forecast import ForecastState, PrefixEventForecaster
from .io import parse_json_value

MAX_FORECAST_BYTES = 32 * 1024 * 1024
_CONFIG = {
    "alpha",
    "max_features",
    "min_turns",
    "event_field",
    "skip_field",
    "groups_field",
    "max_prefixes",
    "max_prefix_cells",
    "max_tokens",
}
_STATE = {
    "vocabulary",
    "log_probabilities",
    "positive_prior",
    "training_conversations",
    "training_prefixes",
    "validation_conversations",
    "validation_prefixes",
    "training_groups",
    "validation_groups",
    "threshold",
    "validation_balanced_accuracy",
}


def checksum(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def encode_forecaster(model: PrefixEventForecaster) -> dict[str, Any]:
    state = model.state
    payload = {
        "format": "turnscope.prefix-event.weighted-nb.v1",
        "config": dict(model._config),
        "state": {
            "vocabulary": list(state.vocabulary),
            "log_probabilities": [
                [row[term] for term in state.vocabulary] for row in state.log_probabilities
            ],
            "positive_prior": state.positive_prior,
            "training_conversations": state.training_conversations,
            "training_prefixes": state.training_prefixes,
            "validation_conversations": state.validation_conversations,
            "validation_prefixes": state.validation_prefixes,
            "training_groups": sorted(state.training_groups),
            "validation_groups": sorted(state.validation_groups),
            "threshold": state.threshold,
            "validation_balanced_accuracy": state.validation_balanced_accuracy,
        },
    }
    return {**payload, "sha256": checksum(payload)}


def _object(value: object, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("invalid forecaster artifact fields")
    return value


def _number(value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError("forecaster parameters must be finite numbers")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError("forecaster parameters must be finite numbers") from error
    if not math.isfinite(converted):
        raise ValueError("forecaster parameters must be finite numbers")
    return converted


def decode_forecaster(value: object) -> PrefixEventForecaster:
    payload = _object(value, {"format", "config", "state", "sha256"})
    if payload["format"] != "turnscope.prefix-event.weighted-nb.v1":
        raise ValueError("unsupported prefix forecaster format")
    model = PrefixEventForecaster(**_object(payload["config"], _CONFIG))
    state = _object(payload["state"], _STATE)
    vocabulary = state["vocabulary"]
    if (
        not isinstance(vocabulary, list)
        or len(vocabulary) > model._config["max_features"]
        or not all(
            isinstance(term, str)
            and term
            and term == term.casefold()
            and not any(char.isspace() for char in term)
            for term in vocabulary
        )
        or len(set(vocabulary)) != len(vocabulary)
    ):
        raise ValueError("invalid forecaster vocabulary")
    logs = state["log_probabilities"]
    if (
        not isinstance(logs, list)
        or len(logs) != 2
        or any(not isinstance(row, list) or len(row) != len(vocabulary) for row in logs)
    ):
        raise ValueError("invalid class likelihood dimensions")
    rows = [[_number(item) for item in row] for row in logs]
    likelihood_floor = math.log(model._config["alpha"]) - math.log(
        model._config["max_tokens"] + model._config["alpha"] * len(vocabulary)
    )
    if any(value < likelihood_floor - 1e-10 for row in rows for value in row):
        raise ValueError("class likelihoods exceed the fitted token-budget numeric bounds")
    if any(value > 0 for row in rows for value in row) or any(
        row and not math.isclose(math.fsum(math.exp(value) for value in row), 1.0, abs_tol=1e-8)
        for row in rows
    ):
        raise ValueError("class likelihoods must be normalized log probabilities")
    for partition in ("training", "validation"):
        count, prefixes = state[partition + "_conversations"], state[partition + "_prefixes"]
        if (
            type(count) is not int
            or type(prefixes) is not int
            or not 2 <= count <= prefixes <= model._config["max_prefixes"]
        ):
            raise ValueError("invalid forecaster sample counts")
        groups = state[partition + "_groups"]
        if (
            not isinstance(groups, list)
            or not count <= len(groups) <= 400_000
            or not all(
                isinstance(group, str) and re.fullmatch(r"[0-9a-f]{64}", group) for group in groups
            )
            or groups != sorted(set(groups))
        ):
            raise ValueError("invalid forecaster group digests")
    if set(state["training_groups"]) & set(state["validation_groups"]):
        raise ValueError("training and validation groups overlap")
    prior = _number(state["positive_prior"])
    if not 0 < prior < 1:
        raise ValueError("invalid training class prior")
    positive_count = prior * state["training_conversations"]
    rounded_count = round(positive_count)
    if (
        not 0 < prior < 1
        or not 1 <= rounded_count < state["training_conversations"]
        or not math.isclose(
            prior, rounded_count / state["training_conversations"], rel_tol=0, abs_tol=1e-12
        )
    ):
        raise ValueError("invalid training class prior")
    threshold, accuracy = (
        _number(state["threshold"]),
        _number(state["validation_balanced_accuracy"]),
    )
    if not 0 <= threshold <= 1 or not 0 <= accuracy <= 1:
        raise ValueError("invalid forecaster validation policy")
    if payload["sha256"] != checksum(
        {key: item for key, item in payload.items() if key != "sha256"}
    ):
        raise ValueError("prefix forecaster checksum mismatch")
    model._state = ForecastState(
        tuple(vocabulary),
        (
            MappingProxyType(dict(zip(vocabulary, rows[0], strict=True))),
            MappingProxyType(dict(zip(vocabulary, rows[1], strict=True))),
        ),
        prior,
        state["training_conversations"],
        state["training_prefixes"],
        state["validation_conversations"],
        state["validation_prefixes"],
        frozenset(state["training_groups"]),
        frozenset(state["validation_groups"]),
        threshold,
        accuracy,
    )
    return model


def save_forecaster(value: dict[str, Any], path: Path) -> None:
    data = (
        json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False, indent=2) + "\n"
    ).encode()
    if len(data) > MAX_FORECAST_BYTES:
        raise ValueError("prefix forecaster artifact exceeds 32 MiB")
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


def load_forecaster(path: Path) -> object:
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_FORECAST_BYTES + 1)
        if len(data) > MAX_FORECAST_BYTES:
            raise ValueError("prefix forecaster artifact exceeds 32 MiB")
        return parse_json_value(data.decode("utf-8"), location="prefix forecaster artifact")
    except (OSError, UnicodeError, RecursionError) as error:
        raise ValueError(f"cannot load prefix forecaster artifact: {error}") from error
