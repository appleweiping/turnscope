"""Deterministic tabular exports for context windows."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import ContextWindow

_BASE_FIELDS = (
    "conversation_id",
    "target_id",
    "target_role",
    "target_timestamp",
    "policy",
    "token_total",
    "context_ids",
    "context_roles",
    "context_token_counts",
)


def window_rows(
    windows: Iterable[ContextWindow], *, include_text: bool = False
) -> tuple[dict[str, str | int], ...]:
    """Return stable, one-row-per-target mappings suitable for CSV export.

    Context collections are encoded as compact JSON arrays so commas and
    Unicode in IDs remain unambiguous. Message text is excluded by default to
    make an accidental tabular export less likely to leak conversation content.
    """

    rows: list[dict[str, str | int]] = []
    for window in windows:
        row: dict[str, str | int] = {
            "conversation_id": window.conversation_id,
            "target_id": window.target.id,
            "target_role": window.target.role,
            "target_timestamp": window.target.timestamp.isoformat().replace("+00:00", "Z"),
            "policy": window.policy,
            "token_total": window.token_total,
            "context_ids": _json_array(item.id for item in window.context),
            "context_roles": _json_array(item.role for item in window.context),
            "context_token_counts": _json_array(item.token_count for item in window.context),
        }
        if include_text:
            row["target_text"] = window.target.text
            row["context_text"] = _json_array(item.text for item in window.context)
        rows.append(row)
    return tuple(rows)


def windows_csv(windows: Iterable[ContextWindow], *, include_text: bool = False) -> str:
    """Serialize context windows as canonical UTF-8-compatible CSV text."""

    rows = window_rows(windows, include_text=include_text)
    fields = _BASE_FIELDS + (("target_text", "context_text") if include_text else ())
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_windows_csv(
    windows: Iterable[ContextWindow], path: str | Path, *, include_text: bool = False
) -> None:
    """Write :func:`windows_csv` to ``path`` using UTF-8."""

    Path(path).write_text(windows_csv(windows, include_text=include_text), encoding="utf-8")


def _json_array(values: Iterable[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


__all__ = ["window_rows", "windows_csv", "write_windows_csv"]
