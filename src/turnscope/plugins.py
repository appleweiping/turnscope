"""Opt-in entry-point discovery for external audit rules and token counters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Literal, cast

from .policies import TokenCounter

PluginKind = Literal["rules", "tokenizers"]
RULE_ENTRY_POINT_GROUP = "turnscope.audit_rules"
TOKENIZER_ENTRY_POINT_GROUP = "turnscope.tokenizers"


@dataclass(frozen=True, slots=True)
class PluginInfo:
    """Stable, non-executable information about an installed plugin."""

    name: str
    kind: PluginKind
    value: str


def list_plugins(kind: PluginKind | None = None) -> tuple[PluginInfo, ...]:
    """List installed plugins without importing their implementation code."""

    kinds: tuple[PluginKind, ...] = (kind,) if kind is not None else ("rules", "tokenizers")
    result: list[PluginInfo] = []
    for selected in kinds:
        entries = sorted(_entry_points(_group(selected)), key=lambda item: (item.name, item.value))
        names: set[str] = set()
        for entry in entries:
            if entry.name in names:
                raise ValueError(f"duplicate {selected} plugin name {entry.name!r}")
            names.add(entry.name)
            result.append(PluginInfo(entry.name, selected, entry.value))
    return tuple(result)


def load_tokenizer(name: str) -> TokenCounter:
    """Load one named tokenizer plugin and validate its callable contract."""

    loaded = _load(name, "tokenizers")
    if isinstance(loaded, type):
        loaded = loaded()
    if not callable(loaded):
        raise TypeError(f"tokenizer plugin {name!r} must be callable")
    return cast(TokenCounter, loaded)


def load_rule(name: str) -> Any:
    """Load one named rule plugin exposing ``name`` and ``check``."""

    loaded = _load(name, "rules")
    if isinstance(loaded, type):
        loaded = loaded()
    if not isinstance(getattr(loaded, "name", None), str) or not loaded.name:
        raise TypeError(f"rule plugin {name!r} must expose a non-empty string name")
    if not callable(getattr(loaded, "check", None)):
        raise TypeError(f"rule plugin {name!r} must expose check(conversation)")
    return loaded


def _load(name: str, kind: PluginKind) -> Any:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("plugin name must be a non-empty string")
    matches = [entry for entry in _entry_points(_group(kind)) if entry.name == name]
    if not matches:
        raise KeyError(f"unknown {kind} plugin {name!r}")
    if len(matches) != 1:
        raise ValueError(f"duplicate {kind} plugin name {name!r}")
    return matches[0].load()


def _group(kind: PluginKind) -> str:
    if kind == "rules":
        return RULE_ENTRY_POINT_GROUP
    if kind == "tokenizers":
        return TOKENIZER_ENTRY_POINT_GROUP
    raise ValueError(f"unknown plugin kind {kind!r}")


def _entry_points(group: str) -> tuple[metadata.EntryPoint, ...]:
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        return tuple(discovered.select(group=group))
    legacy = cast(Iterable[metadata.EntryPoint], discovered)
    return tuple(entry for entry in legacy if entry.group == group)
