"""Strict JSON configuration for reusable build and audit profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from .policies import (
    ReplyChainPolicy,
    TimeWindowPolicy,
    TokenBudgetPolicy,
    TokenCounter,
    TurnWindowPolicy,
    Utf8ByteTokenCounter,
    WhitespaceTokenCounter,
    WindowPolicy,
)


@dataclass(frozen=True, slots=True)
class Profile:
    """One named policy/audit profile loaded from a JSON configuration."""

    name: str
    policy: WindowPolicy
    token_counter: TokenCounter = field(default_factory=WhitespaceTokenCounter)
    token_budget: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("profile name must not be empty")
        if self.token_budget is not None and (
            isinstance(self.token_budget, bool)
            or not isinstance(self.token_budget, int)
            or self.token_budget < 0
        ):
            raise ValueError("profile audit.token_budget must be non-negative")


def load_profiles(path: str | Path) -> dict[str, Profile]:
    """Load strict named profiles from a JSON file."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load profile config {source}: {error}") from error
    if not isinstance(raw, dict) or set(raw) != {"profiles"}:
        raise ValueError("profile config must contain exactly a profiles object")
    profiles = raw["profiles"]
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("profile config profiles must be a non-empty object")
    return {name: _profile(name, value) for name, value in profiles.items()}


def get_profile(path: str | Path, name: str) -> Profile:
    """Load one named profile and reject unknown names."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("profile name must be a non-empty string")
    profiles = load_profiles(path)
    if name not in profiles:
        raise ValueError(f"unknown profile {name!r}; available: {', '.join(sorted(profiles))}")
    return profiles[name]


def _profile(name: str, raw: Any) -> Profile:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("profile names must be non-empty strings")
    if not isinstance(raw, dict) or set(raw) - {"policy", "audit"}:
        raise ValueError(f"profile {name!r} has unknown fields")
    policy_value = raw.get("policy")
    if not isinstance(policy_value, dict):
        raise ValueError(f"profile {name!r} policy must be an object")
    audit = raw.get("audit", {})
    if not isinstance(audit, dict) or set(audit) - {"token_budget"}:
        raise ValueError(f"profile {name!r} audit has unknown fields")
    token_budget = audit.get("token_budget")
    counter_name = policy_value.get("token_counter", "whitespace")
    if counter_name == "whitespace":
        counter: TokenCounter = WhitespaceTokenCounter()
    elif counter_name == "utf8-byte":
        counter = Utf8ByteTokenCounter(policy_value.get("bytes_per_token", 4))
    else:
        raise ValueError(f"profile {name!r} has unsupported token_counter")
    policy_kind = policy_value.get("kind", "turn")
    value = policy_value.get("value")
    if policy_kind == "turn":
        policy: WindowPolicy = TurnWindowPolicy(5 if value is None else value)
    elif policy_kind == "token":
        policy = TokenBudgetPolicy(
            512 if value is None else value,
            include_target=policy_value.get("include_target", False),
        )
    elif policy_kind == "time":
        policy = TimeWindowPolicy(timedelta(seconds=3600 if value is None else value))
    elif policy_kind == "reply-chain":
        policy = ReplyChainPolicy(value)
    else:
        raise ValueError(f"profile {name!r} has unsupported policy kind {policy_kind!r}")
    return Profile(name, policy, counter, token_budget)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate profile config key {key!r}")
        result[key] = value
    return result
