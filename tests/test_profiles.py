from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from turnscope import ContextBuilder, Conversation, Utterance
from turnscope.policies import TokenBudgetPolicy, Utf8ByteTokenCounter
from turnscope.profiles import get_profile, load_profiles


def test_named_profiles_load_policy_counter_and_audit_budget(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "profiles": {
                    "default": {
                        "policy": {
                            "kind": "token",
                            "value": 4,
                            "include_target": True,
                            "token_counter": "utf8-byte",
                            "bytes_per_token": 2,
                        },
                        "audit": {"token_budget": 32},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    profile = get_profile(path, "default")
    assert isinstance(profile.policy, TokenBudgetPolicy)
    assert isinstance(profile.token_counter, Utf8ByteTokenCounter)
    assert profile.token_budget == 32
    conversation = Conversation(
        "c",
        [Utterance("u", "user", "你好", datetime(2026, 1, 1, tzinfo=timezone.utc))],
    )
    assert (
        ContextBuilder(profile.policy, profile.token_counter).build(conversation)[0].token_total
        == 0
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"profiles": {}},
        {"profiles": {"x": {"policy": {"kind": "unknown"}}}},
        {"profiles": {"x": {"policy": {"kind": "turn"}, "extra": 1}}},
        {"profiles": {"x": {"policy": {"kind": "turn"}, "audit": {"token_budget": -1}}}},
    ],
)
def test_profiles_reject_invalid_shapes(tmp_path, payload: object) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_profiles(path)


def test_profiles_reject_unknown_name_and_duplicate_keys(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "profiles.json"
    path.write_text('{"profiles":{"default":{"policy":{"kind":"turn"}}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown profile"):
        get_profile(path, "missing")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"profiles":{},"profiles":{}}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_profiles(duplicate)
