from __future__ import annotations

from importlib import metadata

import pytest

from turnscope.cli import main
from turnscope.plugins import list_plugins, load_rule, load_tokenizer


class _Rule:
    name = "external-rule"

    def check(self, conversation):  # type: ignore[no-untyped-def]
        del conversation
        return ()


class _Entry:
    def __init__(self, name: str, value: str, loaded: object, group: str) -> None:
        self.name = name
        self.value = value
        self.group = group
        self._loaded = loaded

    def load(self) -> object:
        return self._loaded


def test_plugin_discovery_is_sorted_and_loads_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = (
        _Entry("z", "pkg:z", lambda text: len(text), "turnscope.tokenizers"),
        _Entry("a", "pkg:a", _Rule, "turnscope.audit_rules"),
    )
    monkeypatch.setattr(metadata, "entry_points", lambda: entries)
    assert [item.name for item in list_plugins()] == ["a", "z"]
    assert load_rule("a").name == "external-rule"
    assert load_tokenizer("z")("a b") == 3


def test_plugin_loader_rejects_unknown_and_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = (_Entry("bad", "pkg:bad", object(), "turnscope.tokenizers"),)
    monkeypatch.setattr(metadata, "entry_points", lambda: entries)
    with pytest.raises(KeyError, match="unknown tokenizers plugin"):
        load_tokenizer("missing")
    with pytest.raises(TypeError, match="must be callable"):
        load_tokenizer("bad")


def test_plugins_cli_lists_without_importing(monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(metadata, "entry_points", lambda: ())
    assert main(["plugins"]) == 0
    assert capsys.readouterr().out == "[]\n"
