"""Loopback JSON service for deterministic TurnScope operations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .audit import default_auditor
from .builder import ContextBuilder
from .graph import interaction_network
from .io import load_path
from .models import ContextWindow, Conversation, Severity
from .plugins import load_tokenizer
from .policies import (
    ReplyChainPolicy,
    TimeWindowPolicy,
    TokenBudgetPolicy,
    TurnWindowPolicy,
    whitespace_tokens,
)
from .reporting import windows_json
from .search import ConversationSearchIndex
from .tabular import window_rows


class TurnScopeService:
    """Dispatch strict JSON requests to public TurnScope workflows."""

    def dispatch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Execute one request using an input path and return JSON data."""
        if not isinstance(request, Mapping):
            raise ValueError("request must be an object")
        operation = request.get("operation")
        input_path = _required_path(request, "input")
        conversations = load_path(input_path)
        if operation == "audit":
            fail_on = request.get("fail_on", "error")
            if not isinstance(fail_on, str):
                raise ValueError("fail_on must be a severity string")
            report = default_auditor().audit(conversations)
            return {
                "operation": operation,
                "report": report.as_dict(),
                "passed": not report.failing(Severity.parse(fail_on)),
            }
        if operation in {"build", "tabular"}:
            policy_name = request.get("policy", "turn")
            value = request.get("value")
            if not isinstance(policy_name, str) or (
                value is not None and not isinstance(value, int)
            ):
                raise ValueError("policy must be a string and value must be an integer or omitted")
            policy = _service_policy(policy_name, value)
            counter = None
            tokenizer = request.get("tokenizer_plugin")
            if tokenizer is not None:
                if not isinstance(tokenizer, str):
                    raise ValueError("tokenizer_plugin must be a string")
                counter = load_tokenizer(tokenizer)
            targets = request.get("targets")
            if targets is not None and (
                not isinstance(targets, list) or not all(isinstance(item, str) for item in targets)
            ):
                raise ValueError("targets must be an array of strings")
            windows = _build_windows(conversations, policy, targets, counter)
            if operation == "build":
                return {"operation": operation, "windows": json.loads(windows_json(windows))}
            include_text = request.get("include_text", False)
            if not isinstance(include_text, bool):
                raise ValueError("include_text must be a boolean")
            return {
                "operation": operation,
                "rows": list(window_rows(windows, include_text=include_text)),
            }
        if operation == "search":
            query = request.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be a non-empty string")
            limit = request.get("limit", 10)
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError("limit must be a positive integer")
            hits = ConversationSearchIndex(conversations).query(query, limit=limit)
            return {
                "operation": operation,
                "hits": [
                    {
                        "conversation_id": hit.conversation_id,
                        "utterance_id": hit.utterance_id,
                        "score": hit.score,
                        "matched_terms": list(hit.matched_terms),
                    }
                    for hit in hits
                ],
            }
        if operation == "network":
            speaker_field = request.get("speaker_field")
            if speaker_field is not None and (
                not isinstance(speaker_field, str) or not speaker_field.strip()
            ):
                raise ValueError("speaker_field must be a non-empty string when supplied")
            network_report = interaction_network(conversations, speaker_field=speaker_field)
            return {"operation": operation, "network": network_report.to_dict()}
        raise ValueError("operation must be audit, build, tabular, search, or network")


def create_server(
    service: TurnScopeService | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
) -> ThreadingHTTPServer:
    """Create a loopback-first JSON server; call ``serve_forever`` to run it."""
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    target = service or TurnScopeService()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/v1/dispatch":
                self._write(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", "-1"))
                if size < 0 or size > 4 * 1024 * 1024:
                    raise ValueError("Content-Length must be between 0 and 4194304")
                response = target.dispatch(json.loads(self.rfile.read(size).decode("utf-8")))
            except (UnicodeError, json.JSONDecodeError, TypeError, ValueError, OSError) as error:
                self._write(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            self._write(HTTPStatus.OK, response)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _write(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def _required_path(request: Mapping[str, Any], name: str) -> Path:
    value = request.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    return Path(value)


def _service_policy(name: str, value: int | None) -> Any:
    if name == "turn":
        return TurnWindowPolicy(5 if value is None else value)
    if name == "token":
        return TokenBudgetPolicy(512 if value is None else value)
    if name == "time":
        from datetime import timedelta

        return TimeWindowPolicy(timedelta(seconds=3600 if value is None else value))
    if name == "reply-chain":
        return ReplyChainPolicy(value)
    raise ValueError("policy must be turn, token, time, or reply-chain")


def _build_windows(
    conversations: list[Conversation], policy: Any, targets: list[str] | None, counter: Any
) -> tuple[ContextWindow, ...]:
    token_counter = whitespace_tokens if counter is None else counter
    if targets is None:
        return tuple(
            window
            for conversation in conversations
            for window in ContextBuilder(policy, token_counter).build(conversation)
        )
    requested = set(targets)
    occurrences: dict[str, list[str]] = {target: [] for target in requested}
    for conversation in conversations:
        for utterance in conversation.utterances:
            if utterance.id in occurrences:
                occurrences[utterance.id].append(conversation.id)
    missing = sorted(target for target, owners in occurrences.items() if not owners)
    if missing:
        raise KeyError(f"unknown target utterance IDs: {', '.join(missing)}")
    ambiguous = sorted(target for target, owners in occurrences.items() if len(owners) > 1)
    if ambiguous:
        raise ValueError("target utterance IDs are ambiguous: " + ", ".join(ambiguous))
    windows: list[ContextWindow] = []
    for conversation in conversations:
        local = requested & {item.id for item in conversation.utterances}
        if local:
            windows.extend(
                ContextBuilder(policy, token_counter).build(conversation, target_ids=local)
            )
    return tuple(windows)


__all__ = ["TurnScopeService", "create_server"]
