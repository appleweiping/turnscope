"""Stable JSON and readable Markdown representations."""

from __future__ import annotations

import json

from .io import utterance_to_dict
from .models import AuditReport, ContextWindow


def report_json(report: AuditReport, *, indent: int | None = 2) -> str:
    return json.dumps(report.as_dict(), ensure_ascii=False, allow_nan=False, indent=indent) + "\n"


def _markdown_cell(value: object) -> str:
    """Escape untrusted text for a GitHub-flavored Markdown table cell."""
    return (
        str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    )


def report_markdown(report: AuditReport) -> str:
    counts = report.counts()
    lines = [
        "# TurnScope audit",
        "",
        f"Audited **{report.conversations}** conversations and **{report.utterances}** utterances.",
        "",
        f"Findings: **{len(report.issues)}** "
        f"({counts['error']} errors, {counts['warning']} warnings, {counts['info']} info).",
        "",
    ]
    if not report.issues:
        return "\n".join([*lines, "No reliability issues found.", ""])
    lines.extend(
        [
            "| Severity | Rule | Conversation | Utterances | Message |",
            "|---|---|---|---|---|",
        ]
    )
    for issue in report.issues:
        values = _markdown_cell(", ".join(issue.utterance_ids) or "—")
        conversation = _markdown_cell(issue.conversation_id)
        message = _markdown_cell(issue.message)
        lines.append(
            f"| {_markdown_cell(issue.severity.name.lower())} | {_markdown_cell(issue.rule)} "
            f"| {conversation} | {values} | {message} |"
        )
    return "\n".join(lines) + "\n"


def windows_json(windows: tuple[ContextWindow, ...], *, indent: int | None = 2) -> str:
    values = [
        {
            "conversation_id": item.conversation_id,
            "target": utterance_to_dict(item.target),
            "context": [utterance_to_dict(value) for value in item.context],
            "policy": item.policy,
            "token_total": item.token_total,
        }
        for item in windows
    ]
    return json.dumps(values, ensure_ascii=False, allow_nan=False, indent=indent) + "\n"
