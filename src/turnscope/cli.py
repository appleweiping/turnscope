"""Command-line interface for context construction and audit reporting."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

from . import __version__
from .audit import default_auditor
from .builder import ContextBuilder
from .io import DataFormatError, load_path
from .models import ContextWindow, Conversation, Severity
from .policies import (
    ReplyChainPolicy,
    TimeWindowPolicy,
    TokenBudgetPolicy,
    TurnWindowPolicy,
    WindowPolicy,
)
from .reporting import report_json, report_markdown, windows_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="turnscope", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("build", help="build context windows")
    build.add_argument("input", type=Path)
    build.add_argument("--output", "-o", type=Path)
    build.add_argument("--policy", choices=("turn", "token", "time", "reply-chain"), default="turn")
    build.add_argument("--value", type=int, help="turns, tokens, seconds, or reply depth")
    build.add_argument("--target", action="append", help="only build the specified target ID")

    audit = subcommands.add_parser("audit", help="audit conversation reliability")
    audit.add_argument("input", type=Path)
    audit.add_argument("--output", "-o", type=Path)
    audit.add_argument("--format", choices=("json", "markdown"), default="markdown")
    audit.add_argument("--token-budget", type=int)
    audit.add_argument("--fail-on", choices=("info", "warning", "error"), default="error")
    return parser


def _policy(name: str, value: int | None) -> WindowPolicy:
    if name == "turn":
        return TurnWindowPolicy(5 if value is None else value)
    if name == "token":
        return TokenBudgetPolicy(512 if value is None else value)
    if name == "time":
        return TimeWindowPolicy(timedelta(seconds=3600 if value is None else value))
    return ReplyChainPolicy(value)


def _write(text: str, output: Path | None) -> None:
    if output is None:
        sys.stdout.write(text)
    else:
        output.write_text(text, encoding="utf-8")


def _paths_collide(input_path: Path, output_path: Path | None) -> bool:
    if output_path is None:
        return False
    try:
        if input_path.resolve() == output_path.resolve():
            return True
        return input_path.exists() and output_path.exists() and input_path.samefile(output_path)
    except (OSError, RuntimeError):
        return False


def _build_windows(
    conversations: list[Conversation], policy: WindowPolicy, target_ids: Sequence[str] | None
) -> tuple[ContextWindow, ...]:
    """Build windows, interpreting CLI target IDs across the complete dataset."""
    if target_ids is None:
        return tuple(
            window
            for conversation in conversations
            for window in ContextBuilder(policy).build(conversation)
        )

    requested = set(target_ids)
    occurrences: dict[str, list[str]] = {target_id: [] for target_id in requested}
    for conversation in conversations:
        for utterance in conversation.utterances:
            if utterance.id in occurrences:
                occurrences[utterance.id].append(conversation.id)

    missing = sorted(target_id for target_id, owners in occurrences.items() if not owners)
    if missing:
        raise KeyError(f"unknown target utterance IDs: {', '.join(missing)}")
    ambiguous = sorted(target_id for target_id, owners in occurrences.items() if len(owners) > 1)
    if ambiguous:
        raise ValueError(
            "target utterance IDs are ambiguous across the dataset: " + ", ".join(ambiguous)
        )

    windows: list[ContextWindow] = []
    for conversation in conversations:
        local_ids = requested & {item.id for item in conversation.utterances}
        if local_ids:
            windows.extend(ContextBuilder(policy).build(conversation, target_ids=local_ids))
    return tuple(windows)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if _paths_collide(args.input, args.output):
            raise ValueError("output path must differ from the input path")
        conversations = load_path(args.input)
        if args.command == "build":
            policy = _policy(args.policy, args.value)
            windows = _build_windows(conversations, policy, args.target)
            _write(windows_json(windows), args.output)
            return 0
        report = default_auditor(token_budget=args.token_budget).audit(conversations)
        rendered = report_json(report) if args.format == "json" else report_markdown(report)
        _write(rendered, args.output)
        return 1 if report.failing(Severity.parse(args.fail_on)) else 0
    except (DataFormatError, OSError, OverflowError, ValueError, KeyError) as error:
        print(f"turnscope: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
