"""Command-line interface for context construction and audit reporting."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

from . import __version__
from .audit import default_auditor
from .builder import ContextBuilder
from .classifier import ConversationClassifier
from .corpus import CorpusStore
from .graph import interaction_network
from .io import DataFormatError, conversation_to_dict, iter_path, load_path
from .models import ContextWindow, Conversation, Severity
from .policies import (
    ReplyChainPolicy,
    TimeWindowPolicy,
    TokenBudgetPolicy,
    TurnWindowPolicy,
    WindowPolicy,
)
from .reporting import report_json, report_markdown, windows_json
from .search import ConversationSearchIndex
from .transformers import corpus_speaker_profiles


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
    corpus = subcommands.add_parser("corpus", help="store and query a disk-backed corpus")
    actions = corpus.add_subparsers(dest="action", required=True)
    ingest = actions.add_parser("import", help="atomically import JSON or JSONL conversations")
    ingest.add_argument("database", type=Path)
    ingest.add_argument("input", type=Path)
    ingest.add_argument("--replace", action="store_true")
    stats = actions.add_parser("stats", help="print conversation and utterance counts")
    stats.add_argument("database", type=Path)
    listing = actions.add_parser("list", help="print one page of conversation IDs")
    listing.add_argument("database", type=Path)
    listing.add_argument("--after")
    listing.add_argument("--limit", type=int, default=100)
    get = actions.add_parser("get", help="print one conversation as JSON")
    get.add_argument("database", type=Path)
    get.add_argument("id")
    search = subcommands.add_parser("search", help="lexically search indexed utterances")
    search.add_argument("input", type=Path)
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--conversation")
    speaker = subcommands.add_parser(
        "speaker-profile", help="aggregate speaker identities across conversations"
    )
    speaker.add_argument("input", type=Path)
    speaker.add_argument("--field", help="metadata field containing a stable speaker identity")
    speaker.add_argument("--output", "-o", type=Path)
    network = subcommands.add_parser(
        "network", help="aggregate reply interactions across a conversation corpus"
    )
    network.add_argument("input", type=Path)
    network.add_argument("--field", help="metadata field containing a stable speaker identity")
    network.add_argument("--output", "-o", type=Path)
    classify = subcommands.add_parser(
        "classify", help="fit and apply a conversation text classifier"
    )
    classify.add_argument("train", type=Path, help="training conversations")
    classify.add_argument("predict", type=Path, help="conversations to classify")
    classify.add_argument(
        "--labels", type=Path, required=True, help="JSON object mapping training IDs to labels"
    )
    classify.add_argument("--model", type=Path, help="save a fitted authenticated model artifact")
    classify.add_argument("--alpha", type=float, default=1.0)
    classify.add_argument("--max-features", type=int)
    classify.add_argument("--output", "-o", type=Path)
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
        if args.command == "corpus":
            return _corpus_command(args)
        if args.command == "classify":
            return _classify_command(args)
        if args.command in {"build", "audit", "speaker-profile", "network"} and _paths_collide(
            args.input, args.output
        ):
            raise ValueError("output path must differ from the input path")
        conversations = load_path(args.input)
        if args.command == "search":
            hits = ConversationSearchIndex(conversations).query(
                args.query,
                limit=args.limit,
                conversation_id=args.conversation,
            )
            print(
                json.dumps(
                    [
                        {
                            "conversation_id": hit.conversation_id,
                            "utterance_id": hit.utterance_id,
                            "score": hit.score,
                            "matched_terms": list(hit.matched_terms),
                        }
                        for hit in hits
                    ],
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
            return 0
        if args.command == "speaker-profile":
            profiles = corpus_speaker_profiles(conversations, field=args.field)
            rendered = (
                json.dumps(
                    [
                        {
                            "speaker": profile.speaker,
                            "conversations": profile.conversations,
                            "utterances": profile.utterances,
                            "tokens": profile.tokens,
                            "unique_tokens": profile.unique_tokens,
                            "roles": dict(profile.roles),
                            "replies_sent": profile.replies_sent,
                            "replies_received": profile.replies_received,
                        }
                        for profile in profiles
                    ],
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n"
            )
            _write(rendered, args.output)
            return 0
        if args.command == "network":
            network_report = interaction_network(conversations, speaker_field=args.field)
            _write(
                json.dumps(
                    network_report.to_dict(), ensure_ascii=True, allow_nan=False, sort_keys=True
                )
                + "\n",
                args.output,
            )
            return 0
        if args.command == "build":
            policy = _policy(args.policy, args.value)
            windows = _build_windows(conversations, policy, args.target)
            _write(windows_json(windows), args.output)
            return 0
        report = default_auditor(token_budget=args.token_budget).audit(conversations)
        rendered = report_json(report) if args.format == "json" else report_markdown(report)
        _write(rendered, args.output)
        return 1 if report.failing(Severity.parse(args.fail_on)) else 0
    except (DataFormatError, OSError, OverflowError, ValueError, KeyError, sqlite3.Error) as error:
        print(f"turnscope: error: {error}", file=sys.stderr)
        return 2


def _corpus_command(args: argparse.Namespace) -> int:
    if args.action == "import":
        if _paths_collide(args.input, args.database):
            raise ValueError("database path must differ from the input path")
        if not args.input.is_file():
            raise ValueError("input must be an existing file")
    elif not args.database.is_file():
        raise ValueError("database must be an existing file")
    with CorpusStore(args.database) as store:
        if args.action == "import":
            result: object = {"imported": store.put(iter_path(args.input), replace=args.replace)}
        elif args.action == "stats":
            conversations, utterances = store.counts()
            result = {"conversations": conversations, "utterances": utterances}
        elif args.action == "list":
            result = store.ids(after=args.after, limit=args.limit)
        else:
            result = conversation_to_dict(store.get(args.id))
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


def _classify_command(args: argparse.Namespace) -> int:
    """Fit a transparent model, persist it optionally, and emit predictions."""

    for left, right, message in (
        (args.train, args.predict, "training and prediction inputs must differ"),
        (args.output, args.train, "output path must differ from training input"),
        (args.output, args.predict, "output path must differ from prediction input"),
        (args.model, args.train, "model path must differ from training input"),
        (args.model, args.predict, "model path must differ from prediction input"),
    ):
        if left is not None and _paths_collide(left, right):
            raise ValueError(message)
    try:
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load labels: {error}") from error
    if not isinstance(labels, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value.strip()
        for key, value in labels.items()
    ):
        raise ValueError("labels must be a JSON object of non-empty string values")
    training = load_path(args.train)
    model = ConversationClassifier(alpha=args.alpha, max_features=args.max_features).fit(
        training, labels
    )
    if args.model is not None:
        model.save(args.model)
    predictions = []
    for conversation in load_path(args.predict):
        probabilities = model.predict_proba(conversation)
        predictions.append(
            {
                "id": conversation.id,
                "label": model.predict(conversation),
                "probabilities": dict(probabilities),
            }
        )
    _write(
        json.dumps(
            {"model_digest": model.digest(), "predictions": predictions},
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
        args.output,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
