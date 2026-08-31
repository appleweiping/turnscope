"""Generate deterministic native TurnScope JSONL benchmark data."""

from __future__ import annotations

import argparse
import random
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

from turnscope.io import dump_conversations
from turnscope.models import Conversation, Utterance
from turnscope.policies import WhitespaceTokenCounter

_WORDS = ("alpha", "beta", "context", "delta", "evaluation", "model", "prompt", "token")


def conversations(count: int, utterances: int, seed: int) -> Iterator[Conversation]:
    """Yield a reproducible corpus without retaining all conversations."""
    generator = random.Random(seed)
    counter = WhitespaceTokenCounter()
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for conversation_index in range(count):
        items: list[Utterance] = []
        for message_index in range(utterances):
            text = " ".join(generator.choice(_WORDS) for _ in range(4 + message_index % 5))
            items.append(
                Utterance(
                    id=f"m{message_index:07d}",
                    role="user" if message_index % 2 == 0 else "assistant",
                    text=text,
                    timestamp=epoch
                    + timedelta(seconds=conversation_index * utterances + message_index),
                    reply_to=f"m{message_index - 1:07d}" if message_index else None,
                    token_count=counter(text),
                )
            )
        yield Conversation(
            f"c{conversation_index:06d}",
            items,
            {"generator": "turnscope/0.2", "seed": seed},
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--conversations", type=int, default=100)
    parser.add_argument("--utterances", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.conversations < 0 or args.utterances < 0:
        raise SystemExit("--conversations and --utterances must be non-negative")
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {args.output}; pass --force")
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        dump_conversations(
            conversations(args.conversations, args.utterances, args.seed),
            stream,
            format="jsonl",
        )
    records = args.conversations * args.utterances
    print(f"wrote {args.conversations} conversations and {records} utterances to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
