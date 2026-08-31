"""Measure output-sensitive context construction on deterministic messages."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from turnscope.builder import ContextBuilder
from turnscope.models import Utterance
from turnscope.policies import (
    ReplyChainPolicy,
    TimeWindowPolicy,
    TokenBudgetPolicy,
    TurnWindowPolicy,
    WindowPolicy,
)


def utterances(size: int) -> tuple[Utterance, ...]:
    """Create a fixed-width, chronological reply chain."""
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return tuple(
        Utterance(
            id=f"m{index}",
            role="user" if index % 2 == 0 else "assistant",
            text=f"benchmark message number {index}",
            timestamp=epoch + timedelta(seconds=index),
            reply_to=f"m{index - 1}" if index else None,
            token_count=4,
        )
        for index in range(size)
    )


def policy(name: str, value: int) -> WindowPolicy:
    if name == "turn":
        return TurnWindowPolicy(value)
    if name == "token":
        return TokenBudgetPolicy(value)
    if name == "time":
        return TimeWindowPolicy(timedelta(seconds=value))
    return ReplyChainPolicy(value)


def run_once(items: tuple[Utterance, ...], selected_policy: WindowPolicy) -> int:
    """Consume every lazy result and return a stable anti-elision checksum."""
    return sum(
        window.token_total + len(window.context)
        for window in ContextBuilder(selected_policy).iter_build("benchmark", items)
    )


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", choices=("turn", "token", "time", "reply-chain"), default="turn"
    )
    parser.add_argument("--value", type=_positive, default=8)
    parser.add_argument("--sizes", type=_positive, nargs="+", default=(1_000, 10_000, 100_000))
    parser.add_argument("--repeats", type=_positive, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selected_policy = policy(args.policy, args.value)
    results: list[dict[str, int | float]] = []
    for size in args.sizes:
        items = utterances(size)
        run_once(items, selected_policy)
        durations: list[float] = []
        checksum = 0
        for _ in range(args.repeats):
            started = time.perf_counter()
            checksum = run_once(items, selected_policy)
            durations.append(time.perf_counter() - started)
        median = statistics.median(durations)
        results.append(
            {
                "utterances": size,
                "median_seconds": round(median, 6),
                "messages_per_second": round(size / median),
                "checksum": checksum,
            }
        )
    print(
        json.dumps(
            {
                "schema": "turnscope-benchmark/1",
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "policy": selected_policy.name,
                "repeats": args.repeats,
                "results": results,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
