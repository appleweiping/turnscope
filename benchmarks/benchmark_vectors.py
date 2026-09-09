"""Reproduce fitted sparse projection and retrieval on a fixture and synthetic corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

from turnscope import Conversation, SparseSimilarityIndex, TfidfVectorizer, Utterance, sparse_cosine
from turnscope.io import load_path


def measure(training: list[Conversation], queries: list[Conversation]) -> dict[str, object]:
    tracemalloc.start()
    started = time.perf_counter()
    model = TfidfVectorizer().fit(iter(training))
    artifact = model.to_dict()
    model = TfidfVectorizer.from_dict(artifact)
    vectors = model.transform_corpus(training)
    index = SparseSimilarityIndex(vectors)
    query_vectors = model.transform_corpus(queries)
    results = {key: index.query(vector, limit=5) for key, vector in query_vectors.items()}
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Full enumeration is separate from the timed sparse candidate retrieval.
    # Compare IDs and scores to ensure the index did not drop valid candidates.
    oracle_checks = 0
    for key, vector in list(query_vectors.items())[:5]:
        expected = sorted(
            (
                (identifier, sparse_cosine(vector, candidate))
                for identifier, candidate in vectors.items()
            ),
            key=lambda pair: (-pair[1], pair[0]),
        )
        expected = [pair for pair in expected if pair[1] > 0][:5]
        actual = [(hit.id, hit.score) for hit in results[key]]
        if len(actual) != len(expected) or any(
            left[0] != right[0] or abs(left[1] - right[1]) > 1e-12
            for left, right in zip(actual, expected, strict=True)
        ):
            raise RuntimeError("inverted retrieval differs from exhaustive cosine")
        oracle_checks += 1
    if artifact != model.to_dict():
        raise RuntimeError("held-out projection mutated the fitted model")
    return {
        "training_conversations": len(training),
        "query_conversations": len(queries),
        "vocabulary": len(model.state.vocabulary),
        "candidate_nonzeros": sum(len(vector) for vector in vectors.values()),
        "query_nonzeros": sum(len(vector) for vector in query_vectors.values()),
        "matches": sum(len(matches) for matches in results.values()),
        "exhaustive_oracle_queries": oracle_checks,
        "frozen_model_verified": True,
        "elapsed_seconds": elapsed,
        "peak_python_bytes": peak,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--output", type=Path, default=root / "benchmarks/results/vectors.json")
    args = parser.parse_args()
    if args.training < 1 or args.queries < 1:
        parser.error("training and queries must be positive")
    generator = random.Random(args.seed)
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def synthetic(identifier: str, *, heldout: bool = False) -> Conversation:
        text = " ".join(f"term{generator.randrange(1000)}" for _ in range(32))
        if heldout:
            text += " heldout-only"
        return Conversation(identifier, [Utterance("u", "user", text, timestamp)])

    fixture_path = root / "examples/conversations.json"
    fixture = load_path(fixture_path)
    fixture_report = measure(fixture, fixture)
    result = {
        "fixture": {
            "kind": "checked-in-human-authored-fixture",
            "evaluation": "self-retrieval-smoke-only",
            "source": "examples/conversations.json",
            "sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            **fixture_report,
        },
        "synthetic": {
            "seed": args.seed,
            "terms": 1000,
            "tokens_per_training_conversation": 32,
            **measure(
                [synthetic(f"train-{index}") for index in range(args.training)],
                [synthetic(f"query-{index}", heldout=True) for index in range(args.queries)],
            ),
        },
        "source_sha256": {
            str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted((root / "src/turnscope").glob("*.py"))
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
