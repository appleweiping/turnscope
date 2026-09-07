"""Benchmark TurnScope on the checked-in human-authored conversation fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

from turnscope import (
    CallableTransformer,
    ConversationSearchIndex,
    FeaturePipeline,
    TfidfVectorizer,
    conversation_features,
)
from turnscope.io import load_path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "examples" / "conversations.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "benchmarks/results/fixture.json")
    args = parser.parse_args()
    payload = source.read_bytes()
    conversations = load_path(source)
    tracemalloc.start()
    started = time.perf_counter()
    index = ConversationSearchIndex(conversations)
    hits = index.query("fails", limit=10)
    pipeline = FeaturePipeline(
        (
            ("tfidf", TfidfVectorizer()),
            ("counts", CallableTransformer(conversation_features)),
        )
    )
    features = pipeline.fit_transform(conversations)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = {
        "kind": "fixture-real",
        "source": str(source.relative_to(root)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_bytes": len(payload),
        "conversations": len(conversations),
        "utterances": sum(len(item.utterances) for item in conversations),
        "indexed_documents": index.documents,
        "search_hits": len(hits),
        "feature_records": len(features),
        "elapsed_seconds": elapsed,
        "peak_python_bytes": peak,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
