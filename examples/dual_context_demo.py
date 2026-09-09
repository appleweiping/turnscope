"""Original small fit/save/reload/retrieval CLI demonstration, not a quality benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir()
    training = []
    edges = []
    for identifier, word in (("a", "alpha"), ("b", "beta"), ("d", "gamma")):
        training.extend(
            {"conversation_id": "authored-training", "utterance_id": name, "text": word}
            for name in (identifier, "c" + identifier)
        )
        edges.append(
            {
                "conversation_id": "authored-training",
                "source_id": identifier,
                "context_id": "c" + identifier,
            }
        )
    write_rows(root / "training.jsonl", training)
    write_rows(root / "forward.jsonl", edges)
    write_rows(root / "backward.jsonl", edges)
    queries = [
        {"conversation_id": "authored-new", "utterance_id": name, "text": text}
        for name, text in (("query", "alpha"), ("unknown", "unseenword"))
    ]
    candidates = [
        {"conversation_id": "authored-new", "utterance_id": name, "text": text}
        for name, text in (("a", "beta"), ("z", "alpha"))
    ]
    gold = [
        {"conversation_id": "authored-new", "source_id": item["utterance_id"], "context_id": "z"}
        for item in queries
    ]
    write_rows(root / "queries.jsonl", queries)
    write_rows(root / "candidates.jsonl", candidates)
    write_rows(root / "relevance.jsonl", gold)

    def invoke(*arguments: str) -> dict[str, object] | None:
        result = subprocess.run(
            [sys.executable, "-m", "turnscope", "dual-context", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=60,
        )
        return json.loads(result.stdout) if result.stdout.strip() else None

    fit = invoke(
        "fit",
        "training.jsonl",
        "forward.jsonl",
        "backward.jsonl",
        "model.json",
        "--n-components",
        "3",
        "--n-clusters",
        "2",
    )
    model_bytes = (root / "model.json").read_bytes()
    invoke("transform", "model.json", "queries.jsonl", "--output", "predictions.json")
    invoke("terms", "model.json", "--output", "terms.json")
    invoke(
        "evaluate",
        "model.json",
        "queries.jsonl",
        "candidates.jsonl",
        "relevance.jsonl",
        "--direction",
        "forward",
        "--output",
        "evaluation.json",
    )
    assert model_bytes == (root / "model.json").read_bytes()
    report = json.loads((root / "evaluation.json").read_bytes())
    assert report["summary"]["all_gold_queries"]["mean_reciprocal_rank"] == 0.5
    assert report["summary"]["scorable_gold_queries"]["mean_reciprocal_rank"] == 1.0
    print(
        json.dumps(
            {
                "model_unchanged": True,
                "fit": fit,
                "summary": report["summary"],
                "limits": "Original engineering fixture; no semantic quality claim",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
