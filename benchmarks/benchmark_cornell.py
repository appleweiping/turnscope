"""Benchmark frozen conversation vectors on a locally supplied Cornell Movie-Dialogs archive."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

from benchmark_vectors import measure

from turnscope import Conversation, Utterance

_ARCHIVE_SHA256 = "3bde8a571f615201bc2d2453e22878090719638592f774720eddec739de8c900"
_PREFIX = "cornell movie-dialogs corpus/"
_DELIMITER = " +++$+++ "


def load_sample(
    archive: Path, training_count: int, query_count: int
) -> tuple[list[Conversation], list[Conversation], dict[str, object]]:
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != _ARCHIVE_SHA256:
        raise ValueError("archive SHA-256 does not match the pinned Cornell release")
    selected: dict[str, list[tuple[int, str, list[str]]]] = {"training": [], "queries": []}
    limits = {"training": training_count, "queries": query_count}
    with ZipFile(archive) as source:
        with source.open(_PREFIX + "movie_conversations.txt") as raw:
            for position, row in enumerate(io.TextIOWrapper(raw, encoding="latin-1"), 1):
                fields = row.rstrip("\r\n").split(_DELIMITER)
                if len(fields) != 4:
                    raise ValueError(f"invalid conversation fields at line {position}")
                movie_id = fields[2]
                # Whole films, including their speakers and dialogues, remain in one fold.
                bucket = int(hashlib.sha256(movie_id.encode("utf-8")).hexdigest(), 16) % 10
                partition = "training" if bucket < 8 else "queries"
                if len(selected[partition]) >= limits[partition]:
                    continue
                identifiers = ast.literal_eval(fields[3])
                if (
                    not isinstance(identifiers, list)
                    or not identifiers
                    or not all(isinstance(value, str) and value for value in identifiers)
                    or len(set(identifiers)) != len(identifiers)
                ):
                    raise ValueError(f"invalid utterance IDs at conversation line {position}")
                selected[partition].append((position, movie_id, identifiers))
                if all(len(selected[key]) == limits[key] for key in selected):
                    break
        if any(len(selected[key]) != limits[key] for key in selected):
            raise ValueError("requested sample exceeds the available movie-disjoint partitions")
        wanted = {
            identifier
            for rows in selected.values()
            for _, _, identifiers in rows
            for identifier in identifiers
        }
        lines: dict[str, tuple[str, str, str]] = {}
        with source.open(_PREFIX + "movie_lines.txt") as raw:
            for position, row in enumerate(io.TextIOWrapper(raw, encoding="latin-1"), 1):
                fields = row.rstrip("\r\n").split(_DELIMITER, 4)
                if len(fields) != 5:
                    raise ValueError(f"invalid movie line fields at line {position}")
                identifier = fields[0]
                if identifier not in wanted:
                    continue
                if identifier in lines:
                    raise ValueError(f"duplicate selected line ID at line {position}")
                lines[identifier] = (fields[1], fields[2], fields[4])
    if set(lines) != wanted:
        raise ValueError("selected conversations refer to missing movie lines")
    timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc)
    converted: dict[str, list[Conversation]] = {key: [] for key in selected}
    movie_sets = {key: {movie for _, movie, _ in rows} for key, rows in selected.items()}
    if movie_sets["training"] & movie_sets["queries"]:
        raise RuntimeError("movie split leakage")
    for partition, rows in selected.items():
        for position, movie_id, identifiers in rows:
            utterances = []
            for identifier in identifiers:
                speaker, movie, text = lines[identifier]
                if movie != movie_id:
                    raise ValueError("conversation and movie-line film IDs disagree")
                utterances.append(
                    Utterance(identifier, speaker, text, timestamp, metadata={"speaker": speaker})
                )
            converted[partition].append(Conversation(f"cornell-{position}", utterances))
    selection = {
        partition: [position for position, _, _ in rows] for partition, rows in selected.items()
    }
    metadata: dict[str, object] = {
        "archive_sha256": digest.hexdigest(),
        "selection_sha256": hashlib.sha256(
            json.dumps(selection, sort_keys=True).encode()
        ).hexdigest(),
        "split": "sha256(movie_id) mod 10 < 8 trains; otherwise held out; first records per fold",
        "training_movies": len(movie_sets["training"]),
        "query_movies": len(movie_sets["queries"]),
        "movie_overlap": 0,
        "training_utterances": sum(len(item.utterances) for item in converted["training"]),
        "query_utterances": sum(len(item.utterances) for item in converted["queries"]),
        "reply_coordination_evaluated": False,
        "reply_coordination_note": "source has line ordering but no explicit reply-tree edges",
    }
    return converted["training"], converted["queries"], metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--training", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.training < 1 or args.queries < 1:
        parser.error("training and query counts must be positive")
    if args.archive.resolve() == args.output.resolve() or (
        args.output.exists() and args.archive.samefile(args.output)
    ):
        parser.error("output must differ from the archive")
    training, queries, metadata = load_sample(args.archive, args.training, args.queries)
    root = Path(__file__).resolve().parents[1]
    result = {
        "kind": "external-corpus-subset",
        "dataset": "Cornell Movie-Dialogs Corpus",
        "dataset_source": "https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html",
        "task": "frozen TF-IDF projection and lexical cosine retrieval correctness",
        "retrieval_quality_evaluated": False,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **metadata,
        **measure(training, queries),
        "source_sha256": {
            str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted((root / "src/turnscope").glob("*.py"))
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
