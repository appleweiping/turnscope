"""Pinned CGA-WIKI human-event prefix forecasting on untouched official test groups."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

from turnscope import Conversation, PrefixEventForecaster, Utterance

ARCHIVE_SHA256 = "84e2d1ac60a3269251b5e175e549fc65cec875fdc926f99172b5b70d3ca1b122"
PREFIX = "conversations-gone-awry-corpus/"


def load_cga(archive: Path) -> tuple[dict[str, list[Conversation]], dict[str, object]]:
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("CGA-WIKI archive does not match the pinned SHA-256")
    with ZipFile(archive) as source:
        metadata_bytes = source.read(PREFIX + "conversations.json")
        metadata = json.loads(metadata_bytes)
        records: dict[str, list[Utterance]] = defaultdict(list)
        utterance_ids: set[str] = set()
        headers = 0
        with source.open(PREFIX + "utterances.jsonl") as stream:
            for row in io.TextIOWrapper(stream, encoding="utf-8"):
                value = json.loads(row)
                identifier, conversation_id = value["id"], value["conversation_id"]
                if identifier in utterance_ids or conversation_id not in metadata:
                    raise ValueError("duplicate utterance or unknown conversation")
                utterance_ids.add(identifier)
                meta = value["meta"]
                if (
                    type(meta["is_section_header"]) is not bool
                    or type(meta["comment_has_personal_attack"]) is not bool
                ):
                    raise ValueError("CGA-WIKI human labels must be boolean")
                headers += meta["is_section_header"]
                # Exclude parses/toxicity predictions; only supplied human event labels supervise.
                records[conversation_id].append(
                    Utterance(
                        identifier,
                        "speaker",
                        value["text"],
                        datetime.fromtimestamp(value["timestamp"], timezone.utc),
                        metadata={
                            "event": meta["comment_has_personal_attack"],
                            "is_section_header": meta["is_section_header"],
                        },
                    )
                )
    if set(records) != set(metadata):
        raise ValueError("metadata and observed conversation IDs differ")
    partitions: dict[str, list[Conversation]] = {"train": [], "val": [], "test": []}
    group_splits: dict[str, set[str]] = defaultdict(set)
    labels: Counter[str] = Counter()
    for identifier, meta in sorted(metadata.items()):
        partner = meta["pair_id"]
        split = meta["split"]
        if (
            partner not in metadata
            or metadata[partner]["pair_id"] != identifier
            or split not in partitions
        ):
            raise ValueError("invalid reciprocal pair or official split")
        groups = ["page:" + str(meta["page_id"]), "pair:" + min(identifier, partner)]
        for group in groups:
            group_splits[group].add(split)
        items = sorted(records[identifier], key=lambda item: (item.timestamp, item.id))
        annotated = any(
            item.metadata["event"] for item in items if not item.metadata["is_section_header"]
        )
        if (
            type(meta["conversation_has_personal_attack"]) is not bool
            or annotated != meta["conversation_has_personal_attack"]
        ):
            raise ValueError("conversation label disagrees with human utterance annotations")
        labels[split + ":" + str(annotated)] += 1
        partitions[split].append(
            Conversation(identifier, items, metadata={"forecast_groups": groups})
        )
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("official page/pair groups overlap folds")
    return partitions, {
        "archive_sha256": digest.hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "raw_conversations": len(metadata),
        "raw_utterances": len(utterance_ids),
        "section_headers": headers,
        "raw_label_counts_by_split": dict(sorted(labels.items())),
        "page_pair_groups": len(group_splits),
        "cross_split_page_pair_groups": 0,
        "split": "official train/val/test; assert reciprocal pairs and page isolation",
        "ordering": (
            "ascending timestamp and ID; equal-time blocks never split by prediction boundaries"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.archive.resolve() == args.output.resolve() or (
        args.output.exists() and args.archive.samefile(args.output)
    ):
        parser.error("output must differ from source archive")
    partitions, metadata = load_cga(args.archive)
    # Fixed defaults; final test has never participated in parameter or threshold selection.
    tracemalloc.start()
    started = time.perf_counter()
    model = PrefixEventForecaster().fit(partitions["train"], partitions["val"])
    fit_seconds = time.perf_counter() - started
    _, fit_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    artifact = model.to_dict()
    frozen = PrefixEventForecaster.from_dict(artifact)
    tracemalloc.start()
    started = time.perf_counter()
    test_result = frozen.evaluate(partitions["test"])
    evaluation_seconds = time.perf_counter() - started
    _, evaluation_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if frozen.to_dict() != artifact:
        raise RuntimeError("test evaluation changed frozen parameters")
    root = Path(__file__).resolve().parents[1]
    result = {
        "kind": "human-annotated-future-event-heldout-evaluation",
        "dataset": "CGA-WIKI",
        "dataset_source": "https://convokit.cornell.edu/documentation/awry.html",
        "task": "first human-annotated personal attack later within the observed record",
        "model": "conversation-weighted cumulative multinomial Naive Bayes baseline",
        "hyperparameters": artifact["config"],
        "model_digest": model.digest,
        "training_conversations": model.state.training_conversations,
        "training_prefixes": model.state.training_prefixes,
        "validation_conversations": model.state.validation_conversations,
        "validation_prefixes": model.state.validation_prefixes,
        "validation_balanced_accuracy": model.state.validation_balanced_accuracy,
        "vocabulary_size": len(model.state.vocabulary),
        "threshold": model.state.threshold,
        "threshold_selection": (
            "validation conversation-level any-alert balanced accuracy; higher-threshold tie break"
        ),
        "final_test_used_for_tuning": False,
        "frozen_artifact_unchanged": True,
        "test": test_result,
        "fit_seconds": fit_seconds,
        "fit_tracemalloc_peak_bytes": fit_peak,
        "evaluation_seconds": evaluation_seconds,
        "evaluation_tracemalloc_peak_bytes": evaluation_peak,
        "measurement_scope": (
            "fit includes validation threshold selection; excludes parsing and serialization; "
            "tracemalloc is not RSS"
        ),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted((root / "src/turnscope").glob("*.py"))
        },
        **metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
