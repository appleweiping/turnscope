"""Movie-disjoint engineering evaluation of an explicit next-in-sequence context predictor."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import sys
import time
import tracemalloc
from pathlib import Path

from benchmark_cornell import load_sample

from turnscope import ExpectedContextModel, iter_context_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--training", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--max-features", type=int, default=256)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.training <= 5000 or not 1 <= args.queries <= 5000:
        parser.error("training and queries must each be in [1, 5000]")
    if args.archive.resolve() == args.output.resolve() or (
        args.output.exists() and args.archive.samefile(args.output)
    ):
        parser.error("output must differ from the archive")
    training, queries, metadata = load_sample(args.archive, args.training, args.queries)
    # Cornell's ordering is explicit, but does not assert authentic reply-to edges.
    model = ExpectedContextModel(
        relation="sequence-successor",
        n_components=args.components,
        max_features=args.max_features,
        regularization=args.regularization,
    )
    tracemalloc.start()
    started = time.perf_counter()
    model.fit(training)
    fit_seconds = time.perf_counter() - started
    _, fit_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    artifact = model.to_dict()
    restored = ExpectedContextModel.from_dict(artifact)
    tracemalloc.start()
    started = time.perf_counter()
    scores = restored.evaluate(queries)
    evaluation_seconds = time.perf_counter() - started
    _, evaluation_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if restored.to_dict() != artifact:
        raise RuntimeError("heldout evaluation changed the frozen model")
    # Independently aggregate vector-coordinate losses, including the fixed mean baseline.
    prediction_error, mean_error, count = 0.0, 0.0, 0
    source_tokens = context_tokens = known_source_tokens = known_context_tokens = 0
    for pair in iter_context_pairs(queries, relation="sequence-successor"):
        predicted = model.predict(pair.source.text)
        actual = model.project_context(pair.context.text)
        if predicted != restored.predict(pair.source.text):
            raise RuntimeError("artifact roundtrip changed a heldout prediction")
        for dimension in range(model.state.dimensions):
            difference = predicted.vector[dimension] - actual.vector[dimension]
            prediction_error += difference * difference
            difference = model.state.context_mean[dimension] - actual.vector[dimension]
            mean_error += difference * difference
        source_tokens += predicted.tokens
        known_source_tokens += predicted.known_tokens
        context_tokens += actual.tokens
        known_context_tokens += actual.known_tokens
        count += 1
    if not math.isclose(
        prediction_error / (count * model.state.dimensions), scores["mse"], abs_tol=1e-12
    ) or not math.isclose(
        mean_error / (count * model.state.dimensions),
        scores["training_mean_baseline_mse"],
        abs_tol=1e-12,
    ):
        raise RuntimeError("evaluation does not match independently accumulated errors")
    root = Path(__file__).resolve().parents[1]
    result = {
        "kind": "external-corpus-heldout-engineering-evaluation",
        "dataset": "Cornell Movie-Dialogs Corpus",
        "dataset_source": "https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html",
        "task": "predict the next listed utterance in a frozen TF-IDF/SVD context space",
        "relation": "sequence-successor",
        "authentic_reply_tree_evaluated": False,
        "gold_annotation_accuracy_evaluated": False,
        "model_selection_on_holdout": False,
        "hyperparameters": {
            "n_components": args.components,
            "max_features": args.max_features,
            "regularization": args.regularization,
        },
        "python": sys.version.split()[0],
        "numpy": importlib.import_module("numpy").__version__,
        "platform": platform.platform(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "loader_sha256": hashlib.sha256(
            Path(__file__).with_name("benchmark_cornell.py").read_bytes()
        ).hexdigest(),
        **metadata,
        "training_pairs": model.state.training_pairs,
        "training_source_documents": model.state.source_tfidf.documents,
        "training_context_documents": model.state.context_tfidf.documents,
        "source_features": len(model.state.source_tfidf.vocabulary),
        "context_features": len(model.state.context_tfidf.vocabulary),
        "retained_dimensions": model.state.dimensions,
        "training_mse": model.state.training_mse,
        "training_mean_baseline_mse": model.state.mean_baseline_mse,
        "heldout": scores,
        "heldout_source_token_coverage": known_source_tokens / source_tokens,
        "heldout_context_token_coverage": known_context_tokens / context_tokens,
        "heldout_loss_oracle_pairs": count,
        "heldout_roundtrip_prediction_checks": count,
        "frozen_artifact_unchanged": True,
        "model_sha256": model.digest,
        "estimated_dense_cells": model.state.estimated_dense_cells,
        "fit_seconds": fit_seconds,
        "fit_tracemalloc_peak_bytes": fit_peak,
        "evaluation_seconds": evaluation_seconds,
        "evaluation_tracemalloc_peak_bytes": evaluation_peak,
        "measurement_scope": (
            "fit and evaluate separately; excludes parsing, JSON roundtrip and independent checks; "
            "tracemalloc is not process RSS"
        ),
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
