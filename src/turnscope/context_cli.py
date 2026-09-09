"""CLI for frozen SVD-plus-ridge context fitting and heldout evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .expected_context import ExpectedContextModel
from .io import iter_path


def configure_context_parser(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="context_action", required=True)
    fit = actions.add_parser("fit", help="fit a context basis and ridge predictor on training data")
    fit.add_argument("input", type=Path)
    fit.add_argument("model", type=Path)
    fit.add_argument(
        "--relation", choices=("reply", "predecessor", "sequence-successor"), default="reply"
    )
    fit.add_argument("--components", type=int, default=16)
    fit.add_argument("--regularization", type=float, default=1.0)
    fit.add_argument("--min-df", type=int, default=1)
    fit.add_argument("--max-features", type=int, default=256)
    fit.add_argument("--max-pairs", type=int, default=5000)
    fit.add_argument("--max-dense-cells", type=int, default=16_000_000)
    for command in ("transform", "evaluate"):
        action = actions.add_parser(command, help=f"{command} using a frozen context model")
        action.add_argument("model", type=Path)
        action.add_argument("input", type=Path)
        action.add_argument("--output", "-o", type=Path)


def run_context_command(args: argparse.Namespace) -> int:
    from .cli import _paths_collide

    output = args.model if args.context_action == "fit" else args.output
    protected = [args.input] if args.context_action == "fit" else [args.input, args.model]
    if any(_paths_collide(source, output) for source in protected):
        raise ValueError("context output must differ from every input path")
    if args.context_action == "fit":
        model = ExpectedContextModel(
            n_components=args.components,
            regularization=args.regularization,
            relation=args.relation,
            min_document_frequency=args.min_df,
            max_features=args.max_features,
            max_training_pairs=args.max_pairs,
            max_dense_cells=args.max_dense_cells,
        )
        try:
            model.fit(iter_path(args.input))
        except ImportError as error:
            raise ValueError(str(error)) from error
        model.save(args.model)
        print(
            json.dumps(
                {
                    "model_digest": model.digest,
                    "training_pairs": model.state.training_pairs,
                    "dimensions": model.state.dimensions,
                    "relation": model.relation,
                },
                sort_keys=True,
            )
        )
        return 0
    model = ExpectedContextModel.load(args.model)
    if args.context_action == "evaluate":
        payload: object = {"model_digest": model.digest, **model.evaluate(iter_path(args.input))}
    else:
        predictions = []
        seen: set[str] = set()
        for conversation in iter_path(args.input):
            if conversation.id in seen:
                raise ValueError("conversation IDs must be unique")
            seen.add(conversation.id)
            predictions.append(
                {
                    "conversation_id": conversation.id,
                    "predictions": {
                        identifier: prediction.to_dict()
                        for identifier, prediction in model.transform(conversation).items()
                    },
                }
            )
        payload = {"model_digest": model.digest, "predictions": predictions}
    rendered = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        sys.stdout.write(rendered)
    else:
        output.write_text(rendered, encoding="utf-8")
    return 0
