"""CLI boundaries for group-separated supervised prefix forecasting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .forecast import PrefixEventForecaster
from .io import iter_path


def configure_forecast_parser(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="forecast_action", required=True)
    fit = actions.add_parser("fit", help="fit on training groups and tune on validation groups")
    fit.add_argument("training", type=Path)
    fit.add_argument("validation", type=Path)
    fit.add_argument("model", type=Path)
    fit.add_argument("--alpha", type=float, default=1.0)
    fit.add_argument("--max-features", type=int, default=10_000)
    fit.add_argument("--min-turns", type=int, default=2)
    fit.add_argument("--event-field", default="event")
    fit.add_argument("--skip-field", default="is_section_header")
    fit.add_argument("--groups-field", default="forecast_groups")
    fit.add_argument("--max-prefixes", type=int, default=30_000)
    fit.add_argument("--max-prefix-cells", type=int, default=4_000_000)
    fit.add_argument("--max-tokens", type=int, default=2_000_000)
    for command in ("predict", "evaluate"):
        action = actions.add_parser(command)
        action.add_argument("model", type=Path)
        action.add_argument("input", type=Path)
        action.add_argument("--output", "-o", type=Path)


def run_forecast_command(args: argparse.Namespace) -> int:
    from .cli import _paths_collide

    if args.forecast_action == "fit":
        if _paths_collide(args.training, args.validation) or any(
            _paths_collide(source, args.model) for source in (args.training, args.validation)
        ):
            raise ValueError("forecast training, validation and model paths must differ")
        model = PrefixEventForecaster(
            **{
                name: getattr(args, name)
                for name in (
                    "alpha",
                    "max_features",
                    "min_turns",
                    "event_field",
                    "skip_field",
                    "groups_field",
                    "max_prefixes",
                    "max_prefix_cells",
                    "max_tokens",
                )
            }
        )
        model.fit(iter_path(args.training), iter_path(args.validation))
        model.save(args.model)
        print(
            json.dumps(
                {
                    "model_digest": model.digest,
                    "threshold": model.state.threshold,
                    "training_prefixes": model.state.training_prefixes,
                    "validation_prefixes": model.state.validation_prefixes,
                    "validation_balanced_accuracy": model.state.validation_balanced_accuracy,
                },
                sort_keys=True,
            )
        )
        return 0
    if any(_paths_collide(source, args.output) for source in (args.input, args.model)):
        raise ValueError("forecast output must differ from every input")
    model = PrefixEventForecaster.load(args.model)
    if args.forecast_action == "evaluate":
        payload = model.evaluate(iter_path(args.input))
    else:
        predictions = {}
        for conversation in iter_path(args.input):
            if conversation.id in predictions:
                raise ValueError("forecast conversation IDs must be unique")
            predictions[conversation.id] = model.predict(conversation).to_dict()
        payload = {"model_digest": model.digest, "predictions": predictions}
    rendered = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0
