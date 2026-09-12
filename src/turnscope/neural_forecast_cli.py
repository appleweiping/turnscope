"""Private-by-default neural fit, frozen prediction, evaluation and inspection."""

from __future__ import annotations

import argparse
import sys
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .neural_cli_io import (
    NeuralOutputError,
    check_new_output,
    publish_neural_report,
    read_neural_jsonl,
    read_neural_settings,
)
from .neural_forecast import HierarchicalEventForecaster, NeuralForecastConfig
from .neural_forecast_data import SequenceLimits, SequencePolicy, observed_prefix
from .neural_forecast_math import NeuralNumericLimits
from .neural_forecast_train import NeuralTrainingConfig


def configure_neural_forecast_parser(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="neural_action", required=True)
    fit = actions.add_parser(
        "fit", help="train a new CPU model; select epoch and validation-only policy"
    )
    fit.add_argument("training", type=Path, help="bounded conversation JSONL")
    fit.add_argument("validation", type=Path, help="disjoint model-validation JSONL")
    fit.add_argument("model", type=Path, help="new private inference bundle; never overwrites")
    fit.add_argument(
        "--policy-validation", type=Path, help="optional separate threshold-selection JSONL"
    )
    fit.add_argument("--settings", type=Path, help="bounded JSON configuration objects")
    fit.add_argument(
        "--output", "-o", type=Path, help="new aggregate training report; default stdout"
    )
    for name in ("predict", "evaluate"):
        command = actions.add_parser(name, help=f"{name} using frozen NumPy-only inference")
        command.add_argument("model", type=Path)
        command.add_argument("input", type=Path, help="bounded conversation JSONL")
        command.add_argument(
            "--output", "-o", type=Path, help="new aggregate report; default stdout"
        )
        if name == "predict":
            command.add_argument(
                "--include-identifiers",
                action="store_true",
                help="include private conversation IDs (never input text)",
            )
    inspect = actions.add_parser(
        "inspect", help="validate a bundle and print metadata without Torch"
    )
    inspect.add_argument("model", type=Path)
    inspect.add_argument("--output", "-o", type=Path, help="new metadata report; default stdout")


def _settings(path: Path | None) -> HierarchicalEventForecaster:
    raw = {} if path is None else read_neural_settings(path)
    constructors = {
        "config": NeuralForecastConfig,
        "training_config": NeuralTrainingConfig,
        "policy": SequencePolicy,
        "data_limits": SequenceLimits,
        "numeric_limits": NeuralNumericLimits,
    }
    try:
        selected: dict[str, Any] = {key: constructors[key](**value) for key, value in raw.items()}
    except TypeError:
        raise ValueError("neural settings contain unsupported fields or value types") from None
    return HierarchicalEventForecaster(**selected)


def _independent(paths: list[Path]) -> None:
    for index, path in enumerate(paths):
        for previous in paths[:index]:
            if path.resolve() == previous.resolve() or (
                path.exists() and previous.exists() and path.samefile(previous)
            ):
                raise ValueError("neural input paths must be distinct files")


def _fit(args: argparse.Namespace) -> dict[str, Any]:
    # Artifact imports are deliberately local; parser/help need no NumPy or Torch.
    from .neural_forecast_artifact import save_neural_forecaster

    inputs = [args.training, args.validation]
    if args.policy_validation is not None:
        inputs.append(args.policy_validation)
    if args.settings is not None:
        inputs.append(args.settings)
    _independent(inputs)
    check_new_output(args.model, inputs)
    check_new_output(args.output, [*inputs, args.model])
    if args.output is not None and args.output.resolve() == args.model.resolve():
        raise ValueError("neural model and report destinations must differ")
    model = _settings(args.settings)
    training, training_source = read_neural_jsonl(args.training)
    validation, validation_source = read_neural_jsonl(args.validation)
    sources = {"training": training_source, "model_validation": validation_source}
    policy_validation = None
    if args.policy_validation is not None:
        policy_validation, policy_source = read_neural_jsonl(args.policy_validation)
        sources["policy_validation"] = policy_source
    model.fit(training, validation, policy_validation=policy_validation)
    publication = save_neural_forecaster(model, args.model)
    return {
        "format": "turnscope.neural-training-command.v1",
        "completed": True,
        "model_digest": model.digest,
        "publication": asdict(publication),
        "sources": sources,
        "training": model.training_summary,
        "privacy": (
            "The private bundle contains vocabulary and weights; this report omits raw text."
        ),
    }


def _inference(args: argparse.Namespace) -> dict[str, Any]:
    from .neural_forecast_artifact import load_neural_forecaster

    protected = [args.model] if args.neural_action == "inspect" else [args.model, args.input]
    _independent(protected)
    check_new_output(args.output, protected)
    model = load_neural_forecaster(args.model)
    if args.neural_action == "inspect":
        return {
            "format": "turnscope.neural-inspection-command.v1",
            "model_digest": model.digest,
            "training": model.training_summary,
        }
    conversations, source = read_neural_jsonl(args.input)
    if args.neural_action == "evaluate":
        return {**model.evaluate(conversations), "source": source}
    state = model.state
    prefixes = tuple(
        observed_prefix(conversation, policy=state.policy, limits=state.data_limits)
        for conversation in conversations
    )
    predictions = model.transform(prefixes)
    rows = []
    for index, (conversation, prediction) in enumerate(
        zip(conversations, predictions, strict=True)
    ):
        row = {"index": index, **prediction.to_dict()}
        if args.include_identifiers:
            row["conversation_id"] = conversation.id
        rows.append(row)
    return {
        "format": "turnscope.neural-prediction-command.v1",
        "model_digest": model.digest,
        "source": source,
        "predictions": rows,
        "identifiers_included": args.include_identifiers,
        "text_included": False,
    }


def run_neural_forecast_command(args: argparse.Namespace) -> int:
    """Return failure on broken host output without claiming published files vanished."""
    try:
        payload = _fit(args) if args.neural_action == "fit" else _inference(args)
        warning = publish_neural_report(payload, args.output)
    except ImportError as error:
        raise ValueError(str(error)) from None
    except NeuralOutputError:
        return 1
    if warning is not None:
        with suppress(OSError, ValueError):
            sys.stderr.write(warning + "\n")
    return 0
