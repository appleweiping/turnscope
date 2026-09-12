"""Explicit three-partition neural-control commands with private diagnostics."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .neural_ablation import AblationEventForecaster, AblationForecastConfig
from .neural_ablation_math import ABLATION_VARIANTS
from .neural_cli_io import (
    check_new_output,
    publish_neural_report,
    read_neural_jsonl,
    read_neural_settings,
)
from .neural_forecast_cli import _independent
from .neural_forecast_data import SequenceLimits, SequencePolicy, observed_prefix
from .neural_forecast_math import NeuralNumericLimits
from .neural_forecast_train import NeuralTrainingConfig


def configure_neural_ablation_parser(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="neural_ablation_action", required=True)
    fit = actions.add_parser("fit", help="fit one explicitly selected three-partition control")
    fit.add_argument("training", type=Path, help="bounded training conversation JSONL")
    fit.add_argument("validation", type=Path, help="disjoint epoch-selection JSONL")
    fit.add_argument("model", type=Path, help="new private control artifact; never overwrites")
    fit.add_argument(
        "--policy-validation",
        required=True,
        type=Path,
        help="third disjoint threshold-selection JSONL",
    )
    fit.add_argument("--variant", required=True, choices=ABLATION_VARIANTS)
    fit.add_argument(
        "--settings", type=Path, help="five closed configuration sections; no variant field"
    )
    fit.add_argument("--output", "-o", type=Path, help="new training report; default stdout")
    for name in ("predict", "evaluate"):
        command = actions.add_parser(name, help=f"{name} with frozen NumPy-only inference")
        command.add_argument("model", type=Path)
        command.add_argument("input", type=Path, help="bounded conversation JSONL")
        command.add_argument("--output", "-o", type=Path, help="new report; default stdout")
        if name == "predict":
            command.add_argument(
                "--include-identifiers",
                action="store_true",
                help="include private conversation IDs, never source text",
            )
    inspect = actions.add_parser(
        "inspect", help="validate the artifact and show metadata without Torch"
    )
    inspect.add_argument("model", type=Path)
    inspect.add_argument("--output", "-o", type=Path, help="new metadata report; default stdout")


def _settings(path: Path | None, variant: str) -> AblationEventForecaster:
    raw = {} if path is None else read_neural_settings(path)
    config = raw.get("config", {})
    if "variant" in config:
        raise ValueError("variant must be supplied only by the explicit command option")
    constructors = {
        "training_config": NeuralTrainingConfig,
        "policy": SequencePolicy,
        "data_limits": SequenceLimits,
        "numeric_limits": NeuralNumericLimits,
    }
    try:
        selected: dict[str, Any] = {
            key: constructors[key](**value) for key, value in raw.items() if key != "config"
        }
        selected["config"] = AblationForecastConfig(variant=variant, **config)
    except TypeError:
        raise ValueError("unsupported neural-control settings") from None
    # Pooling and training-pooling limits deliberately remain the API defaults;
    # this command does not add unversioned settings to the shared JSON reader.
    return AblationEventForecaster(**selected)


def _fit(args: argparse.Namespace) -> dict[str, Any]:
    from .neural_ablation_artifact import save_ablation_forecaster

    inputs = [args.training, args.validation, args.policy_validation]
    if args.settings is not None:
        inputs.append(args.settings)
    _independent(inputs)
    check_new_output(args.model, inputs)
    check_new_output(args.output, [*inputs, args.model])
    model = _settings(args.settings, args.variant)
    training, training_source = read_neural_jsonl(args.training)
    validation, validation_source = read_neural_jsonl(args.validation)
    policy, policy_source = read_neural_jsonl(args.policy_validation)
    model.fit(training, validation, policy_validation=policy)
    publication = save_ablation_forecaster(model, args.model)
    return {
        "format": "turnscope.neural-ablation-training-command.v1",
        "completed": True,
        "variant": args.variant,
        "model_digest": model.digest,
        "publication": asdict(publication),
        "sources": {
            "training": training_source,
            "model_validation": validation_source,
            "policy_validation": policy_source,
        },
        "training": model.training_summary,
        "privacy": (
            "Private artifact contains vocabulary and weights; "
            "this report omits source text and vocabulary."
        ),
        "source_execution_provenance_claimed": False,
    }


def _inference(args: argparse.Namespace) -> dict[str, Any]:
    from .neural_ablation_artifact import load_ablation_forecaster

    action = args.neural_ablation_action
    protected = [args.model] if action == "inspect" else [args.model, args.input]
    _independent(protected)
    check_new_output(args.output, protected)
    model = load_ablation_forecaster(args.model)
    if action == "inspect":
        return {
            "format": "turnscope.neural-ablation-inspection-command.v1",
            "variant": model.state.config.variant,
            "model_digest": model.digest,
            "training": model.training_summary,
            "vocabulary_included": False,
            "weights_included": False,
            "source_execution_provenance_claimed": False,
        }
    conversations, source = read_neural_jsonl(args.input)
    if action == "evaluate":
        return {**model.evaluate(conversations), "source": source}
    prefixes = tuple(
        observed_prefix(conversation, policy=model.state.policy, limits=model.state.data_limits)
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
        "format": "turnscope.neural-ablation-prediction-command.v1",
        "variant": model.state.config.variant,
        "model_digest": model.digest,
        "source": source,
        "predictions": rows,
        "identifiers_included": args.include_identifiers,
        "text_included": False,
    }


def _diagnostic(message: str) -> None:
    try:
        sys.stderr.write("turnscope: " + message + "\n")
        sys.stderr.flush()
    except (OSError, ValueError, AttributeError, UnicodeError):
        # A closed or absent embedding stream must not hide the command result.
        return


def run_neural_ablation_command(args: argparse.Namespace) -> int:
    """0 completed; 1 completed work/report failed; 2 controlled command failure."""
    try:
        if args.neural_ablation_action not in ("fit", "predict", "evaluate", "inspect"):
            raise ValueError("unknown neural-control command")
        payload = _fit(args) if args.neural_ablation_action == "fit" else _inference(args)
    except (ValueError, TypeError, OSError, ImportError, ArithmeticError, RuntimeError, KeyError):
        _diagnostic(
            "error: neural-control command failed; "
            "check inputs, settings, dependencies and budgets."
        )
        return 2
    try:
        warning = publish_neural_report(payload, args.output)
    except (OSError, ValueError, AttributeError, UnicodeError):
        if args.neural_ablation_action == "fit":
            _diagnostic(
                "model artifact was published, but its report failed; "
                "do not infer rollback or automatically retrain."
            )
        else:
            _diagnostic("inference completed, but its report could not be published.")
        return 1
    if warning is not None:
        _diagnostic("report was published; private temporary-file cleanup was incomplete.")
    return 0
