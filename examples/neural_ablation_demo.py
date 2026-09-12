"""Fit all three authored controls, then run each deployment command without Torch.

Run with an installed package and Python -I. This helper never changes sys.path
or downloads training data. Generated inputs and models remain private locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any

VARIANTS = ("current-turn.v1", "mean-word.v1", "order-erased.v1")
_TRAIN_COMMAND = """
import sys
import torch
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
from turnscope.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
_FROZEN_COMMAND = """
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise AssertionError('frozen control attempted to import Torch')
sys.meta_path.insert(0, NoTorch())
from turnscope.cli import main
code = main(sys.argv[1:])
if any(name.split('.')[0] == 'torch' for name in sys.modules):
    raise AssertionError('frozen control imported Torch')
raise SystemExit(code)
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources() -> dict[str, str]:
    import turnscope

    package = Path(turnscope.__file__).resolve().parent
    result = {
        f"src/turnscope/{path.relative_to(package).as_posix()}": _sha(path)
        for path in sorted(package.rglob("*.py"))
    }
    result["src/turnscope/py.typed"] = _sha(package / "py.typed")
    result["examples/neural_ablation_demo.py"] = _sha(Path(__file__))
    return result


def _records(partition: str, *, observed: bool = False) -> list[dict[str, Any]]:
    records = []
    for index in range(4):
        event = index % 2 == 0
        texts = (
            ["We disagree about this evidence.", "Please reconsider this disputed claim."]
            if event
            else ["We agree to compare evidence.", "Thank you for checking this claim."]
        )
        if not observed:
            texts.append("An authored annotated event." if event else "A censored ending.")
        records.append(
            {
                "id": f"authored-control-{partition}-{index}",
                "metadata": {"forecast_groups": [f"authored-control:{partition}:{index}"]},
                "utterances": [
                    {
                        "id": f"turn-{turn}",
                        "role": "person",
                        "text": text,
                        "timestamp": f"2026-01-01T00:00:{turn:02d}+00:00",
                        "metadata": {"event": event and turn == 2},
                    }
                    for turn, text in enumerate(texts)
                ],
            }
        )
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _run(arguments: list[str], *, frozen: bool) -> None:
    command = [sys.executable, "-I", "-X", "utf8"]
    command.extend(["-c", _FROZEN_COMMAND if frozen else _TRAIN_COMMAND])
    subprocess.run(
        [*command, "neural-ablation", *arguments],
        capture_output=True,
        check=True,
        encoding="utf-8",
        timeout=180,
    )


def _source_receipt(receipt: dict[str, Any], path: Path, *, turns: int) -> None:
    if receipt["sha256"] != _sha(path) or any(
        type(receipt[key]) is not int or receipt[key] != expected
        for key, expected in {
            "bytes": path.stat().st_size,
            "conversations": 4,
            "utterances": turns,
        }.items()
    ):
        raise ValueError("command source binding differs from the authored input")


def _number(value: Any, *, minimum: float = -math.inf, maximum: float = math.inf) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError("demo numerical receipt requires a finite number in range")
    return float(value)


def _match_metric(actual: Any, expected: Any) -> None:
    """Closed known scalar metric tree; no unchecked fields enter the aggregate."""
    if type(expected) is dict:
        if type(actual) is not dict or actual.keys() != expected.keys():
            raise ValueError("demo metric inventory differs")
        for name, value in expected.items():
            _match_metric(actual[name], value)
    elif expected is None:
        if actual is not None:
            raise ValueError("undefined authored metric must remain null")
    elif type(expected) is int:
        if type(actual) is not int or actual != expected:
            raise ValueError("authored metric count differs")
    elif not math.isclose(_number(actual), expected, rel_tol=1e-13, abs_tol=1e-14):
        raise ValueError("authored metric differs from the delivered prediction values")


def _check_numerical_receipts(
    rows: list[dict[str, Any]], metrics: dict[str, Any], threshold: Any
) -> None:
    threshold = _number(threshold, minimum=0, maximum=1)
    probabilities = []
    for row in rows:
        probability = _number(row["probability"], minimum=1e-15, maximum=1 - 1e-15)
        logit = _number(row["logit"])
        exponential = math.exp(-abs(logit))
        sigmoid = 1 / (1 + exponential) if logit >= 0 else exponential / (1 + exponential)
        sigmoid = min(1 - 1e-15, max(1e-15, sigmoid))
        if (
            not math.isclose(probability, sigmoid, rel_tol=1e-13, abs_tol=1e-14)
            or _number(row["threshold"], minimum=0, maximum=1) != threshold
            or type(row["alert"]) is not bool
            or row["alert"] is not (probability >= threshold)
        ):
            raise ValueError("prediction logit, probability, threshold or decision disagrees")
        probabilities.append(probability)
    # The fixture has exactly one prefix per conversation, alternating positive
    # and censored negative labels. Compute independently of production metrics.
    positives, negatives = probabilities[::2], probabilities[1::2]
    tp = sum(value >= threshold for value in positives)
    fp = sum(value >= threshold for value in negatives)
    tn, fn = 2 - fp, 2 - tp
    expected = {
        "prefixes": 4,
        "conversations": 4,
        "positive_conversations": 2,
        "conversation_weighted_prefix_brier": math.fsum(
            (value - (index % 2 == 0)) ** 2 for index, value in enumerate(probabilities)
        )
        / 4,
        "conversation_weighted_prefix_log_loss": -math.fsum(
            math.log(value) if index % 2 == 0 else math.log1p(-value)
            for index, value in enumerate(probabilities)
        )
        / 4,
        "conversation_weighted_prefix_roc_auc": math.fsum(
            1.0 if positive > negative else 0.5 if positive == negative else 0.0
            for positive in positives
            for negative in negatives
        )
        / 4,
        "any_alert": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": (tp + tn) / 4,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / 2,
            "false_positive_rate": fp / 2,
            "balanced_accuracy": (tp + tn) / 4,
            "f1": 2 * tp / (2 * tp + fp + fn),
        },
        "true_positive_first_alerts": tp,
        "mean_lead_turns_for_true_positives": 1.0 if tp else None,
        "threshold": threshold,
    }
    _match_metric(metrics, expected)


def _validate_reports(
    inputs: Path, directory: Path, variant: str, reports: dict[str, Any]
) -> dict[str, Any]:
    """Check actual receipts, variant, support and privacy; never require good scores."""
    formats = {
        "training": "turnscope.neural-ablation-training-command.v1",
        "inspection": "turnscope.neural-ablation-inspection-command.v1",
        "prediction": "turnscope.neural-ablation-prediction-command.v1",
        "evaluation": "turnscope.neural-ablation-report.v1",
    }
    if reports.keys() != formats.keys() or any(
        reports[key].get("format") != expected or reports[key].get("variant") != variant
        for key, expected in formats.items()
    ):
        raise ValueError("command report format or explicitly selected control differs")
    training = reports["training"]
    digest = training["model_digest"]
    if training["completed"] is not True or any(
        report["model_digest"] != digest for report in reports.values()
    ):
        raise ValueError("commands disagree about completed model identity")
    model = directory / "private-model.tsa"
    publication = training["publication"]
    if (
        Path(publication["path"]) != model
        or publication["sha256"] != _sha(model)
        or type(publication["bytes_written"]) is not int
        or publication["bytes_written"] != model.stat().st_size
    ):
        raise ValueError("artifact publication does not bind the delivered bytes")
    summary = training["training"]
    if reports["inspection"]["training"] != summary or summary["variant"] != variant:
        raise ValueError("restored training state differs from fitted control")
    if (
        any(
            type(summary[key]) is not int or summary[key] != 4
            for key in (
                "training_conversations",
                "model_validation_conversations",
                "policy_validation_conversations",
                "training_prefixes",
                "model_validation_prefixes",
                "policy_validation_prefixes",
            )
        )
        or summary["policy_reuses_model_validation"] is not False
    ):
        raise ValueError("training did not use the three disjoint authored partitions")
    if (
        len(summary["history"]) != 2
        or any(
            type(epoch["optimizer_steps"]) is not int or epoch["optimizer_steps"] != 2
            for epoch in summary["history"]
        )
        or summary["training_config"]["seed"] != 17
        or summary["config"]["variant"] != variant
    ):
        raise ValueError("authored full/partial batch training contract differs")
    constant, array_count = {
        "current-turn.v1": (76993, 16),
        "mean-word.v1": (39361, 9),
        "order-erased.v1": (2113, 5),
    }[variant]
    if (
        summary["parameter_count"] != 64 * summary["vocabulary_size"] + constant
        or summary["canonical_main_parameter_count"] != 64 * summary["vocabulary_size"] + 89281
        or summary["fixed_pad_parameters"] != 64
        or len(summary["initial_parameter_sha256"]) != array_count
        or len(summary["main_initial_parameter_sha256"]) != 17
        or any(
            summary["main_initial_parameter_sha256"].get(name) != value
            for name, value in summary["initial_parameter_sha256"].items()
        )
    ):
        raise ValueError("active control parameters differ from canonical initialization subset")
    sources = training["sources"]
    source_files = {
        "training": "training.jsonl",
        "model_validation": "validation.jsonl",
        "policy_validation": "policy.jsonl",
    }
    if sources.keys() != source_files.keys():
        raise ValueError("fitting source inventory differs")
    for name, file in source_files.items():
        _source_receipt(sources[name], inputs / file, turns=12)
    for action, file, turns in (
        ("prediction", "observed.jsonl", 8),
        ("evaluation", "heldout.jsonl", 12),
    ):
        _source_receipt(reports[action]["source"], inputs / file, turns=turns)
    inspection = reports["inspection"]
    if (
        any(
            inspection[key] is not False
            for key in (
                "vocabulary_included",
                "weights_included",
                "source_execution_provenance_claimed",
            )
        )
        or training["source_execution_provenance_claimed"] is not False
    ):
        raise ValueError("command privacy or standalone provenance claims differ")
    prediction = reports["prediction"]
    if (
        prediction["identifiers_included"] is not False
        or prediction["text_included"] is not False
        or len(prediction["predictions"]) != 4
        or any(
            type(row["index"]) is not int
            or row["index"] != index
            or row["variant"] != variant
            or row["model_digest"] != digest
            or type(row["observed_turns"]) is not int
            or row["observed_turns"] != 2
            or "conversation_id" in row
            or "text" in row
            for index, row in enumerate(prediction["predictions"])
        )
    ):
        raise ValueError("private observed-only prediction inventory differs")
    evaluation = reports["evaluation"]
    if any(
        type(evaluation["support"][key]) is not int or evaluation["support"][key] != count
        for key, count in {
            "source_conversations": 4,
            "source_turns": 12,
            "eligible_conversations": 4,
            "excluded_conversations": 0,
            "observed_turns": 8,
            "prefixes": 4,
            "positive_conversations": 2,
            "negative_conversations": 2,
        }.items()
    ) or any(
        type(evaluation["metrics"][key]) is not int or evaluation["metrics"][key] != count
        for key, count in {"conversations": 4, "prefixes": 4, "positive_conversations": 2}.items()
    ):
        raise ValueError("authored heldout denominators differ")
    if any(
        evaluation[key] is not False
        for key in (
            "policy_reuses_model_validation",
            "probability_calibration_claimed",
            "whole_repository_parity_claimed",
        )
    ):
        raise ValueError("heldout claim boundaries differ")
    _check_numerical_receipts(
        prediction["predictions"], evaluation["metrics"], summary["threshold"]
    )
    return {
        "variant": variant,
        "model_digest": digest,
        "model_file_sha256": publication["sha256"],
        "parameter_count": summary["parameter_count"],
        "vocabulary_digest": summary["vocabulary_digest"],
        "main_initial_parameter_sha256": summary["main_initial_parameter_sha256"],
        "selected_epoch": summary["selected_epoch"],
        "optimizer_steps_executed": sum(epoch["optimizer_steps"] for epoch in summary["history"]),
        "authored_heldout_metrics": evaluation["metrics"],
        "frozen_commands_with_torch_forbidden": ["inspect", "predict", "evaluate"],
        "torch_training_version": summary["torch_version"],
        "numpy_training_version": summary["numpy_version"],
    }


def run_demo(directory: Path) -> dict[str, Any]:
    from turnscope.neural_cli_io import publish_neural_report

    source_bindings = _sources()
    directory = directory.absolute()
    directory.mkdir()  # Exclusive; preserve failed attempts instead of silently restarting.
    for name in ("training", "validation", "policy", "heldout"):
        _write_jsonl(directory / f"{name}.jsonl", _records(name))
    _write_jsonl(directory / "observed.jsonl", _records("heldout", observed=True))
    publish_neural_report(
        {"training_config": {"epochs": 2, "patience": 2, "batch_conversations": 3, "seed": 17}},
        directory / "settings.json",
    )
    input_bindings = {path.name: _sha(path) for path in directory.iterdir()}
    results = []
    for variant in VARIANTS:
        target = directory / variant
        target.mkdir()
        model = target / "private-model.tsa"
        _run(
            [
                "fit",
                str(directory / "training.jsonl"),
                str(directory / "validation.jsonl"),
                str(model),
                "--policy-validation",
                str(directory / "policy.jsonl"),
                "--variant",
                variant,
                "--settings",
                str(directory / "settings.json"),
                "--output",
                str(target / "training-report.json"),
            ],
            frozen=False,
        )
        for action, file, name in (
            ("inspect", None, "inspection"),
            ("predict", "observed.jsonl", "prediction"),
            ("evaluate", "heldout.jsonl", "evaluation"),
        ):
            arguments = [action, str(model)]
            if file is not None:
                arguments.append(str(directory / file))
            _run([*arguments, "--output", str(target / f"{name}-report.json")], frozen=True)
        reports = {
            name: json.loads((target / f"{name}-report.json").read_text(encoding="utf-8"))
            for name in ("training", "inspection", "prediction", "evaluation")
        }
        results.append(_validate_reports(directory, target, variant, reports))
    if any(
        result["main_initial_parameter_sha256"] != results[0]["main_initial_parameter_sha256"]
        or result["vocabulary_digest"] != results[0]["vocabulary_digest"]
        for result in results[1:]
    ):
        raise ValueError("controls did not share the same vocabulary and untrained initialization")
    if _sources() != source_bindings or any(
        _sha(directory / name) != digest for name, digest in input_bindings.items()
    ):
        raise ValueError("runtime, helper or input source changed during the demo")
    report = {
        "format": "turnscope.neural-ablation-installed-demo.v1",
        "completed": True,
        "authored_fixture_only": True,
        "real_data_quality_claimed": False,
        "whole_repository_parity_claimed": False,
        "python_version": sys.version.split()[0],
        "variants": results,
        "source_sha256": source_bindings,
        "file_sha256": {
            path.relative_to(directory).as_posix(): _sha(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        },
    }
    publish_neural_report(report, directory / "demo-report.json")
    return report


def main(argv: list[str] | None = None) -> int:
    from turnscope.neural_cli_io import NeuralOutputError, publish_neural_report

    options: dict[str, Any] = {}
    if sys.version_info >= (3, 14):
        options = {"color": False, "formatter_class": partial(argparse.HelpFormatter, color=False)}
    parser = argparse.ArgumentParser(description=__doc__, **options)
    parser.add_argument("directory", type=Path, help="new private output directory")
    args = parser.parse_args(argv)
    report = run_demo(args.directory)
    try:
        publish_neural_report(report, None)
    except NeuralOutputError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
