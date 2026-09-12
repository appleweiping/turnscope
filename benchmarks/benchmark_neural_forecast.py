"""Fit a fixed CGA candidate privately without producing final-test predictions.

This first stage intentionally cannot declare real-data quality or whole-project
parity. Later matched-baseline/ablation evaluation must use the frozen candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

from turnscope import (
    HierarchicalEventForecaster,
    NeuralForecastConfig,
    NeuralNumericLimits,
    NeuralTrainingConfig,
    SequenceLimits,
    SequencePolicy,
)
from turnscope.neural_cli_io import NeuralOutputError, publish_neural_report
from turnscope.neural_forecast_math import NUMERICAL_VERSION
from turnscope.neural_forecast_train import INITIALIZATION_VERSION, TRAINING_VERSION

SEEDS = (17, 101, 202)


def _adapter() -> ModuleType:
    # Load exactly the shipped neighboring helper; never extend the import path
    # with a reference checkout or treat an arbitrary module name as a plugin.
    path = Path(__file__).with_name("neural_cga_data.py")
    spec = importlib.util.spec_from_file_location("_turnscope_fixed_neural_cga", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("the shipped CGA adapter could not be located")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fit_protocol(seed: int) -> dict[str, Any]:
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("the preregistered candidate seed must be 17, 101 or 202")
    return {
        "format": "turnscope.neural-cga-candidate-fit.v1",
        "source_partition_protocol": _adapter().protocol(),
        "model_config": NeuralForecastConfig().to_dict(),
        "training_config": NeuralTrainingConfig(seed=seed).to_dict(),
        "eligibility_policy": SequencePolicy().to_dict(),
        "data_limits": SequenceLimits().to_dict(),
        "numeric_limits": NeuralNumericLimits().to_dict(),
        "numerical_version": NUMERICAL_VERSION,
        "training_version": TRAINING_VERSION,
        "initialization_version": INITIALIZATION_VERSION,
        "torch_cpu_threads": 1,
        "torch_deterministic_algorithms": True,
        "preregistered_seeds": list(SEEDS),
        "seed_selection_by_test_results": False,
        "test_in_fit": "source schema/partition audit only; no model test predictions or scores",
        "test_cohort_history": "official test previously inspected by lexical/context benchmarks",
        "required_later_comparisons": [
            "same-token cumulative NB with independent policy threshold",
            "eligible-training-conversation prior with independent policy threshold",
            "separately trained current-turn-only encoder",
            "separately trained mean-word rather than word-recurrence encoder",
            "separately trained order-erased baseline",
            "frozen heldout prefix-safe order intervention (OOD diagnostic, not retraining)",
        ],
        "real_data_quality_claimed": False,
        "whole_repository_parity_claimed": False,
    }


def source_bindings() -> dict[str, str]:
    import turnscope

    package = Path(turnscope.__file__).resolve().parent
    result = {
        "src/turnscope/" + path.relative_to(package).as_posix(): _sha(path.read_bytes())
        for path in sorted(package.rglob("*.py"))
    }
    result["src/turnscope/py.typed"] = _sha((package / "py.typed").read_bytes())
    for name in ("benchmark_neural_forecast.py", "neural_cga_data.py"):
        result["benchmarks/" + name] = _sha(Path(__file__).with_name(name).read_bytes())
    return result


def validate_fit_summary(
    summary: dict[str, Any], protocol: dict[str, Any], prepared: dict[str, Any]
) -> None:
    """Bind the returned candidate to the admitted plan and independently prepared source."""
    for name, planned in (
        ("config", "model_config"),
        ("training_config", "training_config"),
        ("eligibility_policy", "eligibility_policy"),
        ("data_limits", "data_limits"),
        ("numeric_limits", "numeric_limits"),
    ):
        if _canonical(summary[name]) != _canonical(protocol[planned]):
            raise ValueError("fitted candidate differs from the declared protocol: " + name)
    for name, partition in (
        ("training", "training"),
        ("model_validation", "validation"),
        ("policy_validation", "policy_validation"),
    ):
        source = prepared[partition]
        if summary[name + "_partition_digest"] != source["partition_digest"]:
            raise ValueError("fitted candidate partition differs from the admitted source")
        for suffix, key in (("conversations", "eligible_conversations"), ("prefixes", "prefixes")):
            value = summary[name + "_" + suffix]
            if type(value) is not int or value != source["audit"][key]:
                raise ValueError("fitted candidate support differs from the admitted source")
    if (
        summary["policy_reuses_model_validation"] is not False
        or type(summary["torch_threads"]) is not int
        or summary["torch_threads"] != protocol["torch_cpu_threads"]
        or summary["probability_calibration_claimed"] is not False
    ):
        raise ValueError("fitted candidate changes the selection/runtime/claim policy")


def fit_candidate(archive: Path, directory: Path, *, seed: int = 17) -> dict[str, Any]:
    """Fit one predeclared seed; never infer on the official test partition."""
    protocol = fit_protocol(seed)
    directory = directory.absolute()
    checkout = Path(__file__).resolve().parents[1]
    if checkout == directory.resolve() or checkout in directory.resolve().parents:
        raise ValueError("real-data candidates must be stored outside the project checkout")
    if archive.resolve() == directory.resolve():
        raise ValueError("source archive and private output directory must differ")
    before = source_bindings()
    directory.mkdir()  # An interrupted or failed directory is never silently reused.
    publish_neural_report(
        {
            "protocol": protocol,
            "protocol_sha256": _sha(_canonical(protocol)),
            "source_sha256": before,
        },
        directory / "fit-plan.json",
    )
    started = time.perf_counter()
    partitions = _adapter().load_neural_cga(archive)
    parsed_seconds = time.perf_counter() - started
    prepared = partitions.prepared_audit()
    source = partitions.audit_dict()
    publish_neural_report(
        {"source": source, "prepared": prepared, "load_seconds": parsed_seconds},
        directory / "source-audit.json",
    )
    model = HierarchicalEventForecaster(training_config=NeuralTrainingConfig(seed=seed))
    # This explicit benchmark-process policy is distinct from the package API,
    # which never changes a caller's thread or deterministic-kernel settings.
    import torch

    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        model.fit(
            partitions.training,
            partitions.validation,
            policy_validation=partitions.policy_validation,
        )
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
    fit_seconds = time.perf_counter() - started
    validate_fit_summary(model.training_summary, protocol, prepared)
    if source_bindings() != before:
        raise RuntimeError("bound source changed during candidate fitting; no candidate published")
    saved = model.save(directory / "private-model.tsn")
    restored = HierarchicalEventForecaster.load(directory / "private-model.tsn")
    if restored.digest != model.digest or restored.training_summary != model.training_summary:
        raise RuntimeError("published candidate did not reproduce the complete fitted state")
    if source_bindings() != before:
        raise RuntimeError(
            "bound source changed during publication; model may remain without receipt"
        )
    report = {
        "format": "turnscope.neural-cga-candidate-receipt.v1",
        "candidate_fit_completed": True,
        "final_evaluation_completed": False,
        "official_test_predictions_produced": False,
        "protocol_sha256": _sha(_canonical(protocol)),
        "source_sha256": before,
        "source": source,
        "prepared": prepared,
        "publication": asdict(saved),
        "model_digest": model.digest,
        "training": model.training_summary,
        "fit_seconds": fit_seconds,
        "source_load_seconds": parsed_seconds,
        "timing_scope": (
            "fit includes preparation, optimization and policy selection; "
            "excludes source load and save/reload"
        ),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "real_data_quality_claimed": False,
        "whole_repository_parity_claimed": False,
        "remaining_before_quality_acceptance": protocol["required_later_comparisons"],
        "privacy": (
            "local private vocabulary and parameters; no model weights or source "
            "records may be published by this script"
        ),
    }
    publish_neural_report(report, directory / "candidate-receipt.json")
    return report


def main(argv: list[str] | None = None) -> int:
    options: dict[str, Any] = {}
    if sys.version_info >= (3, 14):
        options = {"color": False, "formatter_class": partial(argparse.HelpFormatter, color=False)}
    parser = argparse.ArgumentParser(description=__doc__, **options)
    parser.add_argument("archive", type=Path)
    parser.add_argument("directory", type=Path, help="new private directory outside the checkout")
    parser.add_argument("--seed", choices=SEEDS, type=int, default=17)
    args = parser.parse_args(argv)
    report = fit_candidate(args.archive, args.directory, seed=args.seed)
    try:
        publish_neural_report(report, None)
    except NeuralOutputError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
