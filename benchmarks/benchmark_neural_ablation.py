"""Fit one private, preregistered neural control; never score the official test set."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import math
import platform
import stat
import sys
import time
from contextlib import suppress
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

from turnscope.neural_ablation import AblationEventForecaster, AblationForecastConfig
from turnscope.neural_ablation_artifact import load_ablation_forecaster, save_ablation_forecaster
from turnscope.neural_ablation_math import (
    ABLATION_VARIANTS,
    AblationPoolingLimits,
    ablation_numerical_version,
    ablation_parameter_shapes,
)
from turnscope.neural_ablation_train import (
    ABLATION_INITIALIZATION_VERSION,
    ABLATION_TRAINING_VERSION,
    AblationTrainingLimits,
)
from turnscope.neural_cli_io import publish_neural_report
from turnscope.neural_forecast_data import (
    SequenceLimits,
    SequencePolicy,
    prepare_sequence_forecasts,
)
from turnscope.neural_forecast_math import NeuralArchitecture, NeuralNumericLimits
from turnscope.neural_forecast_train import INITIALIZATION_VERSION, NeuralTrainingConfig
from turnscope.neural_token_data import fit_sequence_vocabulary

SEEDS = (17, 101, 202)
_HELPERS = ("benchmark_neural_ablation.py", "benchmark_neural_forecast.py", "neural_cga_data.py")


def _helper() -> ModuleType:
    """Load only the shipped neighboring main driver, never an arbitrary plugin."""
    path = Path(__file__).with_name("benchmark_neural_forecast.py")
    spec = importlib.util.spec_from_file_location("_turnscope_ablation_main_driver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("the shipped main candidate helper is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fit_protocol(variant: str, seed: int) -> dict[str, Any]:
    ablation_numerical_version(variant)
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("the preregistered seed must be 17, 101 or 202")
    protocol: dict[str, Any] = _helper().fit_protocol(seed)
    protocol.update(
        {
            "format": "turnscope.neural-cga-ablation-fit.v1",
            "variant": variant,
            "model_config": AblationForecastConfig(variant).to_dict(),
            "reference_dimensions": {
                "embedding_dim": 64,
                "word_hidden": 64,
                "turn_hidden": 64,
                "word_layers": 1,
                "turn_layers": 1,
            },
            "pooling_limits": AblationPoolingLimits().to_dict(),
            "training_limits": AblationTrainingLimits().to_dict(),
            "numerical_version": ablation_numerical_version(variant),
            "training_version": ABLATION_TRAINING_VERSION,
            "initialization_version": ABLATION_INITIALIZATION_VERSION,
            "main_initialization_version": INITIALIZATION_VERSION,
            "preregistered_variants": list(ABLATION_VARIANTS),
            "prior_trained_parameters_used": False,
            "vocabulary_matching": "independent train-only refit; exact frozen vocabulary digest",
        }
    )
    return protocol


def source_bindings() -> dict[str, str]:
    """Bind the imported package location and every shipped helper actually reused."""
    import turnscope

    package = Path(turnscope.__file__).resolve().parent
    for name, module in tuple(sys.modules.items()):
        if name != "turnscope" and not name.startswith("turnscope."):
            continue
        filename = getattr(module, "__file__", None)
        if filename is not None and not Path(filename).resolve().is_relative_to(package):
            raise RuntimeError("imported TurnScope modules come from mixed package locations")
    paths = {
        "src/turnscope/" + path.relative_to(package).as_posix(): path
        for path in sorted(package.rglob("*.py"))
    }
    paths["src/turnscope/py.typed"] = package / "py.typed"
    paths.update({"benchmarks/" + name: Path(__file__).with_name(name) for name in _HELPERS})
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def _vocabulary_pin(partitions: Any, prepared: dict[str, Any]) -> dict[str, Any]:
    training = prepare_sequence_forecasts(partitions.training)
    if training.digest != prepared["training"]["partition_digest"]:
        raise ValueError("training source changed between independent preparations")
    vocabulary = fit_sequence_vocabulary(
        training,
        min_document_frequency=1,
        max_features=10_000,
        max_turn_tokens=128,
        long_turn_policy="head",
    )
    return {
        "digest": vocabulary.digest,
        "size": vocabulary.size,
        "documents": vocabulary.documents,
        "tokenizer_version": vocabulary.tokenizer_version,
    }


def validate_fit_summary(
    summary: dict[str, Any],
    protocol: dict[str, Any],
    prepared: dict[str, Any],
    vocabulary: dict[str, Any],
) -> None:
    helper = _helper()
    helper.validate_fit_summary(summary, protocol, prepared)
    for field in (
        "variant",
        "numerical_version",
        "initialization_version",
        "main_initialization_version",
        "pooling_limits",
        "training_limits",
    ):
        if helper._canonical(summary[field]) != helper._canonical(protocol[field]):
            raise ValueError("fitted ablation differs from its protocol: " + field)
    if summary["format"] != "turnscope.neural-ablation-forecast.v1":
        raise ValueError("fitted candidate uses a different model family")
    for field, expected in (("vocabulary_size", vocabulary["size"]), ("fixed_pad_parameters", 64)):
        if type(summary[field]) is not int or summary[field] != expected:
            raise ValueError("fitted ablation inventory differs from the admitted input")
    if summary["vocabulary_digest"] != vocabulary["digest"]:
        raise ValueError("fitted vocabulary differs from the independent training-only pin")
    architecture = NeuralArchitecture(vocabulary["size"])
    expected_count = sum(
        math.prod(shape)
        for shape in ablation_parameter_shapes(protocol["variant"], architecture).values()
    )
    for field, expected in (
        ("parameter_count", expected_count),
        ("canonical_main_parameter_count", architecture.parameter_count),
    ):
        if type(summary[field]) is not int or summary[field] != expected:
            raise ValueError("fitted parameter count differs from the closed active inventory")
    if helper._canonical(summary["reference_architecture"]) != helper._canonical(
        architecture.to_dict()
    ):
        raise ValueError("fitted reference architecture differs from the fixed dimensions")


def _publication(path: Path) -> tuple[str, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 64 * 1024 * 1024:
        raise ValueError("private candidate must be a bounded regular artifact")
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            count += len(block)
            if count > 64 * 1024 * 1024:
                raise ValueError("private candidate changed or exceeds artifact bounds")
            digest.update(block)
    after = path.lstat()
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        count,
        after.st_mtime_ns,
    ):
        raise ValueError("private candidate changed during verification")
    return digest.hexdigest(), count


def fit_candidate(archive: Path, directory: Path, *, variant: str, seed: int) -> dict[str, Any]:
    """One separately trained control; official-test access is source audit only."""
    import turnscope

    protocol = fit_protocol(variant, seed)
    helper = _helper()
    adapter = helper._adapter()
    directory = Path(directory).absolute()
    resolved = directory.resolve()
    for protected in (
        Path(__file__).resolve().parents[1],
        Path(turnscope.__file__).resolve().parent,
    ):
        if resolved == protected or protected in resolved.parents:
            raise ValueError("private output must be outside the checkout and imported package")
    if Path(archive).resolve() == resolved:
        raise ValueError("source archive and private output directory must differ")
    before = source_bindings()
    directory.mkdir()  # Existing/failed attempts must never be overwritten or resumed.
    warnings = []

    def publish(payload: dict[str, Any], name: str) -> None:
        warning = publish_neural_report(payload, directory / name)
        if warning:
            warnings.append({"file": name, "warning": warning})

    publish(
        {
            "protocol": protocol,
            "protocol_sha256": helper._sha(helper._canonical(protocol)),
            "source_sha256": before,
        },
        "fit-plan.json",
    )
    start = time.perf_counter()
    partitions = adapter.load_neural_cga(archive)
    prepared = partitions.prepared_audit()
    source = partitions.audit_dict()
    if (
        source.get("source_archive_verified") is not True
        or source.get("archive_sha256") != protocol["source_partition_protocol"]["archive_sha256"]
        or helper._canonical(source["protocol"])
        != helper._canonical(protocol["source_partition_protocol"])
        or source.get("protocol_sha256") != helper._sha(helper._canonical(source["protocol"]))
    ):
        raise ValueError(
            "source audit does not bind the fixed verified archive and partition policy"
        )
    vocabulary = _vocabulary_pin(partitions, prepared)
    parsed_seconds = time.perf_counter() - start
    publish(
        {
            "source": source,
            "prepared": prepared,
            "vocabulary": vocabulary,
            "load_and_audit_seconds": parsed_seconds,
        },
        "source-audit.json",
    )
    if source_bindings() != before:
        raise RuntimeError("bound source changed before fitting; no optimization attempted")
    model = AblationEventForecaster(
        config=AblationForecastConfig(variant),
        training_config=NeuralTrainingConfig(seed=seed),
        policy=SequencePolicy(),
        data_limits=SequenceLimits(),
        numeric_limits=NeuralNumericLimits(),
        pooling_limits=AblationPoolingLimits(),
        training_limits=AblationTrainingLimits(),
    )
    torch = importlib.import_module("torch")

    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    start = time.perf_counter()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        model.fit(
            partitions.training,
            partitions.validation,
            policy_validation=partitions.policy_validation,
        )
    finally:
        try:
            torch.set_num_threads(old_threads)
        finally:
            torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn)
    fit_seconds = time.perf_counter() - start
    summary = model.training_summary
    validate_fit_summary(summary, protocol, prepared, vocabulary)
    if source_bindings() != before:
        raise RuntimeError("bound source changed during fitting; no model published")
    target = directory / "private-model.tsa"
    saved = save_ablation_forecaster(model, target)
    saved_hash, saved_bytes = _publication(target)
    if (
        Path(saved.path).resolve() != target.resolve()
        or type(saved.bytes_written) is not int
        or saved.sha256 != saved_hash
        or saved.bytes_written != saved_bytes
    ):
        raise RuntimeError("publication receipt differs from the actual private artifact")
    restored = load_ablation_forecaster(target)
    if restored.digest != model.digest or helper._canonical(
        restored.training_summary
    ) != helper._canonical(summary):
        raise RuntimeError("private ablation did not roundtrip its complete fitted state")
    if _publication(target) != (saved_hash, saved_bytes):
        raise RuntimeError("private artifact changed during reload; no completed receipt")
    if source_bindings() != before:
        raise RuntimeError("bound source changed during publication; private model may remain")
    receipt = {
        "format": "turnscope.neural-cga-ablation-candidate-receipt.v1",
        "candidate_fit_completed": True,
        "final_evaluation_completed": False,
        "official_test_predictions_produced": False,
        "protocol_sha256": helper._sha(helper._canonical(protocol)),
        "source_sha256": before,
        "variant": variant,
        "seed": seed,
        "source": source,
        "prepared": prepared,
        "vocabulary": vocabulary,
        "publication": asdict(saved),
        "model_digest": model.digest,
        "training": summary,
        "fit_seconds": fit_seconds,
        "source_load_and_audit_seconds": parsed_seconds,
        "timing_scope": (
            "fit includes optimization and policy selection; excludes source/audit and save/reload"
        ),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "report_cleanup_warnings": list(warnings),
        "real_data_quality_claimed": False,
        "whole_repository_parity_claimed": False,
        "remaining_before_quality_acceptance": protocol["required_later_comparisons"],
        "privacy": (
            "private vocabulary/weights stay local; "
            "no source records or test predictions are exported"
        ),
    }
    # A failure here leaves a private artifact; it does not roll back optimization.
    warning = publish_neural_report(receipt, directory / "candidate-receipt.json")
    if warning:
        with suppress(OSError, ValueError, UnicodeError):
            print("Candidate receipt published; private temporary cleanup failed.", file=sys.stderr)
    return receipt


def main(argv: list[str] | None = None) -> int:
    options: dict[str, Any] = {}
    if sys.version_info >= (3, 14):
        options = {"color": False, "formatter_class": partial(argparse.HelpFormatter, color=False)}
    parser = argparse.ArgumentParser(description=__doc__, **options)
    parser.add_argument("archive", type=Path)
    parser.add_argument("directory", type=Path, help="new private directory outside the checkout")
    parser.add_argument("--variant", choices=ABLATION_VARIANTS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = fit_candidate(args.archive, args.directory, variant=args.variant, seed=args.seed)
        publish_neural_report(receipt, None)
    except (OSError, ValueError, RuntimeError, ImportError):
        # No exception text, paths, source records or weights in routine CLI diagnostics.
        with suppress(OSError, ValueError, UnicodeError):
            print(
                "Ablation command failed; private output may remain. "
                "Inspect local evidence before any retry.",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
